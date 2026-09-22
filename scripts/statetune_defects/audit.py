"""Read immutable failures; deduplicate review work without manufacturing gold labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    raw = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def payload(call):
    try:
        value = json.loads(call["messages"][0]["content"].split("\n")[-1])
        return value if isinstance(value, dict) else {}
    except (KeyError, IndexError, TypeError, ValueError):
        return {}


def scope_observations(call):
    """Literal observations only. Matching a value is NOT semantic correctness."""
    inp = payload(call)
    try:
        answer = json.loads(call["raw_text"])
        texts = list(inp["evidence"].values())
        quote, value, scope = (answer.get(k) for k in ["quote", "value", "source_scope"])
        return {
            "quote_is_literal": isinstance(quote, str) and bool(quote) and any(quote in s for s in texts),
            "value_is_literal_in_quote": isinstance(value, str) and bool(value) and isinstance(quote, str) and value in quote,
            "scope_equals_title": bool(inp.get("source_title")) and scope == inp["source_title"],
            "scope_equals_object": bool(inp.get("object")) and scope == inp["object"],
            "semantic_correctness": "not_inferred",
        }
    except (KeyError, AttributeError, TypeError, ValueError):
        return {"inspection_error": True}


def family(error):
    if error == "model stage did not complete: length":
        return "termination"
    if error:
        return "evidence_binding"
    return "unclassified_execution"  # Never invent a model cause for missing diagnostics.


def group_jobs(defects):
    groups = {}
    for d in defects:
        if not d.get("call"):
            continue
        # Aggregate repeated tasks over the SAME evidence bundle, retaining all observations.
        key = (d["error"], d["material_fingerprint"])
        g = groups.setdefault(key, {"id": digest(key), "error": d["error"],
            "material_fingerprint": d["material_fingerprint"], "occurrences": [], "distinct_prompts": {},
            "diagnostic_only": True, "gold_target": None, "review_status": "unreviewed"})
        g["occurrences"].append({"case": d["case"], "call_id": d["call"]["call_id"],
                                  "raw_output_sha256": digest(d["call"]["raw_text"].encode())})
        g["distinct_prompts"].setdefault(d["prompt_sha256"], {
            "case": d["case"], "call_id": d["call"]["call_id"], "object": d["input"].get("object"),
            "field": d["input"].get("field"), "input": d["input"],
            "observed_output": d["call"]["raw_text"], "gold_target": None})
    return [dict(g, distinct_prompts=list(g["distinct_prompts"].values()))
            for _, g in sorted(groups.items())]


def audit(run):
    defects, bindings, cases, purpose_counts = [], {}, [], Counter()
    paths = sorted(run.glob("[0-9][0-9][0-9].json"))
    if [p.stem for p in paths] != [f"{i:03d}" for i in range(36)]:
        raise ValueError("Expected all 36 frozen questions; refusing partial audit")
    for path in paths:
        raw = path.read_bytes()
        bindings[str(path.relative_to(ROOT))] = digest(raw)
        response = json.loads(raw)["response"]
        calls = response["generation"]["model_calls"]
        by_id = {c["call_id"]: c for c in calls}
        if len(by_id) != len(calls):
            raise ValueError("Duplicate call ID in question")
        for c in calls:
            purpose = c.get("purpose", c["stage"])
            key = "fact_quantity_binding" if purpose.endswith(":quantity_binding") else purpose.split(":")[0]
            purpose_counts[key] += 1
        graph = response["retrieval"]["funnel"]
        cases.append({"ordinal": int(path.stem), "model_calls": len(calls),
            "final_source_count": len(response.get("sources", [])),
            "writer_status": next(c["status"] for c in reversed(calls) if c["stage"] == "writer")})
        for index, failure in enumerate(graph.get("failures", [])):
            call = by_id.get(failure.get("call_id"))
            inp = payload(call) if call else {}
            # Evidence IDs can vary without changing material. Exact text bundles only;
            # no claim of document-level/near-duplicate separation.
            material = (sorted(inp["evidence"].values()) if isinstance(inp.get("evidence"), dict)
                        else inp or {"unparsed_prompt": call.get("prompt", "") if call else failure})
            error = failure.get("error", "")
            defects.append({"case": int(path.stem), "failure_index": index, "error": error,
                "family": family(error), "raw_failure": failure,
                "prompt_sha256": call.get("prompt_sha256") if call else None,
                "material_fingerprint": digest(material), "input": inp,
                "scope_observations": scope_observations(call) if call and error == "scope not verbatim in supplied source" else None,
                "call": {k: call[k] for k in ["call_id", "purpose", "messages", "prompt", "raw_text", "status"] if k in call} if call else None,
                "gold_target": None, "review_status": "unreviewed", "partition": "diagnostic_only"})
    by_error = defaultdict(list)
    for d in defects:
        by_error[d["error"]].append(d)
    summary = {"questions": len(cases), "model_calls": sum(purpose_counts.values()),
        "purpose_counts": dict(purpose_counts), "failure_records": len(defects),
        "failure_records_with_trace": sum(d["call"] is not None for d in defects),
        "unique_failed_call_ids": len({d["call"]["call_id"] for d in defects if d["call"]}),
        "errors": {error: {"occurrences": len(items), "affected_questions": sorted({d["case"] for d in items}),
            "distinct_prompts": len({d["prompt_sha256"] for d in items if d["prompt_sha256"]}),
            "distinct_exact_material_bundles": len({d["material_fingerprint"] for d in items})}
            for error, items in sorted(by_error.items())},
        "scope_literal_observations": {key: sum(bool((d["scope_observations"] or {}).get(key)) for d in defects)
            for key in ["quote_is_literal", "value_is_literal_in_quote", "scope_equals_title", "scope_equals_object"]},
        "cases": cases, "semantic_error_rate": None,
        "limits": ["First validation failure can mask other errors; these are not exclusive semantic causes.",
                   "Exact material bundles are not independent documents or template families.",
                   "Historical regression inputs are diagnostic-only, never unseen evaluation data.",
                   "No failed model output is a gold target; all generation jobs require annotation."]}
    return defects, summary, bindings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = ROOT / "data/quality-runs/typed-funnel-v10-integration-20260921/run1"
    defects, summary, bindings = audit(run)
    jobs = group_jobs(defects)
    summary["annotation_jobs"] = len(jobs)
    args.out.mkdir(parents=True, exist_ok=False)
    for name, value in [("SUMMARY.json", summary), ("SOURCE-PINS.json", bindings)]:
        with (args.out / name).open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
    for name, rows in [("FAILURES.jsonl", defects), ("ANNOTATION-JOBS.jsonl", jobs)]:
        with (args.out / name).open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print({k: summary[k] for k in ["questions", "model_calls", "failure_records", "annotation_jobs", "scope_literal_observations"]})


if __name__ == "__main__":
    main()
