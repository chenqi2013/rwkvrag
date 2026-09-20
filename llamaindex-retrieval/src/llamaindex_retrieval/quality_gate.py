"""Audit frozen native-material smoke runs; semantic judgments require explicit review.

This never calls a model, edits an answer, or treats execution as semantic success.
Exit codes: 0 = this development suite passed, 1 = failed, 2 = invalid artifacts,
3 = semantic review pending. Passing does not certify full RAG or unseen questions.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import median

from .citation_audit import audit_citations


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return sha256(encode(value)).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"invalid JSON constant: {value}")

    return json.loads(path.read_bytes(), object_pairs_hook=unique, parse_constant=invalid)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_file(root, relative, expected_sha):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "artifact path escapes run directory")
    require(sha256(path.read_bytes()).hexdigest() == expected_sha,
            f"artifact hash mismatch: {relative}")
    return path


def load_run(run):
    manifest = read_json(run / "manifest.json")
    require(manifest["schema"] == "rwkvrag-native-material-smoke-v1",
            "only frozen native-material smoke runs are supported")
    fixture_path = checked_file(run, "frozen/fixtures.jsonl", manifest["fixture_sha256"])
    # The frozen fixture has already been validated by the runner. Recheck its
    # identity, denominator and source quotes independently of mutable dev files.
    fixtures = [json.loads(line) for line in fixture_path.read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in fixtures]
    require(ids == [f"smoke_{i:03}" for i in range(1, 9)] and ids == manifest["case_ids"]
            and len(ids) == manifest["planned_cases"], "fixture denominator mismatch")
    split = manifest.get("evaluation_split", "development_smoke_not_blind")
    require(split in {"development_smoke_not_blind", "frozen_holdout"}
            and all(row.get("phase") == split for row in fixtures), "evaluation split mismatch")
    for row in fixtures:
        require(re.fullmatch(r"smoke_[0-9]{3}", row["id"]) is not None, "invalid case ID")
        materials = {m["id"]: m for m in row["materials"]}
        require(len(materials) == len(row["materials"]), "duplicate material ID")
        require(digest(row["materials"]) == row["provenance"]["materials_sha256"],
                "material provenance mismatch")
        require(len({f["id"] for f in row["expected_facts"]}) == len(row["expected_facts"]),
                "duplicate fact ID")
        for fact in row["expected_facts"]:
            require(bool(fact["supporting_quotes"]), "fact has no supporting quote")
            for quote in fact["supporting_quotes"]:
                text = materials[quote["material_id"]]["snippet"]
                start, end = quote["start"], quote["end"]
                require(type(start) is int and type(end) is int and 0 <= start < end <= len(text)
                        and text[start:end] == quote["quote"], "invalid gold source span")
    checked_file(run, "frozen/run_smoke.py", manifest["runner_sha256"])
    require(digest(manifest["production_sources"]) == manifest["production_sources_sha256"],
            "source inventory mismatch")
    for item in manifest["production_sources"]:
        checked_file(run, "frozen/src/" + item["path"], item["sha256"])
    records = read_json(run / "results.json")
    require([row["id"] for row in records] == ids, "result denominator/order mismatch")
    summary = read_json(run / "summary.json")
    checked_file(run, "results.json", summary["results_sha256"])
    require(summary["source_unchanged"] is True, "source changed during evaluation")
    manifest_sha = sha256((run / "manifest.json").read_bytes()).hexdigest()
    for fixture, record in zip(fixtures, records, strict=True):
        require(read_json(run / "completed" / (record["id"] + ".json")) == record,
                "result differs from completed receipt")
        require(digest(record["model_traces"]) == record["model_traces_sha256"],
                "model trace hash mismatch")
        if record["attempted"]:
            started = read_json(run / "started" / (record["id"] + ".json"))
            request = started["request"]
            require(started["manifest_sha256"] == manifest_sha, "request manifest mismatch")
            require(digest(request) == record["request_sha256"] == started["request_sha256"],
                    "request hash mismatch")
            require(request == {k: fixture[k] for k in ("id", "question", "history", "materials")},
                    "request differs from fixture or includes gold")
        if record["response"] is not None:
            require(record["attempted"] is True, "unattempted case contains a response")
            require(digest(record["response"]) == record["response_sha256"],
                    "response hash mismatch")
            require(record["response"]["sources"] == fixture["materials"],
                    "fixed-material response sources changed")
            require(record["response"]["generation"]["model_calls"] == record["model_traces"],
                    "response trace differs from recorded model calls")
            config = manifest["configuration"]
            if config["transport"] == "rwkvos_batch" and "writer_transport_protocol" in config:
                for call in record["model_traces"]:
                    if call.get("stage") == "writer" and call.get("completion_attempted"):
                        require(call.get("writer_prompt_protocol") == config["writer_transport_protocol"],
                                "Writer transport protocol differs from manifest")
                        require(call.get("state_id") == (config.get("writer_state_id") or config["state_id"]),
                                "Writer state differs from manifest")
    return manifest, fixtures, records, manifest_sha


def automatic_checks(record, model):
    response = record["response"]
    if not response:
        return {"execution_completed": False, "answer_contract_valid": False,
                "citation_labels_valid": False}
    generation = response["generation"]
    raw = generation.get("raw_model_answer")
    span = generation.get("answer_span")
    span_valid = (isinstance(raw, str) and isinstance(span, list) and len(span) == 2
                  and all(type(i) is int for i in span) and 0 <= span[0] <= span[1] <= len(raw))
    body = raw[span[0]:span[1]] if span_valid else ""
    sources = response["sources"]
    citations = audit_citations(body, sources)
    expected_map = {str(i): source["id"] for i, source in enumerate(sources, 1)}
    writers = [call for call in record["model_traces"] if call.get("stage") == "writer"]
    return {
        "execution_completed": (record["status"] == generation.get("status") == "completed"
                                and bool(writers) and writers[-1].get("status") == "completed"
                                and not generation.get("planner_fallback")
                                and all(call.get("status") == "completed"
                                        and not call.get("parse_error")
                                        for call in record["model_traces"])),
        "answer_contract_valid": (generation.get("pipeline") == "rwkv"
                                  and generation.get("model") == model
                                  and generation.get("output_mode") == "immutable"
                                  and generation.get("answer_modified") is False
                                  and response["answer"] == raw and bool(body.strip())
                                  and bool(writers) and writers[-1].get("raw_text") == raw),
        # Existence is checked here; whether a citation supports a claim is reviewed below.
        "citation_labels_valid": (span_valid and generation.get("citation_map") == expected_map
                                  and not citations["unknown_label_ids"] and not citations["invalid_labels"]),
    }


def review_template(manifest_sha, fixtures, records):
    return {
        "schema": "rwkvrag-material-review-v1", "manifest_sha256": manifest_sha,
        "reviewer": "", "reviewed_at": "",
        "cases": [{"id": fixture["id"], "response_sha256": record.get("response_sha256"),
                   "decision_correct": None, "no_unsupported_claims": None,
                   "no_uncontrolled_repetition": None, "notes": "",
                   "facts": [{"id": fact["id"], "correct": None, "citation_supported": None}
                             for fact in fixture["expected_facts"]]}
                  for fixture, record in zip(fixtures, records, strict=True)],
    }


def semantic_checks(review, fixture, record):
    if review is None:
        return None
    require(review["response_sha256"] == record.get("response_sha256"), "stale response review")
    facts = review["facts"]
    require([f["id"] for f in facts] == [f["id"] for f in fixture["expected_facts"]],
            "review fact denominator mismatch")
    values = [review[k] for k in ("decision_correct", "no_unsupported_claims",
                                 "no_uncontrolled_repetition")]
    values += [fact[k] for fact in facts for k in ("correct", "citation_supported")]
    if any(value is None for value in values):
        return None
    require(all(type(value) is bool for value in values), "review judgments must be booleans")
    require(isinstance(review["notes"], str) and bool(review["notes"].strip()),
            "completed review requires an evidence explanation")
    return {
        "facts_correct": all(fact["correct"] for fact in facts) if facts else None,
        "citations_supported": all(fact["citation_supported"] for fact in facts) if facts else None,
        "decision_correct": review["decision_correct"],
        "no_unsupported_claims": review["no_unsupported_claims"],
        "no_uncontrolled_repetition": review["no_uncontrolled_repetition"],
    }


def audit(run, review_path=None):
    manifest, fixtures, records, manifest_sha = load_run(run)
    reviews = {}
    if review_path:
        review = read_json(review_path)
        require(review["schema"] == "rwkvrag-material-review-v1"
                and review["manifest_sha256"] == manifest_sha, "review manifest mismatch")
        rows = review["cases"]
        require([row["id"] for row in rows] == manifest["case_ids"], "review denominator mismatch")
        require(all(isinstance(review[key], str) and review[key].strip()
                    for key in ("reviewer", "reviewed_at")), "review author and date required")
        reviews = {row["id"]: row for row in rows}
    cases = []
    for fixture, record in zip(fixtures, records, strict=True):
        checks = automatic_checks(record, manifest["configuration"]["model"])
        semantics = semantic_checks(reviews.get(record["id"]), fixture, record)
        failed = not all(checks.values()) or (
            semantics is not None and any(value is False for value in semantics.values()))
        cases.append({"id": record["id"], "status": record["status"],
                      "response_sha256": record.get("response_sha256"),
                      "automatic": checks, "semantic": semantics,
                      "decision": "fail" if failed else "pending_review" if semantics is None
                      else "pass"})
    counts = Counter(row["decision"] for row in cases)
    times = [row["elapsed_ms"] for row in records if row.get("elapsed_ms") is not None]
    metrics = {}
    for group in ("automatic", "semantic"):
        assessed = [row[group] for row in cases if row[group] is not None]
        keys = sorted({key for row in assessed for key in row})
        metrics[group] = {
            key: {"passed": sum(row[key] is True for row in assessed),
                  "assessed": sum(row[key] is not None for row in assessed), "total": len(cases)}
            for key in keys}
    report = {
        "schema": "rwkvrag-material-quality-gate-v1", "manifest_sha256": manifest_sha,
        "auditor_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "auditor_sources": {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                            for name in ("quality_gate.py", "citation_audit.py")},
        "review_sha256": sha256(review_path.read_bytes()).hexdigest() if review_path else None,
        "scope": ("fixed_material_holdout_only" if manifest.get("evaluation_split") == "frozen_holdout"
                  else "fixed_material_development_only"),
        "evaluation_split": manifest.get("evaluation_split", "development_smoke_not_blind"),
        "fixture_sha256": manifest["fixture_sha256"], "full_rag_verified": False,
        "planned_cases": len(fixtures), "reviewed_cases": sum(c["semantic"] is not None for c in cases),
        "passed_cases": counts["pass"], "failed_cases": counts["fail"],
        "pending_cases": counts["pending_review"],
        "decision": "fail" if counts["fail"] else "pending_review" if counts["pending_review"]
        else "pass",
        "status_counts": dict(Counter(row["status"] for row in records)),
        "model_calls": sum(len(row["model_traces"]) for row in records),
        "median_case_elapsed_ms": median(times) if times else None,
        "metric_unit": "case", "metrics": metrics,
        "configuration": manifest["configuration"], "cases": cases,
    }
    return report, review_template(manifest_sha, fixtures, records)


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        file.write(encode(value) + b"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--review-template", type=Path)
    args = parser.parse_args()
    try:
        report, template = audit(args.run, args.review)
        if args.output:
            write_new(args.output, report)
        if args.review_template:
            write_new(args.review_template, template)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(json.dumps({"decision": "invalid_artifacts", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return {"pass": 0, "fail": 1, "pending_review": 3}[report["decision"]]


if __name__ == "__main__":
    raise SystemExit(main())
