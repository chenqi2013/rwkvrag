"""Supplemental oracle-span relation isolation after staged-v1 contract failures.

Same frozen relation prompt and 256-token budget; no retries or output repairs.
Hand-selected verbatim spans isolate relationship reasoning from extraction.
This is a diagnostic on exposed cases, not a new blind holdout.
"""
import argparse
import json
from pathlib import Path
import time
from run_probe import HERE, ROOT, digest, relation_prompt, request, save, strict_json, task, validate_relation, wrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads((HERE / "PLAN-relations.json").read_text())
    for rel, sha in plan["bindings"].items():
        assert digest((ROOT / rel).read_bytes()) == sha, rel
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "RUN.json", {"endpoint": args.endpoint, "model": args.model,
        "plan_sha256": digest((HERE / "PLAN-relations.json").read_bytes()), "started_at": time.time()})
    results = []
    for line in (HERE / "cases.jsonl").read_text().splitlines():
        case = json.loads(line)
        if case["group"] != "controlled":
            continue
        out = args.output / case["id"]
        out.mkdir()
        atoms = [{"source_label": f"资料 {q['source']}", "quote": q["quote"]} for q in case["oracle_quotes"]]
        for q in case["oracle_quotes"]:
            assert q["quote"] in case["materials"][q["source"] - 1]["snippet"]
        prompt = wrap(relation_prompt(task(case), atoms))
        body = {"model": args.model, "contents": [prompt], "max_tokens": 256,
            "top_p": 0, "alpha_presence": 0, "alpha_frequency": 0,
            "stream": False, "stop_tokens": [0], "state_id": None}
        save(out / "relation.started.json", {"request": body, "prompt_sha256": digest(prompt.encode())})
        result = {"id": case["id"], "expected_relation": case["expected_relation"]}
        tick = time.monotonic()
        try:
            response = request(args.endpoint, body)
            save(out / "relation.completed.json", {"response": response, "elapsed_seconds": time.monotonic() - tick})
            choice = response["choices"][0]
            result.update(raw_text=choice["message"]["content"], finish_reason=choice["finish_reason"])
            if choice["finish_reason"] != "stop":
                raise ValueError("relation exhausted generation budget")
            parsed = validate_relation(result["raw_text"], {a["source_label"] for a in atoms})
            result.update(status="completed", parsed=parsed, relation_label_match=parsed["status"] == case["expected_relation"])
        except Exception as exc:
            result.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        result["elapsed_seconds"] = time.monotonic() - tick
        save(out / "RESULT.json", result)
        results.append(result)
        print(json.dumps({k: result[k] for k in ("id", "status")}), flush=True)
    save(args.output / "SUMMARY.json", {"results": results,
        "semantic_scoring": "label match alone is not full correctness; manually inspect rationale and sources"})


if __name__ == "__main__":
    main()
