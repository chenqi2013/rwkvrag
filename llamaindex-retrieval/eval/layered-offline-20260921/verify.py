"""CPU-only audit and exact-input replay index; no service or model calls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from llamaindex_retrieval.offline_replay import audit_assessment_ids, extract_call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--focus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Exclusive creation prevents overwriting an earlier experiment.
    args.output.mkdir(parents=True, exist_ok=False)
    focused = {c["index"] for c in json.loads(args.focus.read_text())["cases"]}
    stages, issues, results = Counter(), Counter(), Counter()
    errors, findings, replay, hashes = [], [], [], []
    for arm in ("zero", "2000"):
        for index in range(160):
            directory = args.source_root / arm / f"{index:04d}"
            result_path = directory / "result.json"
            if not result_path.exists():
                errors.append({"path": str(result_path), "error": "missing result"})
                continue
            result = json.loads(result_path.read_text())
            results[arm + ":" + result["status"]] += 1
            hashes.append((str(result_path.relative_to(args.source_root)), hashlib.sha256(result_path.read_bytes()).hexdigest()))
            for path in sorted((directory / "traces").glob("*.json")):
                rel = str(path.relative_to(args.source_root))
                raw = path.read_bytes()
                h = hashlib.sha256(raw).hexdigest()
                hashes.append((rel, h))
                trace = json.loads(raw)
                stages[arm + ":" + trace["stage"]] += 1
                try:
                    call = extract_call(trace)
                except (ValueError, KeyError, TypeError) as exc:
                    errors.append({"path": rel, "error": str(exc)})
                    continue
                for issue in audit_assessment_ids(trace):
                    issues[arm + ":" + issue] += 1
                    findings.append({"path": rel, "trace_sha256": h, "issue": issue,
                                     "empty_input_evidence": not trace["evidence_ids"]})
                if index in focused:
                    # Bind full input via verified trace; avoid duplicating HTTP/State tokens.
                    call.pop("payload")
                    call["historical_state_binding"].pop("read_ref", None)
                    call["historical_state_binding"].pop("write_ref", None)
                    replay.append({"path": rel, "trace_sha256": h, "index": index,
                                   "arm": arm, **call})
    # Inputs are not pairs of differing upstream prompts. Only zero-arm nodes
    # are eligible as a common input for a later fresh-State A/B experiment.
    common = [r for r in replay if r["arm"] == "zero" and r["stage"] == "assess"]
    changed = [p for p, h in hashes if hashlib.sha256((args.source_root / p).read_bytes()).hexdigest() != h]
    report = {
        "status": "PASS" if not errors and not changed else "FAIL",
        "scope": "offline transport/provenance only; no semantic accuracy measurement",
        "model_calls": 0, "training_updates": 0, "result_count": sum(results.values()),
        "trace_count": sum(stages.values()), "stages": dict(stages), "results": dict(results),
        "assessment_issues": dict(issues), "transport_errors": errors, "changed_source_files": changed,
        "focused_trace_count": len(replay), "common_assess_inputs": len(common),
        "r0_three_repetitions_calls": 3 * sum(r["arm"] == "zero" for r in replay),
        "r1_ab_ba_calls": 4 * len(common),
        "execution_ready": False,
        "remaining": ["fresh engine/model/State bindings", "reviewed semantic gold", "frozen execution budget"],
    }
    for name, data in [("REPORT.json", report), ("FOCUSED-INPUTS.json", replay),
                       ("ASSESS-FINDINGS.json", findings), ("SOURCE-HASHES.json", hashes)]:
        (args.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
