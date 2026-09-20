"""Export completed frozen experiments for review; never changes model decisions."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = json.loads((args.run / "SUMMARY.json").read_text())
    bindings = json.loads((args.run / "BINDINGS.json").read_text())
    assert len(reports) == len(bindings["profiles"])
    assert all(r["execution_status"] == "completed" for r in reports)
    args.output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))

    save("BINDINGS.json", bindings)
    save("SUMMARY.json", reports)
    cases, decisions, calls, digests, timing, health = {}, {}, [], {}, {}, {}
    for report in reports:
        name = report["profile"]["id"]
        folder = args.run / name
        rows = json.loads((folder / "ROWS.json").read_text())
        assert len(rows) == report["expected_cases"]
        decisions[name] = {r["case_id"]: r for r in rows}
        timing[name] = {"observed_median_elapsed_ms": median(r["elapsed_ms"] for r in rows),
                        "observed_total_elapsed_ms": sum(r["elapsed_ms"] for r in rows)}
        health[name] = json.loads((folder / "HEALTH.json").read_text())
        for row in rows:
            path = folder / (row["case_id"] + ".json")
            original = json.loads(path.read_text())
            case = original["case"]
            if case["id"] in cases:
                assert cases[case["id"]] == case
            cases[case["id"]] = case
            assert original["result"] == row
            digests[str(path)] = sha256(path.read_bytes()).hexdigest()
            calls.append({"profile": name, **original})
    save("TIMING.json", timing)
    save("HEALTH.json", health)
    save("TRACE-DIGESTS.json", digests)
    # Preserve raw_output/prompt/trace fields exactly; omit HTTP receipts from the export.
    (args.output / "calls.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in calls) + "\n")
    queue = []
    for identity, case in cases.items():
        rows = {name: result[identity] for name, result in decisions.items()}
        failed = [name for name, r in rows.items() if r["status"] != "valid" or r["prediction"] != case["expected"]]
        if failed:
            queue.append({"case": case, "failed_profiles": failed, "decisions": rows,
                          "review_status": "needs_independent_label_review", "eligible_for_training": False})
    save("REVIEW-QUEUE.json", queue)
    save("EXPORT.json", {"calls": len(calls), "distinct_cases": len(cases),
                          "review_cases": len(queue), "raw_outputs_modified": False,
                          "http_receipts_exported": False})
    print(json.dumps({"calls": len(calls), "cases": len(cases), "review_cases": len(queue)}))


if __name__ == "__main__":
    main()
