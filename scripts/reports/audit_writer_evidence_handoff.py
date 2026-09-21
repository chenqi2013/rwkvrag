"""Mechanical audit only; output equality is not a semantic quality score."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = "writer-evidence-handoff-20260921"


def main():
    frozen = ROOT / "llamaindex-retrieval/eval" / NAME
    run = ROOT / "data/quality-runs" / NAME
    for path, digest in json.loads((frozen / "PINS.json").read_text()).items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    rows = json.loads((frozen / "INPUTS.json").read_text())
    records = []
    comparisons = []
    for row in rows:
        outputs = {}
        for rnd in [1, 2]:
            for arm in ["baseline", "candidate"]:
                d = json.loads((run / "run1" / f"{row['ordinal']:03d}-{arm}-r{rnd}.json").read_text())
                assert (d["ordinal"], d["id"], d["round"], d["arm"]) == (row["ordinal"], row["id"], rnd, arm)
                assert d["raw_text"] == d["trace"]["raw_text"]
                assert hashlib.sha256(d["raw_text"].encode()).hexdigest() == d["trace"]["raw_text_sha256"]
                outputs[(arm, rnd)] = d["raw_text"]
                records.append(d)
        comparisons.append({"ordinal": row["ordinal"], "id": row["id"],
            "target": row["baseline_prompt"] != row["candidate_prompt"],
            "baseline_rounds_equal": outputs[("baseline", 1)] == outputs[("baseline", 2)],
            "candidate_rounds_equal": outputs[("candidate", 1)] == outputs[("candidate", 2)],
            "paired_equal": {str(r): outputs[("baseline", r)] == outputs[("candidate", r)] for r in [1, 2]},
            "baseline_historical_equal": {str(r): outputs[("baseline", r)] == row["previous_answer"] for r in [1, 2]}})
    result = {"recorded": len(records), "frozen_pins_verified": True,
        "baseline_or_control_input_checks": dict(Counter(str(d["baseline_input_aligned"]) for d in records)),
        "groups": {f"{arm}-r{rnd}": {
            "statuses": dict(Counter(d["status"] for d in records if d["arm"] == arm and d["round"] == rnd)),
            "total_elapsed_s": sum(d["trace"]["elapsed_ms"] / 1000 for d in records if d["arm"] == arm and d["round"] == rnd)}
            for arm in ["baseline", "candidate"] for rnd in [1, 2]},
        "comparisons": comparisons, "semantic_review": "Separate implementer review; no automatic correctness inference."}
    with (run / "STRUCTURAL-AUDIT.json").open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print({k: v for k, v in result.items() if k != "comparisons"})


if __name__ == "__main__":
    main()
