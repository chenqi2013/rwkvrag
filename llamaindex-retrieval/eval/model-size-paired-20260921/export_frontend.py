"""Export immutable answers into a read-only frontend dataset; no model calls."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("answers", type=Path, help="ALL-ANSWERS.json from export_results.py")
    parser.add_argument("review", type=Path, help="review1/REVIEW.json")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.answers.read_text())
    reviews = {r["key"]: r for r in json.loads(args.review.read_text())}
    cases = []
    for case in data["inputs"]:
        row = {k: case[k] for k in ["ordinal", "id", "category", "question", "history", "evidence"]}
        row["answers"] = []
        for answer in data["answers"]:
            if answer["ordinal"] != case["ordinal"]:
                continue
            review = reviews[f'{answer["arm"]}/{answer["round"]}/{answer["ordinal"]}']
            assert hashlib.sha256(answer["raw_text"].encode()).hexdigest() == review["raw_answer_sha256"] == answer["raw_text_sha256"]
            assert review["input_sha256"] == case["prompt_sha256"]
            row["answers"].append({
                **{k: answer[k] for k in ["arm", "round", "raw_text", "finish_reason", "elapsed_ms", "raw_text_sha256"]},
                "review": review["review"], "strict_pass": review["strict_pass"],
            })
        assert {(a["arm"], a["round"]) for a in row["answers"]} == {(m, n) for m in ["2.9b", "7.2b"] for n in [1, 2]}
        assert len(row["answers"]) == 4
        cases.append(row)
    assert len(cases) == 188
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "scope": "Fixed-material replay, zero State, non-independent implementer review; not production retrieval",
        "cases": cases,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
