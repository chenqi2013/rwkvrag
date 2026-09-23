"""Summarize a frozen label-first pilot and its fail-closed admission receipt."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.run / "RUN.json").read_text())
    summary = json.loads((args.run / "SUMMARY.json").read_text())
    files = sorted((args.run / "results").glob("*.json"))
    if len(files) != len(run["job_ids"]) or {f.stem for f in files} != set(run["job_ids"]):
        raise ValueError("incomplete result files")
    rows = [row for file in files for row in json.loads(file.read_text())["rows"]]
    tokens = [json.loads(line) for line in (args.run / "PLAN-TOKENS.jsonl").read_text().splitlines()]
    audit = json.loads((args.run / "ADMISSION.json").read_text())
    if (not rows or len(rows) != len(tokens) or len(rows) != summary["question_rows"] or
            len({row["id"] for row in rows}) != len(rows) or audit["admitted"]):
        raise ValueError("pilot rows or admission result mismatch")
    for token in tokens:
        count = token["prompt_tokens"]
        if (token["input_ids"][-1] != 0 or token["labels"][:count] != [-100] * count or
                token["labels"][count:] != token["input_ids"][count:]):
            raise ValueError("prompt mask or EOS mismatch")
    receipt = {"protocol": "retrieval_label_first_pilot_v1",
               "run_sha256": digest(args.run / "RUN.json"),
               "summary_sha256": digest(args.run / "SUMMARY.json"),
               "result_files_sha256": sha256("".join(f.name + digest(f) for f in files).encode()).hexdigest(),
               "cases_sha256": digest(args.run / "CASES.jsonl"),
               "tokens_sha256": digest(args.run / "PLAN-TOKENS.jsonl"),
               "admission_sha256": digest(args.run / "ADMISSION.json"),
               "jobs": len(files), "possible_items": len(files) * 8,
               "structural_plans": summary["plan_rows"], "bound_questions": len(rows),
               "max_token_length": max(len(row["input_ids"]) for row in tokens),
               "coverage": dict(Counter(row["plan"]["coverage"] for row in rows)),
               "object_counts": dict(Counter(len(row["plan"]["objects"]) for row in rows)),
               "kinds": dict(Counter(row["kind"] for row in rows)),
               "admitted": False, "train_families": audit["train_families"],
               "independent_semantic_review": False,
               "training_started": False,
               "limits": "Literal label checks and token integrity only; source cohort relevance and semantics failed pilot review."}
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(receipt, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({k: receipt[k] for k in ("structural_plans", "bound_questions", "max_token_length",
                                                 "coverage", "admitted")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
