"""Bind teacher pilot counts to immutable raw files without publishing replies."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from generate_plan_drafts import compile_draft_v3


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def score_run(path, jobs, *, strict):
    run = json.loads((path / "RUN.json").read_text())
    summary = json.loads((path / "SUMMARY.json").read_text())
    files = sorted((path / "results").glob("*.json"))
    if len(files) != 12 or len(run["job_ids"]) != 12 or set(run["job_ids"]) != {f.stem for f in files}:
        raise ValueError("pilot run is incomplete")
    structural = sum(len(json.loads(f.read_text())["rows"]) for f in files)
    if structural != summary["rows"] or summary["jobs"] != 12 or summary["stopped"]:
        raise ValueError("pilot receipts disagree")
    result = {"run_sha256": digest(path / "RUN.json"),
              "summary_sha256": digest(path / "SUMMARY.json"),
              "results_sha256": sha256("".join(f.name + digest(f) for f in files).encode()).hexdigest(),
              "calls": 12, "possible_items": 96, "structural_rows": structural,
              "conservative_cost_usd": summary["cost_upper_usd"],
              "independent_review": False, "training_exported": False}
    if strict:
        rows, rejected = [], []
        for file in files:
            saved = json.loads(file.read_text())
            accepted, failures = compile_draft_v3(jobs[saved["job_id"]], saved["draft"])
            rows.extend(accepted)
            rejected.extend(failures)
        result.update(strict_rows=len(rows), strict_failures=len(rejected),
                      full_object_mentions=sum(all(obj.casefold() in row["question"].casefold()
                                                   for obj in row["plan"]["objects"]) for row in rows),
                      kinds=dict(Counter(row["kind"] for row in rows)),
                      coverage=dict(Counter(row["plan"]["coverage"] for row in rows)),
                      object_counts=dict(Counter(len(row["plan"]["objects"]) for row in rows)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    jobs = {row["id"]: row for row in map(json.loads, args.jobs.read_text().splitlines())}
    report = {"protocol": "retrieval_plan_teacher_pilots_v1",
              "jobs_sha256": digest(args.jobs),
              "strict_scorer_sha256": digest(Path(__file__).resolve().parent / "generate_plan_drafts.py"),
              "runs": {name: score_run(args.root / name, jobs, strict=name != "pilot-00-12")
                       for name in ("pilot-00-12", "pilot-v2-00-12", "pilot-v3-00-12")},
              "limits": "Structural checks only; no independent semantic review or training."}
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({name: {"structural": row["structural_rows"],
                             "strict": row.get("strict_rows")}
                      for name, row in report["runs"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
