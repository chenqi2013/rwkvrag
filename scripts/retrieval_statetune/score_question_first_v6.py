"""Preserve the V6 pilot's joined draft rows and source/hash audit."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from llamaindex_retrieval.retrieval_plan import RetrievalPlanV1


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out-cases", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.run / "RUN.json").read_text())
    summary = json.loads((args.run / "SUMMARY.json").read_text())
    sources = json.loads(args.sources.read_text())
    if (run["sources_sha256"] != digest(args.sources) or
            sources.get("schema") != "retrieval_curated_sources_v4" or
            summary["stopped"] or summary["independent_review"] or
            summary["training_exported"]):
        raise ValueError("pilot binding or status invalid")
    by_repo = {source["repo"]: source for source in sources["sources"]}
    files = [args.run / "results" / f"{job_id}.json" for job_id in run["job_ids"]]
    if len(files) != summary["jobs"]:
        raise ValueError("pilot job count mismatch")
    rows, failures = [], Counter()
    for file in files:
        result = json.loads(file.read_text())
        for issue in result["question_failures"]:
            failures[f"question:{issue['error']}"] += 1
        for issue in result["plan_failures"]:
            failures["plan:structural"] += 1
        for row in result["rows"]:
            plan = RetrievalPlanV1.model_validate(row["plan"], strict=True)
            if (row["review"].get("accepted") is not False or
                    len(plan.objects) != len(row["source_families"]) or
                    len(plan.objects) != len(row["source_hashes"])):
                raise ValueError("row review or source length mismatch")
            for repo, family, source_hash in zip(plan.objects, row["source_families"],
                                                 row["source_hashes"], strict=True):
                source = by_repo[repo]
                if source["split"] != "train" or source["family"] != family or source["sha256"] != source_hash:
                    raise ValueError("row source identity mismatch")
            rows.append({**row, "provenance": {"run_sha256": digest(args.run / "RUN.json"),
                         "result_sha256": digest(file)}})
    if len(rows) != summary["joined_rows"] or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("pilot rows incomplete or duplicated")
    with args.out_cases.open("x", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"schema": "retrieval_question_first_pilot_v6",
              "run_sha256": digest(args.run / "RUN.json"),
              "summary_sha256": digest(args.run / "SUMMARY.json"),
              "result_files_sha256": sha256("".join(file.name + digest(file) for file in files).encode()).hexdigest(),
              "cases_sha256": digest(args.out_cases),
              "jobs": len(files), "possible_rows": len(files) * 8,
              "joined_rows": len(rows),
              "kinds": dict(Counter(row["kind"] for row in rows)),
              "object_counts": dict(Counter(len(row["plan"]["objects"]) for row in rows)),
              "failures": dict(failures), "independent_semantic_review": False,
              "admitted_training_rows": 0, "training_started": False,
              "limits": "Naturalness checked only by author spot sampling; 65 joined rows are unreviewed drafts."}
    with args.out_summary.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(json.dumps({"joined_rows": len(rows), "possible_rows": report["possible_rows"],
                      "kinds": report["kinds"], "failures": report["failures"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
