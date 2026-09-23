"""Package structurally valid teacher drafts with their immutable provenance.

No semantic acceptance is inferred. The output is inspectable candidate data,
not a StateTune training export.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata

from llamaindex_retrieval.retrieval_plan import RetrievalPlanV1

ROOT = Path(__file__).resolve().parents[2]
EXCLUSIONS = ROOT / "llamaindex-retrieval/statetune/retrieval-v1-20260923/EXCLUSIONS.json"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def package(run_paths, jobs_path, sources_path, expected_split="train"):
    jobs = [json.loads(line) for line in jobs_path.read_text().splitlines() if line.strip()]
    by_job = {job["id"]: job for job in jobs}
    if len(by_job) != len(jobs):
        raise ValueError("duplicate registered job ID")
    sources = json.loads(sources_path.read_text())
    by_repo = {row["repo"]: row for row in sources["sources"]}
    processed = set()
    rows, runs, failures = [], [], []
    for path in run_paths:
        run_file = path / "RUN.json"
        run = json.loads(run_file.read_text())
        summary = json.loads((path / "SUMMARY.json").read_text())
        if (run["sources_sha256"] != digest(sources_path) or
                run["jobs_sha256"] != digest(jobs_path) or
                run["split"] != expected_split or summary["stopped"] or
                summary["training_exported"] or summary["independent_review"]):
            raise ValueError(f"run bindings or status invalid: {path}")
        if len(run["job_ids"]) != summary["jobs"]:
            raise ValueError(f"run incomplete: {path}")
        runs.append({"path": str(path), "run_sha256": digest(run_file),
                     "summary_sha256": digest(path / "SUMMARY.json"),
                     "jobs": summary["jobs"], "questions": summary["questions"],
                     "plan_prompt_sha256": run["plan_prompt_sha256"],
                     "question_prompt_sha256": run["question_prompt_sha256"],
                     "script_sha256": run["script_sha256"]})
        for job_id in run["job_ids"]:
            if job_id in processed or job_id not in by_job:
                raise ValueError(f"duplicate or unregistered job: {job_id}")
            processed.add(job_id)
            file = path / "results" / f"{job_id}.json"
            result = json.loads(file.read_text())
            job = by_job[job_id]
            if result["job_id"] != job_id:
                raise ValueError("result ID mismatch")
            if result["status"] != "unreviewed_drafts":
                failures.append({"job_id": job_id, "error": result.get("error_type", "failed")})
            for row in result["rows"]:
                if (row["split"] != expected_split or row["id"].split(":")[0] != job_id or
                        row["source_families"] != job["source_families"] or
                        row["source_hashes"] != job["source_hashes"] or
                        row["review"].get("accepted") is not False or
                        row["review"].get("independent_of_author") is not False):
                    raise ValueError(f"candidate provenance mismatch: {row.get('id')}")
                plan = RetrievalPlanV1.model_validate(row["plan"], strict=True)
                if plan.objects != job["repos"]:
                    raise ValueError(f"candidate object mismatch: {row['id']}")
                for repo, family, source_hash in zip(job["repos"], job["source_families"],
                                                     job["source_hashes"], strict=True):
                    source = by_repo[repo]
                    if (source["split"] != expected_split or source["family"] != family or
                            source["sha256"] != source_hash):
                        raise ValueError(f"candidate source mismatch: {row['id']}")
                rows.append({**row, "provenance": {"run_sha256": digest(run_file),
                             "result_sha256": digest(file),
                             "source_manifest_sha256": digest(sources_path)}})
    if processed != set(by_job):
        raise ValueError(f"registered jobs missing: {len(set(by_job) - processed)}")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate ID")
    counts = Counter(row["kind"] for row in rows)
    families = {family for row in rows for family in row["source_families"]}
    exclusions = json.loads(EXCLUSIONS.read_text())
    forbidden_questions = set(exclusions["question_hashes"])
    questions = [sha256(re.sub(r"\s+", "", unicodedata.normalize("NFKC", row["question"]))
                       .casefold().encode()).hexdigest() for row in rows]
    if any(q in forbidden_questions for q in questions):
        raise ValueError("candidate question collides with held-out old evaluation")
    family_use = Counter(family for row in rows for family in row["source_families"])
    suspect = Counter()
    for row in rows:
        question = row["question"]
        if row["kind"] == "missing" and not any(word in question for word in
                ("不足", "缺", "未说明", "找不到", "无法确认")):
            suspect["missing_without_explicit_gap_request"] += 1
        if row["kind"] == "conflict" and not any(word in question for word in
                ("同一版本", "同一项目", "同一条件", "各项目内部", "每个项目")):
            suspect["conflict_scope_not_explicit"] += 1
        if any(word in question for word in ("uptime_monitors", "terminal_emulators", "api_clients")):
            suspect["internal_cohort_label_in_question"] += 1
    return rows, {"schema": "retrieval_plan_candidate_package_v1", "split": expected_split,
                  "registered_jobs": len(jobs), "run_receipts": runs,
                  "structural_candidates": len(rows), "kinds": dict(counts),
                  "source_families": len(families), "failed_jobs": failures,
                  "normalized_question_unique_fraction": len(set(questions)) / len(rows) if rows else 0,
                  "largest_family_occurrence_fraction": max(family_use.values(), default=0) / len(rows) if rows else 0,
                  "semantic_suspect_flags": dict(suspect),
                  "exclusions_sha256": digest(EXCLUSIONS),
                  "semantic_review": "not performed independently",
                  "admitted_training_rows": 0, "training_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "heldout"), default="train")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = package(args.run, args.jobs, args.sources, args.split)
    with args.out.open("x", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["candidates_sha256"] = digest(args.out)
    with args.summary.open("x", encoding="utf-8") as destination:
        json.dump(summary, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(json.dumps({"structural_candidates": len(rows),
                      "source_families": summary["source_families"],
                      "kinds": summary["kinds"], "admitted_training_rows": 0},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
