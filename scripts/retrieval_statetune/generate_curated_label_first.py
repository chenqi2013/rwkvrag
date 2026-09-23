"""Draft V2 retrieval plans and natural questions from curated source cohorts.

All results remain unreviewed candidates. This never exports a training set or
invokes a student model. Each raw teacher request/response is kept verbatim.
"""

import argparse
import asyncio
from collections import Counter
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/retrieval_statetune"))
from generate_label_first import build_plans, bind_questions  # noqa: E402
from generate_plan_drafts import Teacher, now, profile, write  # noqa: E402


V1 = ROOT / "llamaindex-retrieval/statetune/retrieval-v1-20260923"
V2 = ROOT / "llamaindex-retrieval/statetune/retrieval-v2-20260923"


def request_body(config, prompt, value):
    return {"model": config["teacher_model"],
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": json.dumps(value, ensure_ascii=False)}],
            "response_format": {"type": "json_object"},
            "thinking": {"type": config["thinking"]},
            "temperature": config["generation_temperature"],
            "max_tokens": config["generation_max_tokens"], "stream": False}


async def run(args):
    config_path = V1 / "TEACHER-CONFIG.json"
    config = json.loads(config_path.read_text())
    credentials = json.loads(args.credentials.read_text())
    if (args.credentials.stat().st_mode & 0o077 or
            credentials.get("base_url") != config["teacher_base_url"] or
            credentials.get("model") != config["teacher_model"]):
        raise ValueError("private teacher identity or file permissions mismatch")
    source_path = V2 / "SOURCES-CURATED.json"
    job_path = V2 / {"train": "TRAIN-JOBS.jsonl", "dev": "DEV-JOBS.jsonl",
                     "heldout": "BLIND-JOBS.jsonl"}[args.split]
    source_map = {row["repo"]: row for row in json.loads(source_path.read_text())["sources"]}
    all_jobs = [json.loads(line) for line in job_path.read_text().splitlines()]
    if not 0 <= args.start < args.stop <= len(all_jobs):
        raise ValueError("invalid job interval")
    jobs = all_jobs[args.start:args.stop]
    if any(job["split"] != args.split for job in jobs):
        raise ValueError("split mismatch in registered jobs")
    plan_file, question_file = V2 / "LABEL-FIRST-PLAN-v2.txt", V2 / "LABEL-FIRST-QUESTION-v2.txt"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    guard = (args.out.parent / "teacher.lock").open("a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior = 0.0
    for path in args.out.parent.glob("*/calls/*.request.json"):
        receipt = path.with_name(path.name.replace(".request.json", ".response.json"))
        prior += (json.loads(receipt.read_text())["conservative_cost_usd"] if receipt.exists()
                  else json.loads(path.read_text())["reserved_upper_usd"])
    args.out.mkdir(exist_ok=False)
    (args.out / "calls").mkdir()
    (args.out / "results").mkdir()
    write(args.out / "RUN.json", {"started_at": now(), "split": args.split,
          "job_ids": [job["id"] for job in jobs], "prior_cost_upper_usd": prior,
          "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
          "config_sha256": sha256(config_path.read_bytes()).hexdigest(),
          "sources_sha256": sha256(source_path.read_bytes()).hexdigest(),
          "jobs_sha256": sha256(job_path.read_bytes()).hexdigest(),
          "plan_prompt_sha256": sha256(plan_file.read_bytes()).hexdigest(),
          "question_prompt_sha256": sha256(question_file.read_bytes()).hexdigest(),
          "independent_review": False, "training_exported": False})
    teacher = Teacher(credentials, config, args.out, prior)
    queue = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)
    counts = Counter()

    async def worker():
        while not queue.empty() and not teacher.stopped.is_set():
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            result = {"job_id": job["id"], "status": "failed", "plans": [],
                      "plan_failures": [], "rows": [], "question_failures": []}
            try:
                profiles = [profile(source_map[repo]) for repo in job["repos"]]
                plan_input = {"focus": job["focus"], "coverage_mode": job["coverage_mode"],
                              "cohort": job["topic"], "role": job["role"],
                              "comparison_rule": job["comparison_rule"], "repos": profiles}
                draft = await teacher.call(job["id"], "plan", request_body(
                    config, plan_file.read_text(), plan_input))
                result["plan_raw"] = draft
                result["plans"], result["plan_failures"] = build_plans(job, draft)
                if result["plans"]:
                    question_input = {"focus": job["focus"], "role": job["role"],
                                      "comparison_rule": job["comparison_rule"],
                                      "plans": result["plans"]}
                    question_draft = await teacher.call(job["id"], "question", request_body(
                        config, question_file.read_text(), question_input))
                    result["question_raw"] = question_draft
                    result["rows"], result["question_failures"] = bind_questions(
                        job, result["plans"], question_draft)
                result["status"] = "unreviewed_drafts"
                counts["plans"] += len(result["plans"])
                counts["questions"] += len(result["rows"])
            except Exception as error:
                result["error_type"] = type(error).__name__
                result["error"] = str(error) if isinstance(error, (ValueError, RuntimeError, KeyError)) else "generation failed"
                counts["failed_jobs"] += 1
            write(args.out / "results" / f"{job['id']}.json", result)
            counts["jobs"] += 1
            print(json.dumps({"job": job["id"], "status": result["status"],
                              "plans": len(result["plans"]), "questions": len(result["rows"]),
                              "cost_upper_usd": round(teacher.spent, 4)}, ensure_ascii=False), flush=True)

    try:
        await asyncio.gather(*(worker() for _ in range(config["concurrency"])))
    finally:
        await teacher.close()
        write(args.out / "SUMMARY.json", {**counts, "planned_jobs": len(jobs),
              "cost_upper_usd": teacher.spent, "stopped": teacher.stopped.is_set(),
              "independent_review": False, "training_exported": False})
        guard.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "dev", "heldout"), required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--credentials", type=Path,
                        default=Path.home() / ".config/rwkvrag/teacher-deepseek-20260922.json")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
