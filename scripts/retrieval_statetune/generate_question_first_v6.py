"""Pilot question-first plans with per-question selected project subsets.

Raw teacher calls are preserved and all joined rows remain unreviewed.
"""

import argparse
import asyncio
from collections import Counter
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import sys

from llamaindex_retrieval.retrieval_plan import RetrievalPlanV1
from llamaindex_retrieval.schemas import ConversationMessage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/retrieval_statetune"))
from generate_plan_drafts import Teacher, now, profile, write  # noqa: E402


V1 = ROOT / "llamaindex-retrieval/statetune/retrieval-v1-20260923"
V6 = ROOT / "llamaindex-retrieval/statetune/retrieval-v6-20260924"


def bind_questions(job, draft):
    if not isinstance(draft, dict) or set(draft) != {"items"} or not isinstance(draft["items"], list):
        raise ValueError("question-first response must contain items")
    items = draft["items"]
    if len(items) != 8 or [item.get("index") if isinstance(item, dict) else None for item in items] != list(range(8)):
        raise ValueError("question-first response needs ordered indices 0..7")
    valid, failures, seen = {}, [], set()
    for index, item in enumerate(items):
        try:
            if not {"index", "history", "question"} <= set(item) or not isinstance(item["question"], str):
                raise ValueError("question item fields invalid")
            if len(item["question"].strip()) < 12 or not isinstance(item["history"], list):
                raise ValueError("question too short or history invalid")
            history = [ConversationMessage.model_validate(message, strict=True).model_dump()
                       for message in item["history"]]
            if any(message["role"] != "user" for message in history):
                raise ValueError("history must contain user messages only")
            if (job["focus"] == "history") != bool(history):
                raise ValueError("history focus mismatch")
            if history and not any(word in item["question"] for word in ("现在", "撤回", "改", "不再", "取消")):
                raise ValueError("latest question has no explicit revision")
            latest = item["question"].casefold()
            selected = [index for index, repo in enumerate(job["repos"]) if repo.casefold() in latest]
            if (job["focus"] == "ordinary" and len(selected) != 1 or
                    job["focus"] in {"comparison", "selection"} and len(selected) < 2 or
                    not selected):
                raise ValueError("question selected too few or too many projects")
            normalized = "".join(item["question"].split()).casefold()
            if normalized in seen:
                raise ValueError("duplicate normalized question in job")
            seen.add(normalized)
            valid[index] = {"history": history, "question": item["question"],
                            "selected_indices": selected}
        except (ValueError, KeyError, TypeError) as error:
            failures.append({"index": index, "error": str(error)})
    return valid, failures


def validate_plans(job, questions, draft):
    if not isinstance(draft, dict) or set(draft) != {"plans"} or not isinstance(draft["plans"], list):
        raise ValueError("planning response must contain plans")
    expected = list(questions)
    if len(draft["plans"]) != len(expected):
        raise ValueError("one plan per validated question required")
    accepted, failures = [], []
    for item in draft["plans"]:
        index = item.get("index") if isinstance(item, dict) else None
        try:
            if (not isinstance(item, dict) or set(item) !=
                    {"index", "dimensions", "conditions", "assignments", "queries"} or
                    index not in questions):
                raise ValueError("plan fields or question index invalid")
            selected = questions[index]["selected_indices"]
            repos = [job["repos"][position] for position in selected]
            pairs = []
            if job["coverage_mode"] == "grid":
                if item["assignments"] != []:
                    raise ValueError("grid assignments must be empty")
            else:
                assignments = item["assignments"]
                if (not isinstance(assignments, list) or
                        [a.get("object") if isinstance(a, dict) else None for a in assignments] != repos):
                    raise ValueError("listed assignments must cover selected repos in order")
                for assignment in assignments:
                    if set(assignment) != {"object", "dimensions"} or not isinstance(assignment["dimensions"], list):
                        raise ValueError("listed assignment fields invalid")
                    pairs.extend({"object": assignment["object"], "dimension": dimension}
                                 for dimension in assignment["dimensions"])
            plan = RetrievalPlanV1.model_validate({"coverage": job["coverage_mode"],
                "objects": repos, "dimensions": item["dimensions"],
                "conditions": item["conditions"], "listed_pairs": pairs,
                "initial_queries": item["queries"]}, strict=True)
            if any("site:" in query.query.casefold() or "http://" in query.query or
                   "https://" in query.query for query in plan.initial_queries):
                raise ValueError("search query uses site syntax or URL")
            accepted.append({"index": index, "plan": plan.model_dump()})
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            failures.append({"index": index, "error": str(error)})
    if {item["index"] for item in accepted} | {item["index"] for item in failures} != set(expected):
        raise ValueError("planning response omitted or repeated an index")
    return accepted, failures


def join(job, questions, plans):
    rows = []
    for item in plans:
        index = item["index"]
        if index not in questions:
            continue
        question = questions[index]
        selected = question["selected_indices"]
        rows.append({"id": f"{job['id']}:{index}", "split": job["split"],
                     "kind": job["focus"], "question": question["question"],
                     "history": question["history"], "plan": item["plan"],
                     "source_families": [job["source_families"][position] for position in selected],
                     "source_hashes": [job["source_hashes"][position] for position in selected],
                     "review": {"accepted": False, "independent_of_author": False,
                                "author": "deepseek-flash", "reviewer": None,
                                "reason": "unreviewed question-first subset draft"}})
    return rows


def body(config, prompt, value):
    return {"model": config["teacher_model"],
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": json.dumps(value, ensure_ascii=False)}],
            "response_format": {"type": "json_object"},
            "thinking": {"type": config["thinking"]},
            "temperature": config["generation_temperature"],
            "max_tokens": config["generation_max_tokens"], "stream": False}


async def run(args):
    config_file = V1 / "TEACHER-CONFIG.json"
    config = json.loads(config_file.read_text())
    credential = json.loads(args.credentials.read_text())
    if (args.credentials.stat().st_mode & 0o077 or
            credential.get("base_url") != config["teacher_base_url"] or
            credential.get("model") != config["teacher_model"]):
        raise ValueError("private teacher configuration mismatch")
    source_map = {row["repo"]: row for row in json.loads(args.sources.read_text())["sources"]}
    jobs = [json.loads(line) for line in args.jobs.read_text().splitlines()]
    if not jobs or any(job["split"] != "train" for job in jobs):
        raise ValueError("pilot needs train-only registered jobs")
    qfile, pfile = V6 / "QUESTION-FIRST-v6.txt", V6 / "PLAN-FROM-QUESTION-v6.txt"
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
    write(args.out / "RUN.json", {"started_at": now(), "job_ids": [job["id"] for job in jobs],
          "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
          "sources_sha256": sha256(args.sources.read_bytes()).hexdigest(),
          "jobs_sha256": sha256(args.jobs.read_bytes()).hexdigest(),
          "config_sha256": sha256(config_file.read_bytes()).hexdigest(),
          "question_prompt_sha256": sha256(qfile.read_bytes()).hexdigest(),
          "plan_prompt_sha256": sha256(pfile.read_bytes()).hexdigest(),
          "independent_review": False, "training_exported": False})
    teacher = Teacher(credential, config, args.out, prior)
    queue = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)
    stats = Counter()

    async def worker():
        while not queue.empty() and not teacher.stopped.is_set():
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            result = {"job_id": job["id"], "status": "failed", "rows": [],
                      "question_failures": [], "plan_failures": []}
            try:
                profiles = [profile(source_map[repo]) for repo in job["repos"]]
                qinput = {"focus": job["focus"], "role": job["role"],
                          "comparison_rule": job["comparison_rule"], "repos": profiles}
                raw_questions = await teacher.call(job["id"], "questions", body(config,
                    qfile.read_text(), qinput))
                result["question_raw"] = raw_questions
                questions, result["question_failures"] = bind_questions(job, raw_questions)
                if questions:
                    pinput = {"focus": job["focus"], "coverage_mode": job["coverage_mode"],
                              "role": job["role"], "comparison_rule": job["comparison_rule"],
                              "items": [{"index": index, "question": question["question"],
                                         "history": question["history"],
                                         "selected_repos": [profiles[position]
                                                            for position in question["selected_indices"]]}
                                        for index, question in questions.items()]}
                    raw_plans = await teacher.call(job["id"], "plans", body(config,
                        pfile.read_text(), pinput))
                    result["plan_raw"] = raw_plans
                    plans, result["plan_failures"] = validate_plans(job, questions, raw_plans)
                    result["rows"] = join(job, questions, plans)
                result["status"] = "unreviewed_drafts"
                stats["joined_rows"] += len(result["rows"])
            except Exception as error:
                result["error_type"] = type(error).__name__
                result["error"] = str(error) if isinstance(error, (ValueError, RuntimeError, KeyError)) else "generation failed"
                stats["failed_jobs"] += 1
            write(args.out / "results" / f"{job['id']}.json", result)
            stats["jobs"] += 1
            print(json.dumps({"job": job["id"], "status": result["status"],
                              "rows": len(result["rows"]),
                              "cost_upper_usd": round(teacher.spent, 4)}, ensure_ascii=False), flush=True)

    try:
        await asyncio.gather(*(worker() for _ in range(config["concurrency"])))
    finally:
        await teacher.close()
        write(args.out / "SUMMARY.json", {**stats, "planned_jobs": len(jobs),
              "cost_upper_usd": teacher.spent, "stopped": teacher.stopped.is_set(),
              "independent_review": False, "training_exported": False})
        guard.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--credentials", type=Path,
                        default=Path.home() / ".config/rwkvrag/teacher-deepseek-20260922.json")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
