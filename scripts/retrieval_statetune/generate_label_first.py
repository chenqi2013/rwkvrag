"""Draft plans first, then natural questions, with source and prompt bindings.

All output remains unreviewed candidate data. No target is approved or sent to
the State optimizer by this script. Raw teacher calls are preserved separately.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_plan_drafts import Teacher, now, profile, write  # noqa: E402

FROZEN = ROOT / "llamaindex-retrieval/statetune/retrieval-v1-20260923"


def build_plans(job, draft):
    if not isinstance(draft, dict) or set(draft) != {"plans"} or not isinstance(draft["plans"], list):
        raise ValueError("plans list missing")
    if len(draft["plans"]) != 8:
        raise ValueError("eight plan candidates required")
    compiled, failures = [], []
    for index, item in enumerate(draft["plans"]):
        try:
            if not isinstance(item, dict) or set(item) != {"dimensions", "conditions", "assignments", "queries"}:
                raise ValueError("plan draft fields invalid")
            dimensions, assignments = item["dimensions"], item["assignments"]
            if (not isinstance(dimensions, list) or not dimensions or
                    any(not isinstance(d, str) or not d.strip() for d in dimensions) or
                    len(dimensions) != len(set(dimensions))):
                raise ValueError("invalid dimensions")
            if not isinstance(assignments, list):
                raise ValueError("assignments must be a list")
            pairs = []
            if job["coverage_mode"] == "grid":
                if assignments:
                    raise ValueError("grid assignments must be empty")
            else:
                if ([a.get("object") for a in assignments] != job["repos"] or
                        any(not isinstance(a, dict) or set(a) != {"object", "dimensions"} or
                            not isinstance(a["dimensions"], list) or not a["dimensions"] or
                            len(a["dimensions"]) != len(set(a["dimensions"]))
                            for a in assignments)):
                    raise ValueError("listed assignments must cover each repo exactly once")
                for assignment in assignments:
                    pairs.extend({"object": assignment["object"], "dimension": dimension}
                                 for dimension in assignment["dimensions"])
                if {p["dimension"] for p in pairs} != set(dimensions):
                    raise ValueError("listed dimension union differs from declared dimensions")
            plan = RetrievalPlanV1.model_validate({"coverage": job["coverage_mode"],
                "objects": job["repos"], "dimensions": dimensions, "conditions": item["conditions"],
                "listed_pairs": pairs, "initial_queries": item["queries"]}, strict=True)
            if any("site:" in query.query.casefold() or "http://" in query.query or "https://" in query.query
                   for query in plan.initial_queries):
                raise ValueError("search query uses site syntax or URL")
            compiled.append({"index": index, "plan": plan.model_dump()})
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            failures.append({"index": index, "error": str(error)})
    return compiled, failures


def bind_questions(job, plans, draft):
    if not isinstance(draft, dict) or set(draft) != {"items"} or not isinstance(draft["items"], list):
        raise ValueError("question item list missing")
    items = draft["items"]
    wanted = [row["index"] for row in plans]
    if len(items) != len(wanted) or [item.get("index") for item in items] != wanted:
        raise ValueError("question indices do not match accepted plan indices")
    rows, failures, seen = [], [], set()
    for plan_row, item in zip(plans, items, strict=True):
        index, plan = plan_row["index"], plan_row["plan"]
        try:
            if not isinstance(item, dict) or set(item) != {"index", "history", "question"}:
                raise ValueError("question item fields invalid")
            if not isinstance(item["question"], str) or len(item["question"].strip()) < 10:
                raise ValueError("question too short")
            if not isinstance(item["history"], list):
                raise ValueError("history must be a list")
            history = [ConversationMessage.model_validate(message, strict=True).model_dump()
                       for message in item["history"]]
            if (job["focus"] == "history") != bool(history):
                raise ValueError("history focus mismatch")
            full_task = " ".join([*(message["content"] for message in history), item["question"]]).casefold()
            if any(label.casefold() not in full_task for label in
                   [*plan["objects"], *plan["dimensions"], *plan["conditions"]]):
                raise ValueError("question/history omitted a plan label")
            normalized = "".join(item["question"].split()).casefold()
            if normalized in seen:
                raise ValueError("duplicate normalized question in one job")
            seen.add(normalized)
            rows.append({"id": f"{job['id']}:{index}", "split": job["split"],
                         "kind": job["focus"], "question": item["question"], "history": history,
                         "plan": plan, "source_families": job["source_families"],
                         "source_hashes": job["source_hashes"],
                         "review": {"accepted": False, "independent_of_author": False,
                                    "author": "deepseek-flash", "reviewer": None,
                                    "reason": "unreviewed label-first draft"}})
        except (ValueError, TypeError, KeyError) as error:
            failures.append({"index": index, "error": str(error)})
    return rows, failures


async def run(args):
    config_path = FROZEN / "TEACHER-CONFIG.json"
    config = json.loads(config_path.read_text())
    credentials = json.loads(args.credentials.read_text())
    if (args.credentials.stat().st_mode & 0o077 or
            credentials.get("base_url") != config["teacher_base_url"] or
            credentials.get("model") != config["teacher_model"]):
        raise ValueError("private teacher configuration mismatch")
    source_path, job_path = FROZEN / "SOURCES.json", FROZEN / "JOBS-v2.jsonl"
    source_map = {row["repo"]: row for row in json.loads(source_path.read_text())["sources"]}
    all_jobs = [json.loads(line) for line in job_path.read_text().splitlines()]
    if not 0 <= args.start < args.stop <= len(all_jobs):
        raise ValueError("invalid job range")
    selected = all_jobs[args.start:args.stop]
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
    plan_prompt = FROZEN / "LABEL-FIRST-PLAN-v1.txt"
    question_prompt = FROZEN / "LABEL-FIRST-QUESTION-v1.txt"
    write(args.out / "RUN.json", {"started_at": now(), "job_ids": [job["id"] for job in selected],
          "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
          "config_sha256": sha256(config_path.read_bytes()).hexdigest(),
          "sources_sha256": sha256(source_path.read_bytes()).hexdigest(),
          "jobs_sha256": sha256(job_path.read_bytes()).hexdigest(),
          "plan_prompt_sha256": sha256(plan_prompt.read_bytes()).hexdigest(),
          "question_prompt_sha256": sha256(question_prompt.read_bytes()).hexdigest(),
          "independent_review": False, "training_exported": False})
    teacher = Teacher(credentials, config, args.out, prior)
    queue = asyncio.Queue()
    for job in selected:
        queue.put_nowait(job)
    counts = Counter()

    def body(prompt, value):
        return {"model": config["teacher_model"],
                "messages": [{"role": "system", "content": prompt.read_text()},
                             {"role": "user", "content": json.dumps(value, ensure_ascii=False)}],
                "response_format": {"type": "json_object"},
                "thinking": {"type": config["thinking"]},
                "temperature": config["generation_temperature"],
                "max_tokens": config["generation_max_tokens"], "stream": False}

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
                plan_raw = await teacher.call(job["id"], "plan", body(plan_prompt, {
                    "focus": job["focus"], "coverage_mode": job["coverage_mode"],
                    "repos": profiles}))
                result["plan_raw"] = plan_raw
                result["plans"], result["plan_failures"] = build_plans(job, plan_raw)
                if result["plans"]:
                    question_raw = await teacher.call(job["id"], "question", body(question_prompt, {
                        "focus": job["focus"], "plans": result["plans"]}))
                    result["question_raw"] = question_raw
                    result["rows"], result["question_failures"] = bind_questions(
                        job, result["plans"], question_raw)
                result["status"] = "drafted"
                counts["plan_rows"] += len(result["plans"])
                counts["question_rows"] += len(result["rows"])
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
    write(args.out / "SUMMARY.json", {**counts, "cost_upper_usd": teacher.spent,
          "stopped": teacher.stopped.is_set(), "independent_review": False,
          "training_exported": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--credentials", type=Path,
                        default=Path.home() / ".config/rwkvrag/teacher-deepseek-20260922.json")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
