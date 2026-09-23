"""Bounded teacher pilot for new-source retrieval-plan drafts.

Same-model structural screening is not independent semantic approval. Drafts
carry review.accepted=false and cannot pass the training admission audit.
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
sys.path.insert(0, str(ROOT / "scripts/statetune_broad"))
from generate import Teacher, now, write  # noqa: E402

FROZEN = ROOT / "llamaindex-retrieval/statetune/retrieval-v1-20260923"


def profile(source):
    path = ROOT / source["readme_path"]
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError("source snapshot changed")
    headings = [line.lstrip("# ").strip() for line in raw.decode("utf-8").splitlines()
                if line.startswith("# ") or line.startswith("## ")]
    return {"repo": source["repo"], "description": source["description"],
            "readme_headings": headings[:12]}


def compile_draft(job, draft):
    if not isinstance(draft, dict) or set(draft) != {"items"} or not isinstance(draft["items"], list):
        raise ValueError("teacher item list missing")
    if len(draft["items"]) != 8:
        raise ValueError("teacher must return eight items")
    rows, failures = [], []
    for index, item in enumerate(draft["items"]):
        try:
            if not isinstance(item, dict) or set(item) != {"kind", "history", "question", "plan"}:
                raise ValueError("item structure invalid")
            if item["kind"] != job["focus"] or not isinstance(item["question"], str) or len(item["question"].strip()) < 8:
                raise ValueError("item kind or question invalid")
            if not isinstance(item["history"], list):
                raise ValueError("history must be a list")
            history = [ConversationMessage.model_validate(message, strict=True).model_dump()
                       for message in item["history"]]
            if (job["focus"] == "history") != bool(history):
                raise ValueError("history focus requires history; other cases must omit it")
            plan = RetrievalPlanV1.model_validate(item["plan"], strict=True)
            if not set(plan.objects) <= set(job["repos"]):
                raise ValueError("planner invented an object outside the source group")
            if job["focus"] == "ordinary" and len(plan.objects) != 1:
                raise ValueError("ordinary job requires one object")
            if job["focus"] != "ordinary" and len(plan.objects) < 2:
                raise ValueError("multi-object job requires at least two objects")
            if any("site:" in query.query.casefold() or "http://" in query.query or "https://" in query.query
                   for query in plan.initial_queries):
                raise ValueError("search query uses site syntax or URL")
            selected = [job["repos"].index(obj) for obj in plan.objects]
            rows.append({"id": f"{job['id']}:{index}", "split": job["split"],
                         "kind": item["kind"], "question": item["question"], "history": history,
                         "plan": plan.model_dump(),
                         "source_families": [job["source_families"][i] for i in selected],
                         "source_hashes": [job["source_hashes"][i] for i in selected],
                         "review": {"accepted": False, "author": "deepseek-flash",
                                    "reviewer": None, "reason": "unreviewed teacher draft"}})
        except (ValueError, TypeError, KeyError) as error:
            failures.append({"index": index, "error": str(error)})
    return rows, failures


def compile_draft_v2(job, draft):
    """V2 derives kind from the frozen job rather than a teacher-owned label."""
    if not isinstance(draft, dict) or set(draft) != {"items"} or not isinstance(draft["items"], list):
        raise ValueError("teacher item list missing")
    if len(draft["items"]) != 8:
        raise ValueError("teacher must return eight items")
    rows, failures = [], []
    for index, item in enumerate(draft["items"]):
        try:
            if not isinstance(item, dict) or set(item) != {"history", "question", "plan"}:
                raise ValueError("item structure invalid")
            if not isinstance(item["question"], str) or len(item["question"].strip()) < 8:
                raise ValueError("question too short")
            if not isinstance(item["history"], list):
                raise ValueError("history must be a list")
            history = [ConversationMessage.model_validate(message, strict=True).model_dump()
                       for message in item["history"]]
            if (job["focus"] == "history") != bool(history):
                raise ValueError("history focus requires history; other cases must omit it")
            plan = RetrievalPlanV1.model_validate(item["plan"], strict=True)
            if not set(plan.objects) <= set(job["repos"]):
                raise ValueError("planner invented an object outside the source group")
            if job["focus"] == "ordinary" and len(plan.objects) != 1:
                raise ValueError("ordinary job requires one object")
            if job["focus"] in {"comparison", "selection"} and len(plan.objects) < 2:
                raise ValueError("comparison or selection job requires at least two objects")
            if any("site:" in query.query.casefold() or "http://" in query.query or "https://" in query.query
                   for query in plan.initial_queries):
                raise ValueError("search query uses site syntax or URL")
            selected = [job["repos"].index(obj) for obj in plan.objects]
            rows.append({"id": f"{job['id']}:{index}", "split": job["split"],
                         "kind": job["focus"], "question": item["question"], "history": history,
                         "plan": plan.model_dump(),
                         "source_families": [job["source_families"][i] for i in selected],
                         "source_hashes": [job["source_hashes"][i] for i in selected],
                         "review": {"accepted": False, "author": "deepseek-flash",
                                    "reviewer": None, "reason": "unreviewed teacher draft"}})
        except (ValueError, TypeError, KeyError) as error:
            failures.append({"index": index, "error": str(error)})
    return rows, failures


def compile_draft_v3(job, draft):
    """Require the registered object count and grid/listed coverage on top of V2."""
    rows, failures = compile_draft_v2(job, draft)
    accepted = []
    for row in rows:
        index = int(row["id"].rsplit(":", 1)[1])
        plan = row["plan"]
        if plan["coverage"] != job["coverage_mode"]:
            failures.append({"index": index, "error": "coverage differs from registered job"})
        elif job["required_objects"] is not None and len(plan["objects"]) != job["required_objects"]:
            failures.append({"index": index, "error": "object count differs from registered job"})
        elif job["required_objects"] is not None and set(plan["objects"]) != set(job["repos"]):
            failures.append({"index": index, "error": "registered project omitted"})
        else:
            accepted.append(row)
    return accepted, failures


async def run(args):
    if args.prompt_version == "v3" and args.jobs_version != "v2":
        raise ValueError("V3 prompt requires V2 registered jobs")
    config = json.loads((FROZEN / "TEACHER-CONFIG.json").read_text())
    credentials = json.loads(args.credentials.read_text())
    if (args.credentials.stat().st_mode & 0o077 or
            credentials.get("base_url") != config["teacher_base_url"] or
            credentials.get("model") != config["teacher_model"]):
        raise ValueError("private teacher configuration mismatch")
    source_path = FROZEN / "SOURCES.json"
    job_path = FROZEN / ("JOBS-v2.jsonl" if args.jobs_version == "v2" else "JOBS.jsonl")
    source_manifest = json.loads(source_path.read_text())
    source_map = {row["repo"]: row for row in source_manifest["sources"]}
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
    prompt_path = FROZEN / {"v1": "TEACHER-PROMPT.txt", "v2": "TEACHER-PROMPT-v2.txt",
                            "v3": "TEACHER-PROMPT-v3.txt"}[args.prompt_version]
    write(args.out / "RUN.json", {"started_at": now(), "job_ids": [job["id"] for job in selected],
          "config_sha256": sha256((FROZEN / "TEACHER-CONFIG.json").read_bytes()).hexdigest(),
          "prompt_version": args.prompt_version,
          "jobs_version": args.jobs_version,
          "prompt_sha256": sha256(prompt_path.read_bytes()).hexdigest(),
          "sources_sha256": sha256(source_path.read_bytes()).hexdigest(),
          "jobs_sha256": sha256(job_path.read_bytes()).hexdigest(),
          "independent_review": False, "training_exported": False})
    teacher = Teacher(credentials, config, args.out, prior)
    queue = asyncio.Queue()
    for job in selected:
        queue.put_nowait(job)
    counts = Counter()

    async def worker():
        while not queue.empty() and not teacher.stopped.is_set():
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            result = {"job_id": job["id"], "status": "failed", "rows": [], "failures": []}
            try:
                body = {"model": config["teacher_model"],
                        "messages": [{"role": "system", "content": prompt_path.read_text()},
                                     {"role": "user", "content": json.dumps({"topic": job["topic"],
                                         "focus": job["focus"], "projects": [profile(source_map[repo])
                                         for repo in job["repos"]]}, ensure_ascii=False)}],
                        "response_format": {"type": "json_object"},
                        "thinking": {"type": config["thinking"]},
                        "temperature": config["generation_temperature"],
                        "max_tokens": config["generation_max_tokens"], "stream": False}
                draft = await teacher.call(job["id"], "generation", body)
                result["draft"] = draft
                compiler = {"v1": compile_draft, "v2": compile_draft_v2,
                            "v3": compile_draft_v3}[args.prompt_version]
                result["rows"], result["failures"] = compiler(job, draft)
                result["status"] = "structurally_checked"
                counts["rows"] += len(result["rows"])
            except Exception as error:
                result["error_type"] = type(error).__name__
                result["error"] = str(error) if isinstance(error, (ValueError, RuntimeError, KeyError)) else "generation failed"
                counts["failed_jobs"] += 1
            write(args.out / "results" / f"{job['id']}.json", result)
            counts["jobs"] += 1
            print(json.dumps({"job": job["id"], "status": result["status"],
                              "accepted_structure": len(result["rows"]),
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
    parser.add_argument("--prompt-version", choices=("v1", "v2", "v3"), default="v3")
    parser.add_argument("--jobs-version", choices=("v1", "v2"), default="v2")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
