"""Draft source-paired comparison positives and single-source negatives.

The teacher can propose labels, but exact spans and split bindings are checked
locally. Results remain unreviewed candidates, never a training export.
"""

import argparse
import asyncio
from collections import Counter
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import sys

from llamaindex_retrieval.rwkvos_batch import render_batch_prompt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/statetune_broad"))
from generate import Teacher, now, write  # noqa: E402

V1 = ROOT / "llamaindex-retrieval/statetune/retrieval-v1-20260923"
V4 = ROOT / "llamaindex-retrieval/statetune/retrieval-v4-20260924"
V7 = ROOT / "llamaindex-retrieval/statetune/retrieval-v7-20260924"
V8 = ROOT / "llamaindex-retrieval/statetune/retrieval-v8-20260924"


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save_rows(path, items):
    with path.open("x", encoding="utf-8") as out:
        for item in items:
            out.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def make_jobs():
    sources = {s["repo"]: s for s in json.loads((V4 / "SOURCES-CURATED.json").read_text())["sources"]}
    original = {job["repo"]: job for job in rows(V7 / "EVIDENCE-JOBS.jsonl")}
    result, seen = [], set()
    for anchor in original.values():
        for peer in anchor["peers"]:
            key = tuple(sorted((anchor["repo"], peer)))
            if key in seen or peer not in original:
                continue
            seen.add(key)
            a, b = (original[repo] for repo in key)
            if a["split"] != b["split"]:
                raise ValueError("pair crosses splits")
            for item in (a, b):
                if sources[item["repo"]]["sha256"] != item["source_hash"]:
                    raise ValueError("pair source hash mismatch")
            result.append({"id": f"pair{len(result):03d}", "split": a["split"],
                           "a": {key: a[key] for key in ("repo", "family", "source_hash", "blocks")},
                           "b": {key: b[key] for key in ("repo", "family", "source_hash", "blocks")}})
    return result


def validate(job, answer):
    if not isinstance(answer, dict) or not isinstance(answer.get("items"), list):
        raise ValueError("teacher items missing")
    blocks_a = {b["id"]: b for b in job["a"]["blocks"]}
    blocks_b = {b["id"]: b for b in job["b"]["blocks"]}
    accepted, failures = [], []
    for index, item in enumerate(answer["items"]):
        try:
            required = {
                "question", "kind", "status", "block_a", "quote_a", "block_b", "quote_b"}
            if not isinstance(item, dict) or not required <= item.keys():
                raise ValueError("invalid fields")
            extras = set(item) - required
            if any(item[key] not in (None, "", []) for key in extras):
                raise ValueError("teacher supplied nonempty qualification outside label")
            question, kind, status = item["question"], item["kind"], item["status"]
            if not isinstance(question, str) or len(question.strip()) < 14:
                raise ValueError("invalid question")
            block_a = blocks_a[item["block_a"]]
            quote_a, quote_b, block_b = item["quote_a"], item["quote_b"], None
            if kind == "comparison" and status == "supported":
                if (job["a"]["repo"] not in question or job["b"]["repo"] not in question or
                        not isinstance(quote_a, str) or not 12 <= len(quote_a) <= 220 or
                        quote_a not in block_a["text"] or item["block_b"] not in blocks_b or
                        not isinstance(quote_b, str) or not 12 <= len(quote_b) <= 220):
                    raise ValueError("comparison must have two exact source quotes")
                block_b = blocks_b[item["block_b"]]
                if quote_b not in block_b["text"]:
                    raise ValueError("second quote is not exact")
            elif kind == "ordinary" and status == "insufficient":
                slug = job["a"]["repo"].rsplit("/", 1)[1]
                if (job["a"]["repo"].casefold() not in question.casefold() and
                        slug.casefold() not in question.casefold()) or (
                        job["b"]["repo"].casefold() in question.casefold() or
                        quote_a is not None or quote_b is not None or item["block_b"] is not None):
                    raise ValueError("ordinary negative is not one-source incomplete")
            else:
                raise ValueError("unsupported kind/status combination")
            accepted.append({"id": f"v8-{job['id']}:{index}", "split": job["split"],
                             "role": "evidence", "kind": kind, "question": question,
                             "status": status, "source_families": [job["a"]["family"]] +
                             ([job["b"]["family"]] if block_b else []),
                             "source_hashes": [job["a"]["source_hash"]] +
                             ([job["b"]["source_hash"]] if block_b else []),
                             "mentioned_peer_family": job["b"]["family"] if block_b else None,
                             "sources": [{"repo": job["a"]["repo"], "block_id": block_a["id"],
                                          "source_sha256": job["a"]["source_hash"],
                                          "block_sha256": block_a["sha256"],
                                          "text": block_a["text"], "quote": quote_a,
                                          "block_start_byte": block_a["start_byte"]}] +
                                        ([{"repo": job["b"]["repo"], "block_id": block_b["id"],
                                           "source_sha256": job["b"]["source_hash"],
                                           "block_sha256": block_b["sha256"],
                                           "text": block_b["text"], "quote": quote_b,
                                           "block_start_byte": block_b["start_byte"]}]
                                         if block_b else []),
                             "review": {"accepted": False, "independent_of_author": False,
                                        "author": "deepseek-flash", "reviewer": None,
                                        "reason": "exact-span check only; comparison semantics unreviewed"}})
        except (KeyError, TypeError, ValueError) as error:
            failures.append({"index": index, "error": str(error)})
    return accepted, failures


def body(config, prompt, job):
    return {"model": config["teacher_model"],
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": json.dumps(job, ensure_ascii=False)}],
            "response_format": {"type": "json_object"},
            "thinking": {"type": config["thinking"]},
            "temperature": config["generation_temperature"],
            "max_tokens": config["generation_max_tokens"], "stream": False}


async def generate(args):
    config = json.loads((V1 / "TEACHER-CONFIG.json").read_text())
    credential = json.loads(args.credentials.read_text())
    if (args.credentials.stat().st_mode & 0o077 or
            credential.get("base_url") != config["teacher_base_url"] or
            credential.get("model") != config["teacher_model"]):
        raise ValueError("private teacher config mismatch")
    jobs = rows(V8 / "PAIR-JOBS.jsonl")
    selected = jobs[args.start:args.stop]
    if not selected or not 0 <= args.start < args.stop <= len(jobs):
        raise ValueError("invalid range")
    prompt_path = V8 / "PAIR-TEACHER-v2.txt"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    guard = (args.out.parent / "teacher.lock").open("a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior = 0.0
    for request in args.out.parent.glob("*/calls/*.request.json"):
        response = request.with_name(request.name.replace(".request.json", ".response.json"))
        prior += (json.loads(response.read_text())["conservative_cost_usd"] if response.exists()
                  else json.loads(request.read_text())["reserved_upper_usd"])
    args.out.mkdir(exist_ok=False)
    (args.out / "calls").mkdir()
    (args.out / "results").mkdir()
    write(args.out / "RUN.json", {"started_at": now(), "start": args.start, "stop": args.stop,
          "jobs_sha256": file_hash(V8 / "PAIR-JOBS.jsonl"),
          "prompt_sha256": file_hash(prompt_path), "script_sha256": file_hash(Path(__file__)),
          "independent_review": False, "training_exported": False})
    teacher = Teacher(credential, config, args.out, prior)
    queue = asyncio.Queue()
    for job in selected:
        queue.put_nowait(job)
    counts = Counter()

    async def worker():
        while not queue.empty() and not teacher.stopped.is_set():
            job = queue.get_nowait()
            result = {"job_id": job["id"], "status": "failed", "rows": [], "failures": []}
            try:
                answer = await teacher.call(job["id"], "generation", body(config,
                    prompt_path.read_text(), job))
                result["teacher_output"] = answer
                result["rows"], result["failures"] = validate(job, answer)
                result["status"] = "unreviewed_drafts"
            except Exception as error:
                result["error_type"] = type(error).__name__
                result["error"] = str(error) if isinstance(error, (ValueError, RuntimeError, KeyError)) else "generation failed"
            write(args.out / "results" / f"{job['id']}.json", result)
            counts["jobs"] += 1
            counts["rows"] += len(result["rows"])
            counts["failed_rows"] += len(result["failures"])
            counts["failed_jobs"] += result["status"] == "failed"
            print(json.dumps({"job": job["id"], "rows": len(result["rows"]),
                              "failures": len(result["failures"]),
                              "cost_upper_usd": round(teacher.spent, 4)}), flush=True)

    try:
        await asyncio.gather(*(worker() for _ in range(config["concurrency"])))
    finally:
        await teacher.close()
        write(args.out / "SUMMARY.json", {**counts, "planned_jobs": len(selected),
              "cost_upper_usd": teacher.spent, "stopped": teacher.stopped.is_set(),
              "independent_review": False, "training_exported": False})


def compile_row(row):
    evidence = [{"id": source["block_id"], "repo": source["repo"], "text": source["text"]}
                for source in row["sources"]]
    prompt_message = ("判断给定官方来源摘录能否完整支持问题。只按提供的文字判断；"
                      "如果支持，quotes 必须给出每个所需来源的逐字连续引文；"
                      "如果不足，quotes 为空数组。只输出 status、quotes 两键 JSON。输入：" +
                      json.dumps({"question": row["question"], "sources": evidence,
                                  "execution_state": "read_success"},
                                 ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    prompt, _ = render_batch_prompt([{"role": "user", "content": prompt_message}],
                                    "<think></think>", "complete")
    quotes = [{"source_id": source["block_id"], "quote": source["quote"]}
              for source in row["sources"] if source["quote"] is not None]
    target = json.dumps({"status": row["status"], "quotes": quotes},
                        ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {**row, "prompt": prompt, "target": target,
            "prompt_sha256": sha256(prompt.encode()).hexdigest(),
            "target_sha256": sha256(target.encode()).hexdigest(),
            "decision_protocol": "retrieval-evidence-quotes-v8",
            "prompt_protocol": "rwkvos_batch_complete_no_final_lf_v1"}


def package(args):
    gathered, run_receipts = [], []
    for run in args.runs:
        manifest = json.loads((run / "RUN.json").read_text())
        summary = json.loads((run / "SUMMARY.json").read_text())
        if summary["stopped"] or manifest["independent_review"] or manifest["training_exported"]:
            raise ValueError("run is incomplete or status invalid")
        for path in sorted((run / "results").glob("*.json")):
            result = json.loads(path.read_text())
            if result["status"] == "unreviewed_drafts":
                gathered.extend(compile_row(row) for row in result["rows"])
        run_receipts.append({"path": str(run), "run_sha256": file_hash(run / "RUN.json"),
                             "summary_sha256": file_hash(run / "SUMMARY.json"),
                             "rows": summary["rows"], "failed_rows": summary["failed_rows"]})
    if len({row["id"] for row in gathered}) != len(gathered):
        raise ValueError("duplicate IDs")
    save_rows(args.out, gathered)
    write(args.summary, {"schema": "retrieval_balanced_evidence_drafts_v8",
          "rows": len(gathered), "splits": dict(Counter(r["split"] for r in gathered)),
          "kind_status": dict(Counter(f"{r['kind']}:{r['status']}" for r in gathered)),
          "candidate_sha256": file_hash(args.out), "runs": run_receipts,
          "independent_review": False, "admitted_training_rows": 0,
          "training_started": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("jobs")
    generation = sub.add_parser("generate")
    generation.add_argument("--start", type=int, required=True)
    generation.add_argument("--stop", type=int, required=True)
    generation.add_argument("--out", type=Path, required=True)
    generation.add_argument("--credentials", type=Path,
                            default=Path.home() / ".config/rwkvrag/teacher-deepseek-20260922.json")
    packaging = sub.add_parser("package")
    packaging.add_argument("--runs", type=Path, action="append", required=True)
    packaging.add_argument("--out", type=Path, required=True)
    packaging.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "jobs":
        jobs = make_jobs()
        V8.mkdir(exist_ok=True)
        save_rows(V8 / "PAIR-JOBS.jsonl", jobs)
        write(V8 / "JOBS-SUMMARY.json", {"jobs": len(jobs),
              "splits": dict(Counter(job["split"] for job in jobs)),
              "v7_evidence_jobs_sha256": file_hash(V7 / "EVIDENCE-JOBS.jsonl"),
              "source_manifest_sha256": file_hash(V4 / "SOURCES-CURATED.json"),
              "training_started": False})
        print(json.dumps({"jobs": len(jobs), "splits": dict(Counter(j["split"] for j in jobs))}))
    elif args.command == "generate":
        asyncio.run(generate(args))
    else:
        package(args)


if __name__ == "__main__":
    main()
