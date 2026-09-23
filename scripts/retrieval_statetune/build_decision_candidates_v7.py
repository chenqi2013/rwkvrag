"""Generate traceable, unreviewed evidence and follow-up StateTune candidates.

All source text is pinned to V4 README snapshots. Teacher output is retained in
an ignored local run directory; structurally valid drafts are *not* admitted
training rows. This script does not start a student model or an optimizer.
"""

import argparse
import asyncio
from collections import Counter, defaultdict
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import random
import sys

from llamaindex_retrieval.rwkvos_batch import render_batch_prompt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/statetune_broad"))
from generate import Teacher, now, write  # noqa: E402

V1 = ROOT / "llamaindex-retrieval/statetune/retrieval-v1-20260923"
V3 = ROOT / "llamaindex-retrieval/statetune/retrieval-v3-20260924"
V4 = ROOT / "llamaindex-retrieval/statetune/retrieval-v4-20260924"
V7 = ROOT / "llamaindex-retrieval/statetune/retrieval-v7-20260924"
BLOCKS = ROOT / "data/corpora/retrieval-evidence-blocks-v2-20260924.jsonl"


def digest(value):
    return sha256(value.encode("utf-8")).hexdigest()


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def checked_sources():
    sources = json.loads((V4 / "SOURCES-CURATED.json").read_text())["sources"]
    for source in sources:
        raw = (ROOT / source["readme_path"]).read_bytes()
        if sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError(f"README snapshot changed: {source['repo']}")
    return sources


def useful_blocks(source, blocks):
    """Filter only obvious markup noise; this is not semantic label approval."""
    candidates = []
    for block in blocks:
        content = block["text"].strip()
        if (block["kind"] not in {"paragraph", "list", "table"} or
                not 110 <= len(content) <= 1500 or
                content.count("http") > 5 or content.count("![") > 0 or
                content.count("[!") > 0 or content.count("|") > 16):
            continue
        candidates.append(block)
    rng = random.Random(digest(source["family"] + ":v7-blocks"))
    rng.shuffle(candidates)
    # Prefer distinct headings so one long installation section does not dominate.
    picked, headings = [], set()
    for block in candidates:
        heading = tuple(block["heading_path"])
        if heading in headings:
            continue
        picked.append(block)
        headings.add(heading)
        if len(picked) == 5:
            break
    for block in candidates:
        if len(picked) == 5:
            break
        if block not in picked:
            picked.append(block)
    return picked


def build_jobs():
    sources = checked_sources()
    blocks = jsonl(BLOCKS)
    by_family = defaultdict(list)
    for block in blocks:
        by_family[block["family"]].append(block)
    peers = defaultdict(list)
    for source in sources:
        peers[(source["split"], source["cohort"])].append(source)
    evidence, skipped = [], []
    for source in sources:
        selected = useful_blocks(source, by_family[source["family"]])
        alternatives = [item for item in peers[(source["split"], source["cohort"])]
                        if item["family"] != source["family"]]
        if not selected or not alternatives:
            skipped.append({"family": source["family"], "blocks": len(selected),
                            "peer_count": len(alternatives)})
            continue
        rng = random.Random(digest(source["family"] + ":v7-peers"))
        rng.shuffle(alternatives)
        evidence.append({"id": f"e{len(evidence):03d}", "role": "evidence",
                         "split": source["split"], "repo": source["repo"],
                         "family": source["family"], "source_hash": source["sha256"],
                         "blocks": [{"id": b["id"], "heading_path": b["heading_path"],
                                     "text": b["text"], "sha256": b["sha256"],
                                     "start_byte": b["start_byte"],
                                     "end_byte": b["end_byte"]} for b in selected],
                         "peers": [p["repo"] for p in alternatives[:3]]})

    plans = jsonl(V3 / "PLAN-CANDIDATES.jsonl") + jsonl(V4 / "PLAN-CANDIDATES-ADDED.jsonl")
    source_by_family = {source["family"]: source for source in sources}
    rng = random.Random(7240924)
    rng.shuffle(plans)
    selected_by_primary = Counter()
    selected_ids, followup = set(), []
    for plan in plans:
        primary = plan["source_families"][0]
        if selected_by_primary[primary] >= 7 or plan["id"] in selected_ids:
            continue
        if any(source_by_family[f]["split"] != "train" or
               source_by_family[f]["sha256"] != h
               for f, h in zip(plan["source_families"], plan["source_hashes"], strict=True)):
            continue
        cells = []
        content = plan["plan"]
        pairs = ([(obj, dim) for obj in content["objects"] for dim in content["dimensions"]]
                 if content["coverage"] == "grid" else
                 [(pair["object"], pair["dimension"]) for pair in content["listed_pairs"]])
        for index, (obj, dim) in enumerate(pairs, 1):
            cells.append({"id": f"c{index}", "object": obj, "dimension": dim,
                          "conditions": content["conditions"]})
        if not cells:
            continue
        target = cells[len(followup) % len(cells)]
        initial = next(q["query"] for q in content["initial_queries"]
                       if q["object"] == target["object"])
        mode = ("unsearched" if len(followup) % 5 < 3 else
                "read_without_verified_evidence")
        # Execution state is explicitly synthetic. It never asserts that an
        # actual search ran or that the README lacks this fact.
        followup.append({"id": f"f{len(followup):04d}", "role": "followup",
                         "split": "train", "kind": plan["kind"],
                         "plan_id": plan["id"], "question": plan["question"],
                         "source_families": plan["source_families"],
                         "source_hashes": plan["source_hashes"],
                         "cells": cells, "focus_cell": target["id"],
                         "focus_object": target["object"],
                         "focus_dimension": target["dimension"],
                         "execution_state": mode,
                         "attempted_queries": [] if mode == "unsearched" else [initial],
                         "verified_evidence_ids": [], "scenario_origin": "synthetic_state_on_pinned_plan"})
        selected_by_primary[primary] += 1
        selected_ids.add(plan["id"])
        if len(followup) == 900:
            break
    if len(followup) < 800:
        raise ValueError(f"not enough balanced follow-up jobs: {len(followup)}")
    return evidence, followup, skipped


def teacher_body(config, prompt, value):
    return {"model": config["teacher_model"],
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": json.dumps(value, ensure_ascii=False)}],
            "response_format": {"type": "json_object"},
            "thinking": {"type": config["thinking"]},
            "temperature": config["generation_temperature"],
            "max_tokens": config["generation_max_tokens"], "stream": False}


def evidence_rows(job, answer):
    if not isinstance(answer, dict) or not isinstance(answer.get("items"), list):
        raise ValueError("teacher evidence items missing")
    blocks = {block["id"]: block for block in job["blocks"]}
    rows, failures = [], []
    for index, item in enumerate(answer["items"]):
        try:
            if not isinstance(item, dict) or set(item) != {"question", "kind", "status", "block_id", "quote", "peer"}:
                raise ValueError("unexpected evidence fields")
            question = item["question"]
            status, kind = item["status"], item["kind"]
            block = blocks[item["block_id"]]
            repo_slug = job["repo"].rsplit("/", 1)[1]
            named = job["repo"].casefold() in question.casefold() or (
                len(repo_slug) >= 3 and repo_slug.casefold() in question.casefold())
            if not isinstance(question, str) or len(question.strip()) < 14 or not named:
                raise ValueError("question does not name the source object")
            if status not in {"supported", "insufficient"} or kind not in {"ordinary", "comparison", "selection", "missing"}:
                raise ValueError("invalid label or kind")
            if status == "supported":
                if kind != "ordinary" or item["peer"] is not None:
                    raise ValueError("supported case must be single-object")
                quote = item["quote"]
                if not isinstance(quote, str) or not 12 <= len(quote) <= 220 or quote not in block["text"]:
                    raise ValueError("quote is not an exact source span")
                quote_start = block["text"].index(quote)
            else:
                peer = item["peer"]
                if (kind not in {"comparison", "selection", "missing"} or
                        peer not in job["peers"] or peer not in question or item["quote"] is not None):
                    raise ValueError("incomplete comparison is not explicit")
                quote, quote_start = None, None
            rows.append({"id": f"v7-{job['id']}:{index}", "split": job["split"],
                         "role": "evidence", "kind": kind, "question": question,
                         "status": status, "source_families": [job["family"]],
                         "source_hashes": [job["source_hash"]], "block_id": block["id"],
                         "block_sha256": block["sha256"], "block_text": block["text"],
                         "block_start_byte": block["start_byte"],
                         "block_end_byte": block["end_byte"],
                         "quote": quote, "quote_start_char": quote_start,
                         "peer": item["peer"], "scenario_origin": "pinned_readme_excerpt",
                         "review": {"accepted": False, "independent_of_author": False,
                                    "author": "deepseek-flash", "reviewer": None,
                                    "reason": "exact-span check only; semantic judgment unreviewed"}})
        except (KeyError, TypeError, ValueError) as error:
            failures.append({"index": index, "error": str(error)})
    return rows, failures


def followup_rows(jobs, answer):
    if not isinstance(answer, dict) or not isinstance(answer.get("items"), list):
        raise ValueError("teacher follow-up items missing")
    by_id = {job["id"]: job for job in jobs}
    rows, failures, seen = [], [], set()
    for index, item in enumerate(answer["items"]):
        try:
            if not isinstance(item, dict) or set(item) != {"id", "action", "query", "reason_code"}:
                raise ValueError("unexpected follow-up fields")
            job = by_id[item["id"]]
            if item["id"] in seen or item["action"] != "search" or item["reason_code"] != job["execution_state"]:
                raise ValueError("wrong action, reason or duplicate ID")
            query = item["query"]
            if (not isinstance(query, str) or not 8 <= len(query) <= 180 or
                    job["focus_object"] not in query or query in job["attempted_queries"] or
                    "site:" in query.casefold() or "http" in query.casefold()):
                raise ValueError("invalid or repeated follow-up query")
            seen.add(item["id"])
            rows.append({**job, "target_decision": {"action": "search", "query": query,
                                                     "reason_code": item["reason_code"]},
                         "review": {"accepted": False, "independent_of_author": False,
                                    "author": "deepseek-flash", "reviewer": None,
                                    "reason": "synthetic execution state; query value unreviewed"}})
        except (KeyError, TypeError, ValueError) as error:
            failures.append({"index": index, "error": str(error)})
    for job in jobs:
        if job["id"] not in seen:
            failures.append({"id": job["id"], "error": "teacher omitted case"})
    return rows, failures


def compile_candidate(row):
    if row["role"] == "evidence":
        data = {"question": row["question"], "source": {"id": row["block_id"],
                "repo": row["source_families"][0].removeprefix("repo:"),
                "text": row["block_text"]}, "execution_state": "read_success"}
        instruction = ("只判断给定来源摘录能否完整支持问题；不能把未给出的其他项目资料猜出来。"
                       "支持时返回逐字原文 quote；不完整时 status=insufficient 且 quote=null。"
                       "只输出含 status、quote 两键的 JSON。输入：")
        target = {"status": row["status"], "quote": row["quote"]}
    else:
        data = {key: row[key] for key in ("question", "cells", "focus_cell", "execution_state",
                                          "attempted_queries", "verified_evidence_ids")}
        instruction = ("只为指定未解决格子决定下一条自然检索式，不回答事实；已尝试的检索式不得原样重复。"
                       "read_without_verified_evidence 仅表示尚无已验收证据，不能推断原文没有该事实。"
                       "只输出含 action、query、reason_code 三键的 JSON。输入：")
        target = row["target_decision"]
    message = instruction + json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    prompt, _ = render_batch_prompt([{"role": "user", "content": message}],
                                    "<think></think>", "complete")
    target_text = json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {**row, "prompt": prompt, "target": target_text,
            "prompt_sha256": digest(prompt), "target_sha256": digest(target_text),
            "prompt_protocol": "rwkvos_batch_complete_no_final_lf_v1",
            "decision_protocol": "retrieval-decision-candidate-v7"}


async def generate(args):
    config = json.loads((V1 / "TEACHER-CONFIG.json").read_text())
    credential = json.loads(args.credentials.read_text())
    if (args.credentials.stat().st_mode & 0o077 or
            credential.get("base_url") != config["teacher_base_url"] or
            credential.get("model") != config["teacher_model"]):
        raise ValueError("private teacher configuration mismatch")
    jobs_path = V7 / ("EVIDENCE-JOBS.jsonl" if args.role == "evidence" else "FOLLOWUP-JOBS.jsonl")
    jobs = jsonl(jobs_path)
    selected = jobs[args.start:args.stop]
    if not selected or not 0 <= args.start < args.stop <= len(jobs):
        raise ValueError("invalid job range")
    prompt_file = V7 / ("EVIDENCE-TEACHER-v2.txt" if args.role == "evidence" else "FOLLOWUP-TEACHER-v2.txt")
    batches = [[job] for job in selected] if args.role == "evidence" else [selected[i:i + 8] for i in range(0, len(selected), 8)]
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
    write(args.out / "RUN.json", {"started_at": now(), "role": args.role,
          "start": args.start, "stop": args.stop, "jobs_sha256": file_hash(jobs_path),
          "source_manifest_sha256": file_hash(V4 / "SOURCES-CURATED.json"),
          "blocks_sha256": file_hash(BLOCKS), "teacher_prompt_sha256": file_hash(prompt_file),
          "script_sha256": file_hash(Path(__file__)), "independent_review": False,
          "training_exported": False})
    teacher = Teacher(credential, config, args.out, prior)
    queue = asyncio.Queue()
    for batch in batches:
        queue.put_nowait(batch)
    counts = Counter()

    async def worker():
        while not queue.empty() and not teacher.stopped.is_set():
            batch = queue.get_nowait()
            name = batch[0]["id"]
            result = {"job_ids": [job["id"] for job in batch], "status": "failed",
                      "rows": [], "failures": []}
            try:
                payload = batch[0] if args.role == "evidence" else {"cases": batch}
                answer = await teacher.call(name, "generation", teacher_body(config,
                    prompt_file.read_text(), payload))
                result["teacher_output"] = answer
                result["rows"], result["failures"] = (
                    evidence_rows(batch[0], answer) if args.role == "evidence" else
                    followup_rows(batch, answer))
                result["status"] = "unreviewed_drafts"
            except Exception as error:
                result["error_type"] = type(error).__name__
                result["error"] = str(error) if isinstance(error, (ValueError, RuntimeError, KeyError)) else "generation failed"
            write(args.out / "results" / f"{name}.json", result)
            counts["batches"] += 1
            counts["rows"] += len(result["rows"])
            counts["failed_rows"] += len(result["failures"])
            counts["failed_batches"] += result["status"] == "failed"
            print(json.dumps({"batch": name, "rows": len(result["rows"]),
                              "failures": len(result["failures"]),
                              "cost_upper_usd": round(teacher.spent, 4)}), flush=True)

    try:
        await asyncio.gather(*(worker() for _ in range(config["concurrency"])))
    finally:
        await teacher.close()
        write(args.out / "SUMMARY.json", {**counts, "planned_batches": len(batches),
              "cost_upper_usd": teacher.spent, "stopped": teacher.stopped.is_set(),
              "independent_review": False, "training_exported": False})


def package(args):
    out = []
    receipt = []
    for run in args.runs:
        register = json.loads((run / "RUN.json").read_text())
        summary = json.loads((run / "SUMMARY.json").read_text())
        if summary["stopped"] or register["training_exported"] or register["independent_review"]:
            raise ValueError("run stopped or status invalid")
        for path in sorted((run / "results").glob("*.json")):
            result = json.loads(path.read_text())
            if result["status"] != "unreviewed_drafts":
                continue
            out.extend(compile_candidate(row) for row in result["rows"])
        receipt.append({"run": str(run), "run_sha256": file_hash(run / "RUN.json"),
                        "summary_sha256": file_hash(run / "SUMMARY.json"),
                        "batches": summary["batches"], "rows": summary["rows"],
                        "failed_rows": summary["failed_rows"],
                        "cost_upper_usd": summary["cost_upper_usd"]})
    if len({row["id"] for row in out}) != len(out):
        raise ValueError("duplicate candidate IDs")
    save_jsonl(args.out, out)
    write(args.summary, {"schema": "retrieval_decision_candidates_v7", "rows": len(out),
          "splits": dict(Counter(row["split"] for row in out)),
          "roles": dict(Counter(row["role"] for row in out)),
          "kinds": dict(Counter(row["kind"] for row in out)),
          "statuses": dict(Counter(row.get("status", row.get("execution_state")) for row in out)),
          "candidate_sha256": file_hash(args.out), "runs": receipt,
          "independent_review": False, "admitted_training_rows": 0,
          "training_started": False})
    print(json.dumps({"rows": len(out), "roles": dict(Counter(row["role"] for row in out)),
                      "admitted_training_rows": 0}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    jobs = sub.add_parser("jobs")
    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--role", choices=("evidence", "followup"), required=True)
    generate_parser.add_argument("--start", type=int, required=True)
    generate_parser.add_argument("--stop", type=int, required=True)
    generate_parser.add_argument("--out", type=Path, required=True)
    generate_parser.add_argument("--credentials", type=Path,
                                 default=Path.home() / ".config/rwkvrag/teacher-deepseek-20260922.json")
    package_parser = sub.add_parser("package")
    package_parser.add_argument("--runs", type=Path, action="append", required=True)
    package_parser.add_argument("--out", type=Path, required=True)
    package_parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "jobs":
        evidence, followup, skipped = build_jobs()
        V7.mkdir(exist_ok=True)
        if (V7 / "EVIDENCE-JOBS.jsonl").exists() or (V7 / "FOLLOWUP-JOBS.jsonl").exists():
            raise FileExistsError("V7 jobs are already frozen; create a new version")
        save_jsonl(V7 / "EVIDENCE-JOBS.jsonl", evidence)
        save_jsonl(V7 / "FOLLOWUP-JOBS.jsonl", followup)
        write(V7 / "JOBS-SUMMARY.json", {"evidence": len(evidence),
              "followup": len(followup), "skipped_evidence_sources": skipped,
              "source_manifest_sha256": file_hash(V4 / "SOURCES-CURATED.json"),
              "blocks_sha256": file_hash(BLOCKS), "plan_inputs_sha256": {
                  str(path): file_hash(path) for path in
                  (V3 / "PLAN-CANDIDATES.jsonl", V4 / "PLAN-CANDIDATES-ADDED.jsonl")},
              "training_started": False})
        print(json.dumps({"evidence_jobs": len(evidence), "followup_jobs": len(followup),
                          "skipped": len(skipped)}))
    elif args.command == "generate":
        asyncio.run(generate(args))
    else:
        package(args)


if __name__ == "__main__":
    main()
