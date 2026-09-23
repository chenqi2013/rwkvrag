"""Assemble a balanced *review queue* from immutable retrieval draft runs.

No row is approved for StateTune. The queue deliberately preserves failures
and source-level provenance, and keeps old evaluation questions out of labels.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random

from llamaindex_retrieval.retrieval_plan import training_prompt
from llamaindex_retrieval.rwkvos_batch import render_batch_prompt
from llamaindex_retrieval.schemas import ConversationMessage

from build_balanced_evidence_v8 import compile_row as compile_pair
from audit_candidates import audit, normalize

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "llamaindex-retrieval/statetune/retrieval-v1-20260923"
V3 = ROOT / "llamaindex-retrieval/statetune/retrieval-v3-20260924"
V4 = ROOT / "llamaindex-retrieval/statetune/retrieval-v4-20260924"
V7 = ROOT / "llamaindex-retrieval/statetune/retrieval-v7-20260924"
V8 = ROOT / "llamaindex-retrieval/statetune/retrieval-v8-20260924"


def digest(value):
    return sha256(value.encode("utf-8")).hexdigest()


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path, values):
    with path.open("x", encoding="utf-8") as out:
        for value in values:
            out.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def save(path, value):
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write("\n")


def review():
    return {"accepted": False, "independent_of_author": False,
            "author": "deepseek-flash", "reviewer": None,
            "reason": "source/format checked; independent semantic review pending"}


def finish(row, prompt, target, protocol):
    target_text = json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {**row, "prompt": prompt, "target": target_text,
            "prompt_sha256": digest(prompt), "target_sha256": digest(target_text),
            "prompt_protocol": "rwkvos_batch_complete_no_final_lf_v1",
            "decision_protocol": protocol, "review": review()}


def batch_message(message):
    prompt, _ = render_batch_prompt([{"role": "user", "content": message}],
                                    "<think></think>", "complete")
    return prompt


def evidence_prompt(question, sources):
    visible = [{"id": s["block_id"], "repo": s["repo"], "text": s["text"]} for s in sources]
    message = ("判断给定官方来源摘录能否完整支持问题。只按提供的文字判断；"
               "如果支持，quotes 必须给出每个所需来源的逐字连续引文；"
               "如果不足，quotes 为空数组。只输出 status、quotes 两键 JSON。输入：" +
               json.dumps({"question": question, "sources": visible,
                           "execution_state": "read_success"},
                          ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return batch_message(message)


def checked_source(source, source_map):
    registered = source_map[source["repo"]]
    if source["source_sha256"] != registered["sha256"]:
        raise ValueError("source snapshot identity changed")
    raw = (ROOT / registered["readme_path"]).read_bytes()
    if sha256(raw).hexdigest() != registered["sha256"]:
        raise ValueError("source text changed")
    start = source["block_start_byte"]
    exact = raw[start:start + len(source["text"].encode())]
    if exact.decode("utf-8") != source["text"] or digest(source["text"]) != source["block_sha256"]:
        raise ValueError("block no longer round trips")
    if source["quote"] is not None and source["quote"] not in source["text"]:
        raise ValueError("quote is not an exact source span")


def plan_candidate(case):
    history = [ConversationMessage.model_validate(item, strict=True) for item in case["history"]]
    prompt = training_prompt(case["question"], history)
    target = case["plan"]
    row = {key: case[key] for key in ("id", "split", "kind", "question",
                                      "source_families", "source_hashes", "history")}
    row["role"] = "plan"
    row["provenance"] = case.get("provenance")
    return finish(row, prompt, target, "retrieval-plan-v1")


def single_evidence(raw, source_map):
    repo = source_map[raw["source_families"][0].removeprefix("repo:")]["repo"]
    source = {"repo": repo, "block_id": raw["block_id"],
              "source_sha256": raw["source_hashes"][0],
              "block_sha256": raw["block_sha256"],
              "block_start_byte": raw["block_start_byte"],
              "text": raw["block_text"], "quote": raw["quote"]}
    checked_source(source, source_map)
    families, hashes = [source_map[repo]["family"]], [source_map[repo]["sha256"]]
    if raw["peer"] is not None:
        peer = source_map[raw["peer"]]
        if peer["split"] != raw["split"]:
            raise ValueError("comparison peer crosses split")
        families.append(peer["family"])
        hashes.append(peer["sha256"])
    row = {"id": raw["id"], "split": raw["split"], "role": "evidence",
           "kind": raw["kind"], "question": raw["question"], "status": raw["status"],
           "sources": [source], "source_families": families, "source_hashes": hashes,
           "evidence_source_hashes": [source["source_sha256"]], "peer": raw["peer"],
           "scenario_origin": "pinned_readme_excerpt"}
    quotes = [] if raw["quote"] is None else [{"source_id": raw["block_id"], "quote": raw["quote"]}]
    return finish(row, evidence_prompt(raw["question"], [source]),
                  {"status": raw["status"], "quotes": quotes},
                  "retrieval-evidence-quotes-v8")


def counterfactual(positive, original_job, source_map):
    current = positive["sources"][0]
    choices = [b for b in original_job["blocks"]
               if b["id"] != current["block_id"] and
               current["quote"] not in b["text"]]
    if not choices:
        return None
    # A different heading avoids many easy paraphrase duplicates. This is
    # still a draft label; semantic review must check that the other block
    # really cannot answer the question.
    choices.sort(key=lambda b: (b["heading_path"] == next(
        x["heading_path"] for x in original_job["blocks"] if x["id"] == current["block_id"]),
        digest(positive["id"] + b["id"])))
    block = choices[0]
    source = {"repo": current["repo"], "block_id": block["id"],
              "source_sha256": current["source_sha256"], "block_sha256": block["sha256"],
              "block_start_byte": block["start_byte"], "text": block["text"], "quote": None}
    checked_source(source, source_map)
    row = {"id": "v8-contrast-" + positive["id"], "split": positive["split"],
           "role": "evidence", "kind": "ordinary", "question": positive["question"],
           "status": "insufficient", "sources": [source],
           "source_families": [source_map[current["repo"]]["family"]],
           "source_hashes": [current["source_sha256"]],
           "evidence_source_hashes": [current["source_sha256"]],
           "contrast_of": positive["id"], "scenario_origin": "different_pinned_readme_block"}
    return finish(row, evidence_prompt(row["question"], [source]),
                  {"status": "insufficient", "quotes": []},
                  "retrieval-evidence-quotes-v8")


def followup_candidate(raw):
    input_data = {key: raw[key] for key in ("question", "cells", "focus_cell", "execution_state",
                                               "attempted_queries", "verified_evidence_ids")}
    message = ("只为指定未解决格子决定下一条自然检索式，不回答事实；已尝试的检索式不得原样重复。"
               "read_without_verified_evidence 仅表示尚无已验收证据，不能推断原文没有该事实。"
               "只输出含 action、query、reason_code 三键的 JSON。输入：" +
               json.dumps(input_data, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    row = {key: raw[key] for key in ("id", "split", "role", "kind", "source_families",
                                      "source_hashes", "plan_id", "cells", "focus_cell",
                                      "execution_state", "attempted_queries",
                                      "verified_evidence_ids", "scenario_origin")}
    row["original_question"] = raw["question"]
    row["question"] = (raw["question"] + "｜待补查格子：" +
                       raw["focus_object"] + " / " + raw["focus_dimension"])
    return finish(row, batch_message(message), raw["target_decision"],
                  "retrieval-followup-candidate-v8")


def stop_candidate(plan, index):
    data = {"question": plan["question"], "cells": [],
            "execution_state": "no_unresolved_cells", "attempted_queries": [],
            "verified_evidence_ids": []}
    message = ("根据任务格子的执行状态决定是否继续检索。没有未解决格子时停止；"
               "不要补造来源。只输出含 action、query、reason_code 三键的 JSON。输入：" +
               json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    row = {"id": f"v8-stop-{index:03d}", "split": plan["split"],
           "role": "followup", "kind": plan["kind"],
           "question": plan["question"] + "｜无未解决格子",
           "original_question": plan["question"],
           "source_families": plan["source_families"],
           "source_hashes": plan["source_hashes"],
           "execution_state": "no_unresolved_cells", "scenario_origin": "synthetic_empty_worklist"}
    return finish(row, batch_message(message),
                  {"action": "stop", "query": None, "reason_code": "no_unresolved_cells"},
                  "retrieval-followup-candidate-v8")


def extract_runs(paths, role, expected_sha):
    result, receipts = [], []
    for run in paths:
        manifest = json.loads((run / "RUN.json").read_text())
        summary = json.loads((run / "SUMMARY.json").read_text())
        if (manifest.get("training_exported") or manifest.get("independent_review") or
                summary.get("stopped") or (role != "pair" and manifest["role"] != role)):
            raise ValueError("invalid or stopped run")
        if (role == "pair" and manifest["jobs_sha256"] != expected_sha) or (
                role != "pair" and manifest["source_manifest_sha256"] != expected_sha):
            raise ValueError("run source binding changed")
        receipt = {"path": str(run), "run_sha256": file_hash(run / "RUN.json"),
                   "summary_sha256": file_hash(run / "SUMMARY.json"),
                   "rows": summary["rows"], "failures": summary["failed_rows"],
                   "cost_upper_usd": summary["cost_upper_usd"]}
        receipts.append(receipt)
        for path in sorted((run / "results").glob("*.json")):
            batch = json.loads(path.read_text())
            if batch["status"] == "unreviewed_drafts":
                result.extend(batch["rows"])
    if len({row["id"] for row in result}) != len(result):
        raise ValueError("duplicate raw candidate ID")
    return result, receipts


def choose(values, count, seed):
    rng = random.Random(seed)
    copy = values.copy()
    rng.shuffle(copy)
    return copy[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v7-evidence", "v7-followup", "v8-pair"):
        parser.add_argument("--" + name, type=Path, action="append", required=True)
    args = parser.parse_args()
    source_path = V4 / "SOURCES-CURATED.json"
    sources = json.loads(source_path.read_text())["sources"]
    source_map = {s["repo"]: s for s in sources}
    source_map.update({s["family"].removeprefix("repo:"): s for s in sources})
    evidence_raw, evidence_receipts = extract_runs(args.v7_evidence, "evidence", file_hash(source_path))
    followup_raw, followup_receipts = extract_runs(args.v7_followup, "followup", file_hash(source_path))
    pair_raw, pair_receipts = extract_runs(args.v8_pair, "pair", file_hash(V8 / "PAIR-JOBS.jsonl"))
    plans_raw = read_rows(V3 / "PLAN-CANDIDATES.jsonl") + read_rows(V4 / "PLAN-CANDIDATES-ADDED.jsonl")
    plan_rows = [plan_candidate(row) for row in plans_raw]
    single_rows = [single_evidence(row, source_map) for row in evidence_raw]
    pair_rows = [compile_pair(row) for row in pair_raw]
    for row in pair_rows:
        for source in row["sources"]:
            checked_source(source, source_map)
    job_by_repo = {j["repo"]: j for j in read_rows(V7 / "EVIDENCE-JOBS.jsonl")}
    contrast_rows = [counterfactual(row, job_by_repo[row["sources"][0]["repo"]], source_map)
                     for row in single_rows if row["kind"] == "ordinary" and row["status"] == "supported"]
    contrast_rows = [row for row in contrast_rows if row is not None]
    followup_rows = [followup_candidate(row) for row in followup_raw]
    used_plan_ids = {row["plan_id"] for row in followup_raw}
    stop_plans = [row for row in plans_raw if row["id"] not in used_plan_ids and row["split"] == "train"]
    stop_rows = [stop_candidate(plan, index) for index, plan in enumerate(choose(stop_plans, 150, 84024))]
    all_rows = plan_rows + single_rows + contrast_rows + pair_rows + followup_rows + stop_rows
    if len({row["id"] for row in all_rows}) != len(all_rows):
        raise ValueError("assembled IDs overlap")

    # This is a review queue, not an acceptance list. Sampling caps the
    # supported/insufficient correlations found in the first teacher pilot.
    shortlist_ids = json.loads((V4 / "PLAN-REVIEW-SHORTLIST.json").read_text())["selected_ids"]
    plan_by_id = {row["id"]: row for row in plan_rows}
    selected_plans = [plan_by_id[id_] for id_ in shortlist_ids]
    extra = [row for row in plan_rows if row["id"] not in set(shortlist_ids) and
             row["kind"] in {"ordinary", "comparison", "selection", "history"}]
    selected_plans += choose(extra, 100, 82401)
    ordinary_pos = [r for r in single_rows if r["kind"] == "ordinary" and r["status"] == "supported" and r["split"] == "train"]
    single_neg = [r for r in single_rows if r["status"] == "insufficient" and r["split"] == "train"]
    pair_pos = [r for r in pair_rows if r["kind"] == "comparison" and r["status"] == "supported" and r["split"] == "train"]
    pair_neg = [r for r in pair_rows if r["kind"] == "ordinary" and r["status"] == "insufficient" and r["split"] == "train"]
    selected_evidence = (choose(ordinary_pos, 700, 82402) +
                         choose([r for r in contrast_rows if r["split"] == "train"], 450, 82403) +
                         choose(single_neg, 400, 82404) +
                         pair_pos + pair_neg)
    if len(selected_evidence) < 1800:
        need = 1800 - len(selected_evidence)
        unused = [r for r in single_rows if r["split"] == "train" and
                  r["id"] not in {s["id"] for s in selected_evidence}]
        selected_evidence += choose(unused, need, 82405)
    selected = selected_plans + selected_evidence + followup_rows + stop_rows
    if len({r["id"] for r in selected}) != len(selected):
        raise ValueError("selected IDs overlap")
    exclusions = json.loads((V1 / "EXCLUSIONS.json").read_text())
    minimums = json.loads((V1 / "MINIMUMS.json").read_text())
    audit_result = audit(selected, exclusions, minimums)
    # Full per-row review errors are predictable and large; save an honest
    # summary while preserving the exact candidates and failing audit status.
    issue_counts = Counter("independent review missing" if "independent review not recorded" in item
                           else item.split(":", 1)[-1].strip() for item in audit_result["issues"])
    V8.mkdir(exist_ok=True)
    all_path = V8 / "CANDIDATES.jsonl"
    review_path = V8 / "REVIEW-QUEUE.jsonl"
    write_rows(all_path, all_rows)
    write_rows(review_path, selected)
    summary = {"schema": "retrieval_review_queue_v8", "candidate_rows": len(all_rows),
               "review_queue_rows": len(selected),
               "all_splits": dict(Counter(r["split"] for r in all_rows)),
               "all_role_status": dict(Counter(f"{r['role']}:{r.get('status', r.get('execution_state', ''))}"
                                               for r in all_rows)),
               "review_roles": dict(Counter(r["role"] for r in selected)),
               "review_kinds": dict(Counter(r["kind"] for r in selected)),
               "review_evidence_kind_status": dict(Counter(f"{r['kind']}:{r['status']}"
                                                          for r in selected if r["role"] == "evidence")),
               "unique_normalized_questions": len({normalize(r["question"]) for r in selected}),
               "audit": {key: val for key, val in audit_result.items() if key != "issues"},
               "audit_issue_counts": dict(issue_counts), "audit_examples": audit_result["issues"][:20],
               "candidate_sha256": file_hash(all_path), "review_queue_sha256": file_hash(review_path),
               "input_hashes": {str(path): file_hash(path) for path in
                                (source_path, V3 / "PLAN-CANDIDATES.jsonl",
                                 V4 / "PLAN-CANDIDATES-ADDED.jsonl",
                                 V7 / "EVIDENCE-JOBS.jsonl", V7 / "FOLLOWUP-JOBS.jsonl",
                                 V8 / "PAIR-JOBS.jsonl")},
               "run_receipts": evidence_receipts + followup_receipts + pair_receipts,
               "independent_review": False, "admitted_training_rows": 0,
               "training_started": False}
    save(V8 / "ASSEMBLY-SUMMARY.json", summary)
    print(json.dumps({"candidates": len(all_rows), "review_queue": len(selected),
                      "roles": summary["review_roles"], "audit_admitted": audit_result["admitted"],
                      "admitted_training_rows": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
