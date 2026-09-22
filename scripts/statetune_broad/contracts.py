"""Teacher output validation and existing student prompt compilation; never repair labels."""
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llamaindex-retrieval/src"))
from llamaindex_retrieval.typed_funnel_contract_v7 import Atomic, validate_atomic
from llamaindex_retrieval.typed_funnel_v8 import atomic_prompt
from llamaindex_retrieval.typed_funnel_v10 import empty_evidence_prompt
from llamaindex_retrieval.writer_prompt import writer_prompt_v2
from llamaindex_retrieval.writer_evidence_handoff import with_evidence_handoff


def source_map(job, draft):
    if job["material_mode"] == "real":
        if draft["sources"] != []:
            raise ValueError("Real-source job added or rewrote material")
        sources = job["sources"]
    else:
        sources = draft["sources"]
        if not 2 <= len(sources) <= 5:
            raise ValueError("Synthetic sources outside declared range")
    result = {}
    for s in sources:
        if (not isinstance(s["id"], str) or not s["id"] or s["id"] in result
                or not isinstance(s["title"], str) or not isinstance(s["text"], str) or not s["text"].strip()):
            raise ValueError("Invalid source record")
        if job["material_mode"] == "synthetic" and not 200 <= len(s["text"]) <= 1400:
            raise ValueError("Synthetic source length outside declared range")
        result[s["id"]] = s
    if not isinstance(draft["items"], list) or not 4 <= len(draft["items"]) <= 8:
        raise ValueError("Expected 4..8 distinct items")
    return result


def history(item):
    value = item.get("history", [])
    if not isinstance(value, list) or len(value) > 8 or any(
            x.get("role") != "user" or not isinstance(x.get("content"), str) or not x["content"].strip() for x in value):
        raise ValueError("Invalid user history")
    return value


def compile_item(item, sources):
    kind = item["kind"]
    if not all(isinstance(item.get(k), str) and item[k].strip() for k in ["rationale", "skill"]):
        raise ValueError("Missing explanation or actual decision skill")
    if kind == "atomic":
        s = sources[item["source_id"]]
        field = item["field"]
        if field["value_type"] not in {"boolean", "quantity", "text"}:
            raise ValueError("Unknown field type")
        if not all(isinstance(x, str) and x.strip() for x in [item["object"], field["name"], field["question"]]):
            raise ValueError("Empty object/field")
        units = {"E1": SimpleNamespace(text=s["text"])}
        validate_atomic(Atomic.model_validate(item["answer"]), field, units)
        prompt = atomic_prompt(item["object"], field, SimpleNamespace(title=s["title"]), units)
        target = json.dumps(item["answer"], ensure_ascii=False)
        role = "resolver"
    elif kind in {"writer", "status"}:
        if not isinstance(item["question"], str) or not item["question"].strip():
            raise ValueError("Empty question")
        target = item["answer"]
        if not isinstance(target, str) or not 1 <= len(target) <= 1200:
            raise ValueError("Invalid final answer")
        task = json.dumps({"history": history(item), "latest_question": item["question"]}, ensure_ascii=False)
        if kind == "writer":
            evidence = []
            if not isinstance(item["evidence"], list) or len(item["evidence"]) > 8:
                raise ValueError("Too many evidence pieces")
            for i, e in enumerate(item["evidence"], 1):
                text = sources[e["source_id"]]["text"]
                q = e["quote"]
                if not isinstance(q, str) or not 1 <= len(q) <= 800 or q not in text:
                    raise ValueError("Nonliteral Writer evidence")
                evidence.append({"label": f"资料 {i}", "text": q})
            cites = re.findall(r"\[资料[^\]]*\]", target)
            if evidence and not cites:
                raise ValueError("Evidence answer has no citations")
            for c in cites:
                m = re.fullmatch(r"\[资料\s*(\d+)\]", c)
                if not m or not 1 <= int(m[1]) <= len(evidence):
                    raise ValueError("Foreign or invalid citation")
            prompt = writer_prompt_v2(task, evidence, [])
        else:
            flow = item["flow"]
            keys = {"input_source_count", "writer_source_count", "failed_node_count", "failed_cell_count", "unexamined_job_count"}
            if set(flow) != keys or any(type(flow[k]) is not int or flow[k] < 0 for k in keys):
                raise ValueError("Invalid execution counters")
            if not 1 <= flow["input_source_count"] <= 30 or flow["writer_source_count"] != 0:
                raise ValueError("Invalid no-selected-evidence state")
            if re.search(r"\[资料", target):
                raise ValueError("Status-only output fabricated citations")
            prompt = with_evidence_handoff(empty_evidence_prompt(task), dict(flow, protocol="evidence-flow-v1", state="no_selected_evidence"))
        role = "writer"
    else:
        raise ValueError("Unknown kind")
    return {"kind": kind, "state_role": role, "prompt_body": prompt,
        "prompt": "User: " + prompt + "\n\nAssistant: <think></think>\n", "target": target}


def review_map(review, size):
    if type(review.get("source_set_valid")) is not bool:
        raise ValueError("Missing source-set review")
    result = {}
    for r in review["reviews"]:
        i = r["index"]
        if type(i) is not int or not 0 <= i < size or i in result or type(r["accept"]) is not bool:
            raise ValueError("Missing/duplicate/invalid review index")
        if not isinstance(r.get("reason"), str) or not r["reason"].strip() or not isinstance(r.get("issues"), list):
            raise ValueError("Review lacks reason")
        if r["accept"] and r["issues"]:
            raise ValueError("Accepted review lists unresolved issues")
        result[i] = dict(r, accept=r["accept"] and review["source_set_valid"])
    if len(result) != size:
        raise ValueError("Incomplete review")
    return result
