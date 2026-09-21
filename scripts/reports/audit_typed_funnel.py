"""Structural audit only; no output edits and no inferred semantic score."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def audit(name):
    inputs = json.loads((ROOT / "llamaindex-retrieval/eval/typed-funnel-integration-20260921/INPUTS.json").read_text())
    run = ROOT / "data/quality-runs" / name / "run1"
    records = []
    for i, task in enumerate(inputs):
        path = run / f"{i:03d}.json"
        if not path.exists():
            raise RuntimeError("Incomplete run")
        response = json.loads(path.read_text())["response"]
        graph = response.get("retrieval", {}).get("funnel", {})
        original = {s["id"]: s for s in task["sources"]}
        facts = graph.get("facts", [])
        bad_offsets = [f["id"] for f in facts if original[f["source_id"]]["snippet"][f["start"]:f["end"]] != f["quote"]]
        calls = response["generation"]["model_calls"]
        writer = [c for c in calls if c.get("stage") == "writer"]
        generated = [c for c in calls if c.get("completion_attempted")]
        records.append({"ordinal": i, "id": task["id"], "status": response["generation"]["status"],
            "calls": len(calls), "generated_calls": len(generated), "source_offsets_invalid": bad_offsets,
            "answer_matches_raw": response["answer"] == (response["generation"].get("raw_model_answer") or ""),
            "facts": len(facts), "verified_facts": sum((f.get("verification") or {}).get("verdict") == "supported" for f in facts),
            "cell_count": len(graph.get("cells", [])),
            "completed_cells": sum(c.get("execution_status") == "completed" for c in graph.get("cells", [])),
            "failed_nodes": len(graph.get("failures", [])),
            "node_budget_exhaustions": sum(f.get("status") == "call_budget_exceeded" for f in graph.get("failures", [])),
            "writer_finish_reason": writer[-1].get("finish_reason") if writer else None,
            "writer_status": writer[-1].get("status") if writer else None,
            "unexamined_jobs": len(graph.get("call_budget", {}).get("unexamined_jobs", []))})
    result = {"kind": "structural audit, not semantic accuracy", "records": records,
        "status_counts": dict(Counter(r["status"] for r in records)),
        "total_calls": sum(r["calls"] for r in records),
        "max_calls": max(r["calls"] for r in records),
        "invalid_source_offsets": sum(len(r["source_offsets_invalid"]) for r in records),
        "altered_answers": sum(not r["answer_matches_raw"] for r in records),
        "node_budget_exhaustions": sum(r["node_budget_exhaustions"] for r in records)}
    target = run.parent / "STRUCTURAL-AUDIT.json"
    with target.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return {k:v for k,v in result.items() if k != "records"}


if __name__ == "__main__":
    import sys
    for name in sys.argv[1:]:
        print(name, audit(name))
