"""Execution provenance only: never infer that a source omits a fact."""
from collections import Counter


def evidence_flow(graph):
    inputs = graph.get("input_source_ids")
    writer = graph.get("writer_source_ids")
    input_count = len(set(inputs)) if isinstance(inputs, list) else None
    writer_count = len(set(writer)) if isinstance(writer, list) else None
    failures = graph.get("failures") or []
    cells = graph.get("cells") or []
    facts = graph.get("facts") or []
    omitted = (graph.get("call_budget") or {}).get("unexamined_jobs") or []
    selected = {identity for cell in cells for identity in cell.get("fact_ids", [])}
    verified = {f["id"] for f in facts if f.get("id") is not None
                and (f.get("verification") or {}).get("verdict") == "supported"}
    stages = Counter()
    for failure in failures:
        # Stage tags are protocol metadata, not question/entity classifications.
        purpose = failure.get("purpose", "")
        stage = purpose.split(":", 1)[0] or "unrecorded"
        stages[stage] += 1
    if input_count is None or writer_count is None:
        state = "unrecorded"
    elif input_count == 0 and writer_count == 0:
        state = "no_input_materials"
    elif input_count == 0 or not set(writer).issubset(set(inputs)):
        state = "inconsistent_trace"
    elif writer_count == 0:
        state = "no_selected_evidence"
    else:
        state = "selected_evidence_present"
    return {
        "protocol": "evidence-flow-v1", "state": state,
        "input_source_count": input_count, "writer_source_count": writer_count,
        "fact_proposal_count": len(facts), "verified_fact_count": len(verified),
        "verified_unselected_fact_ids": sorted(verified - selected),
        "failed_node_count": len(failures), "failures_by_stage": dict(stages),
        "failed_cell_count": sum(c.get("execution_status") == "failed" for c in cells),
        "unexamined_job_count": len(omitted),
        "source_absence_established": False,
        "semantic_support_verified": False,
    }
