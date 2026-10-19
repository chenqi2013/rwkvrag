"""Generic input-order/provenance checks, not semantic repair guarantees."""
import json

import pytest

from llamaindex_retrieval.schemas import SearchRequest
from llamaindex_retrieval.unified_retrieval import UnifiedRetrievalPipeline
from test_unified_control_cycle import plan, settings
from test_unified_retrieval import Adapters
from test_unified_span_pipeline import SpanModel


@pytest.mark.parametrize("profile", ["static_v1", "bounded_v1"])
async def test_revision_record_precedes_unchanged_current_context_for_each_attempt(profile):
    model = SpanModel(mode="retrieve", channels=["knowledge_base"], overrides={
        "plan": "REJECTED_RAW\n{not json", ("plan_revision", 1): ("length", "TRUNCATED_RAW\n{"),
        ("plan_revision", 2): plan(query="model authored replacement")})
    adapters = Adapters()
    result = await UnifiedRetrievalPipeline(settings(plan_contract=profile, max_plan_revisions=2),
        None, model=model, adapters=adapters).ask(SearchRequest(question="original request"))
    controls = [c for c in result.generation["model_calls"] if c["purpose"] in {"plan", "plan_revision"}]
    inputs = [c["data"] for c in model.calls if c["purpose"] in {"plan", "plan_revision"}]
    assert len(controls) == 3
    assert "rejected_plan" not in inputs[0] and "revision_input_layout" not in controls[0]
    for index in [1, 2]:
        old, new = controls[index - 1:index + 1]
        payload = inputs[index]
        assert list(payload)[0] == "rejected_plan"
        assert list(payload)[1:] == list(inputs[0])
        assert payload["rejected_plan"] == {**{k: old.get(k) for k in (
            "call_id", "raw_text", "raw_text_sha256", "status", "contract_status",
            "contract_errors", "contract_error")}, "kind": "rejected_control_output_not_evidence"}
        assert new["revision_input_layout"] == "rejected_first_v1"
        assert new["revision_of_call_id"] == old["call_id"] != new["call_id"]
        for key in ("history", "latest_user_message", "task", "allowed_channels", "selected_channels",
                    "direction", "round", "round_read_units", "entry_scope", "previous_actions", "reviews"):
            assert payload[key] == inputs[0][key]
        assert payload["remaining"]["model_calls"] == inputs[0]["remaining"]["model_calls"] - index
    assert [a.query for a, _ in adapters.calls] == ["model authored replacement"]
    for c in model.calls:
        if c["purpose"] not in {"plan", "plan_revision"}:
            assert "rejected_plan" not in c["data"]
            assert "REJECTED_RAW" not in json.dumps(c["data"])
            assert "TRUNCATED_RAW" not in json.dumps(c["data"])


@pytest.mark.parametrize("profile", ["static_v1", "bounded_v1"])
async def test_reordered_context_still_rejects_the_same_overlapping_plan(profile):
    rejected = {"actions": plan()["actions"], "unplanned_goal_ids": ["G1"]}
    model = SpanModel(mode="retrieve", channels=["knowledge_base"], overrides={
        "plan": rejected, "plan_revision": rejected})
    adapters = Adapters()
    result = await UnifiedRetrievalPipeline(settings(plan_contract=profile), None,
        model=model, adapters=adapters).ask(SearchRequest(question="target"))
    assert not adapters.calls and result.answer == model.calls[-1]["raw"]
    assert result.generation["status"] == "no_evidence" and model.counts["final"] == 1
    assert result.generation["execution_scope"]["control_failures"] == ["plan_failed"]
    assert [p["contract_error"] for p in result.retrieval["plan_attempts"]] == [
        "planned_and_unplanned_goal_ids", "planned_and_unplanned_goal_ids"]
    calls = [c for c in result.generation["model_calls"] if c["purpose"] in {"plan", "plan_revision"}]
    assert calls[0]["raw_text"] == calls[1]["raw_text"]
