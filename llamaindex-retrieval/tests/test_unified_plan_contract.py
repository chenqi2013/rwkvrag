"""Bounded schemas narrow legal syntax, never establish useful retrieval or support."""
import json
from itertools import count

import pytest

from llamaindex_retrieval.schemas import SearchRequest
from llamaindex_retrieval.unified_contracts import Plan, Task, UnifiedPolicy, digest, dumps
from llamaindex_retrieval.unified_plan_contract import bounded_plan_schema
from llamaindex_retrieval.unified_retrieval import UnifiedRetrievalPipeline
from test_unified_control_cycle import SEARCH, plan, settings
from test_unified_retrieval import Adapters
from test_unified_span_pipeline import SpanModel


def task(ids=("G1",)):
    return Task.model_validate({"goals": [{"id": g, "text": "target", "refs": [
        {"message_id": "M1", "quote": "target"}]} for g in ids],
        "constraints": [], "retracted": [], "unresolved": []})


@pytest.mark.parametrize("channels,actions,web,reads,cap,available", [
    (["web"], 5, 1, 12, 1, ["web"]),
    (["web"], 6, 2, 1, 1, ["web"]),
    (["web"], 6, 0, 12, 0, None),
    (["knowledge_base"], 5, 0, 12, 5, ["knowledge_base"]),
    (["knowledge_base", "web"], 6, 1, 12, 6, ["knowledge_base", "web"]),
    (["knowledge_base", "web"], 6, 0, 12, 6, ["knowledge_base"]),
    ([], 6, 2, 12, 0, None),
    (["web"], 0, 2, 12, 0, ["web"]),
    (["knowledge_base"], 6, 2, 0, 0, ["knowledge_base"]),
])
def test_bounds_are_permission_and_resource_only(channels, actions, web, reads, cap, available):
    schema = bounded_plan_schema(task(), channels, remaining_actions=actions,
                                 remaining_web=web, read_units=reads)
    assert schema["properties"]["actions"]["maxItems"] == cap
    if available is not None:
        assert schema["$defs"]["Action"]["properties"]["channel"]["enum"] == available
    assert "contains" not in dumps(schema) and "uniqueItems" not in dumps(schema)


def test_invocation_schemas_do_not_mutate_static_or_other_requests():
    original = Plan.model_json_schema()
    ids = ['引用"\\/目标', "Other"]
    first = bounded_plan_schema(task(ids), ["web"], remaining_actions=6, remaining_web=1, read_units=12)
    second = bounded_plan_schema(task(), ["knowledge_base"], remaining_actions=6, remaining_web=2, read_units=12)
    for field in (first["properties"]["unplanned_goal_ids"], first["$defs"]["Action"]["properties"]["goal_ids"]):
        assert field["items"]["enum"] == ids and field["maxItems"] == 2
    first["properties"]["unplanned_goal_ids"]["items"]["enum"].append("mutated")
    assert second["properties"]["unplanned_goal_ids"]["items"]["enum"] == ["G1"]
    assert Plan.model_json_schema() == original


@pytest.mark.parametrize("budget", [-1, True, 1.0])
def test_invalid_internal_budgets_are_not_coerced(budget):
    with pytest.raises(ValueError, match="invalid_plan_budget"):
        bounded_plan_schema(task(), ["web"], remaining_actions=budget, remaining_web=1, read_units=12)


def test_bounded_is_explicit_cycle_only():
    assert UnifiedPolicy().plan_contract == "static_v1"
    with pytest.raises(ValueError, match="bounded_plan_requires_review_cycle_v1"):
        UnifiedPolicy(reading_protocol="span_ids_v1", plan_contract="bounded_v1")


async def test_each_round_and_revision_get_the_same_prompt_and_decoder_schema():
    model = SpanModel(mode="retrieve", channels=["web"], overrides={
        ("review", 1): SEARCH, ("plan", 2): "FAILED_PLAN_SENTINEL",
        "plan_revision": plan("web", "model next query")})
    adapters = Adapters()
    result = await UnifiedRetrievalPipeline(settings(plan_contract="bounded_v1"), None,
        model=model, adapters=adapters).ask(SearchRequest(question="target", retrieval_mode="web"))
    plans = [c for c in model.calls if c["purpose"] in {"plan", "plan_revision"}]
    assert [c["options"]["structured_schema"]["properties"]["actions"]["maxItems"] for c in plans] == [2, 1, 1]
    for call in plans:
        event = next(c for c in result.generation["model_calls"] if c["call_id"] == call["options"]["trace"]["call_id"])
        schema = call["options"]["structured_schema"]
        printed = event["messages"][0]["content"].split("只输出符合以下结构的JSON，不要代码围栏：", 1)[1].split("\n输入数据：\n", 1)[0]
        assert json.loads(printed) == schema
        assert event["plan_schema_sha256"] == digest(dumps(schema))
        assert event["plan_contract"] == "bounded_v1"
    assert plans[-1]["data"]["rejected_plan"]["raw_text"] == "FAILED_PLAN_SENTINEL"
    assert [a.query for a, _ in adapters.calls] == [json.loads(plans[0]["raw"])["actions"][0]["query"], "model next query"]
    assert result.answer == model.calls[-1]["raw"]


@pytest.mark.parametrize("bad,error", [
    ({"actions": [plan("web", str(i))["actions"][0] for i in range(3)], "unplanned_goal_ids": []}, "web_budget_exceeded"),
    ({"actions": [plan()["actions"][0]] * 2, "unplanned_goal_ids": []}, "duplicate_identity"),
    ({"actions": plan()["actions"], "unplanned_goal_ids": ["G1"]}, "planned_and_unplanned_goal_ids"),
    (plan(goals=["unknown"]), "action_outside_task"),
])
async def test_ignored_decoder_constraints_still_fail_closed_with_raw_retained(bad, error):
    model = SpanModel(mode="retrieve", channels=["knowledge_base", "web"], overrides={"plan": bad})
    adapters = Adapters()
    result = await UnifiedRetrievalPipeline(settings(plan_contract="bounded_v1", max_plan_revisions=0),
        None, model=model, adapters=adapters).ask(SearchRequest(question="target", retrieval_mode="hybrid"))
    # Acquisition still fails closed; a separate final may report the gap.
    assert not adapters.calls and model.counts["final"] == 1
    assert result.answer == model.calls[-1]["raw"]
    assert result.generation["status"] == "no_evidence"
    assert result.generation["execution_scope"]["control_failures"] == ["plan_failed"]
    assert model.calls[-1]["data"]["inherited_span_bindings"] == {}
    event = next(c for c in result.generation["model_calls"] if c["purpose"] == "plan")
    assert event["contract_error"] == error and json.loads(event["raw_text"]) == bad


async def test_static_opt_in_and_nonplan_prompts_are_unchanged(monkeypatch):
    # Only fixture transport IDs and timings are fixed; production metadata is untouched.
    monkeypatch.setattr("llamaindex_retrieval.unified_retrieval.monotonic", lambda: 100)
    models = [SpanModel(mode="retrieve", channels=["knowledge_base"]) for _ in range(2)]
    results = []
    for model, profile in zip(models, ["static_v1", "bounded_v1"], strict=True):
        ids = count()
        monkeypatch.setattr("test_unified_retrieval.uuid4", lambda: f"fixture-{next(ids)}")
        results.append(await UnifiedRetrievalPipeline(settings(plan_contract=profile), None,
            model=model, adapters=Adapters()).ask(SearchRequest(question="target")))
    for a, b in zip(results[0].generation["model_calls"], results[1].generation["model_calls"], strict=True):
        if a["purpose"] == "plan":
            assert "plan_contract" not in a and b["plan_contract"] == "bounded_v1"
        else:
            assert a["messages"] == b["messages"]
