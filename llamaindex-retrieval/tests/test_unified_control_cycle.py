"""Bounded model-controlled recovery; fixtures are not semantic quality evidence."""
import asyncio
import json

import pytest

from llamaindex_retrieval.schemas import SearchRequest
from llamaindex_retrieval.unified_retrieval import UnifiedRetrievalPipeline
from test_unified_span_pipeline import SpanModel, settings as span_settings
from test_unified_retrieval import Adapters, material


def settings(**kwargs):
    return span_settings(control_protocol="review_cycle_v1", **kwargs)


def plan(channel="knowledge_base", query="query", goals=None):
    return {"actions": [{"channel": channel, "query": query, "goal_ids": goals or ["G1"]}],
            "unplanned_goal_ids": []}


SEARCH = {"next": "search", "missing": ["need another source"], "reason": "fixture gap"}


@pytest.mark.parametrize("method", ["ask", "search"])
async def test_review_redirection_expands_only_with_new_model_decision_and_preserves_scope(method):
    model = SpanModel(mode="retrieve", channels=["knowledge_base"], overrides={
        ("review", 1): SEARCH, ("direction", 2): {"choice": "web"},
        ("plan", 2): plan("web", "next query")})
    adapters = Adapters([[material(i) for i in range(20)], [material(100)]])
    response = await getattr(UnifiedRetrievalPipeline(settings(max_units=6), None,
        model=model, adapters=adapters), method)(SearchRequest(
            question="original request", retrieval_mode="auto", knowledge_base_id="only-this-kb"))
    assert [a.channel for a, _ in adapters.calls] == ["knowledge_base", "web"]
    assert all(r.knowledge_base_id == "only-this-kb" for _, r in adapters.calls)
    assert model.counts["direction"] == 2 and model.counts["leaf"] == 4
    directions = response.retrieval["directions"]
    assert [d["choice"] for d in directions] == ["knowledge_base", "web"]
    second = [c for c in model.calls if c["purpose"] == "plan"][1]
    assert second["data"]["selected_channels"] == ["web"]
    assert second["data"]["latest_user_message"]["content"] == "original request"
    assert second["data"]["round_read_units"] == 3
    assert response.retrieval["protocol"] == "rwkv-unified-v4"
    assert [b["read_limit"] for b in response.retrieval["execution_scope"]["round_budgets"]] == [3, 3]
    assert model.counts["final"] == int(method == "ask")
    if method == "ask":
        assert response.answer == model.calls[-1]["raw"]


async def test_second_direction_cannot_expand_caller_permissions_and_root_still_reserved():
    model = SpanModel(mode="retrieve", channels=["knowledge_base"], overrides={
        ("review", 1): SEARCH, ("direction", 2): {"choice": "web"}})
    adapters = Adapters()
    result = await UnifiedRetrievalPipeline(settings(), None, model=model, adapters=adapters).ask(
        SearchRequest(question="target", retrieval_mode="knowledge_base"))
    assert len(adapters.calls) == 1 and model.counts["plan"] == 1
    assert result.generation["status"] == "partial"
    assert result.generation["execution_scope"]["control_failures"] == ["routing_failed"]
    assert model.counts["final"] == 1
    assert result.retrieval["directions"][-1]["status"] == "rejected"


@pytest.mark.parametrize("invalid", ["INVALID_PLAN_SENTINEL", ("length", "TRUNCATED_PLAN_SENTINEL"),
    {"actions": [plan()["actions"][0]] * 2, "unplanned_goal_ids": []},
    {"actions": plan()["actions"], "unplanned_goal_ids": ["G1"]}, plan("web")])
async def test_rejected_plan_has_unchanged_raw_and_a_new_model_call_not_program_repair(invalid):
    model = SpanModel(mode="retrieve", channels=["knowledge_base"], overrides={
        ("plan", 1): invalid, "plan_revision": plan(query="model authored revision")})
    adapters = Adapters()
    result = await UnifiedRetrievalPipeline(settings(), None, model=model, adapters=adapters).ask(
        SearchRequest(question="target"))
    assert [a.query for a, _ in adapters.calls] == ["model authored revision"]
    calls = result.generation["model_calls"]
    old = next(c for c in calls if c["purpose"] == "plan")
    new = next(c for c in calls if c["purpose"] == "plan_revision")
    repair = next(c for c in model.calls if c["purpose"] == "plan_revision")
    assert repair["data"]["rejected_plan"]["raw_text"] == old["raw_text"]
    assert new["revision_of_call_id"] == old["call_id"] != new["call_id"]
    assert new["revision_of_raw_sha256"] == old["raw_text_sha256"]
    assert [p["status"] for p in result.retrieval["plan_attempts"]] == ["rejected", "admitted"]
    for call in model.calls:
        if call["purpose"] in {"review", "merge", "final"}:
            assert "rejected_plan" not in call["data"]
            assert "PLAN_SENTINEL" not in json.dumps(call["data"])
    assert result.generation["execution_scope"]["plan_revision_calls"] == 1


@pytest.mark.parametrize("revisions", [0, 1, 2])
async def test_revision_limit_never_loops_or_runs_invalid_actions(revisions):
    model = SpanModel(mode="retrieve", channels=["knowledge_base"], overrides={
        "plan": "INVALID", "plan_revision": "ALSO_INVALID"})
    adapters = Adapters()
    result = await UnifiedRetrievalPipeline(settings(max_plan_revisions=revisions), None,
        model=model, adapters=adapters).ask(SearchRequest(question="target"))
    assert not adapters.calls and model.counts["plan_revision"] == revisions
    assert result.generation["status"] == "no_evidence"
    assert result.generation["execution_scope"]["control_failures"] == ["plan_failed"]
    assert model.counts["final"] == 1 and result.answer == model.calls[-1]["raw"]


async def test_native_transport_failure_is_not_a_revision_retry():
    model = SpanModel(mode="retrieve", channels=["knowledge_base"], overrides={
        "plan": ("transport_error", None)})
    adapters = Adapters()
    result = await UnifiedRetrievalPipeline(settings(), None, model=model, adapters=adapters).ask(
        SearchRequest(question="target"))
    assert model.counts["plan_revision"] == 0 and not adapters.calls
    assert result.generation["status"] == "model_failed"


async def test_revision_must_not_expand_direction_even_when_caller_permits_channel():
    model = SpanModel(mode="retrieve", channels=["knowledge_base"], overrides={
        "plan": "BAD", "plan_revision": plan("web")})
    adapters = Adapters()
    result = await UnifiedRetrievalPipeline(settings(), None, model=model, adapters=adapters).ask(
        SearchRequest(question="target", retrieval_mode="auto"))
    assert not adapters.calls
    assert result.retrieval["plan_attempts"][-1]["contract_error"] == "action_outside_task"


async def test_provided_direction_can_be_reconsidered_but_material_entry_is_closed():
    model = SpanModel(overrides={("review", 1): SEARCH, ("direction", 2): {"choice": "web"},
                                 "plan": plan("web")})
    adapters = Adapters()
    result = await UnifiedRetrievalPipeline(settings(), None, model=model, adapters=adapters).ask(
        SearchRequest(question="target", retrieval_mode="web"))
    assert len(adapters.calls) == 1 and model.counts["direction"] == 2
    assert result.retrieval["directions"][0]["choice"] == "provided"
    closed = SpanModel()
    result = await UnifiedRetrievalPipeline(settings(max_units=4), None, model=closed,
        adapters=Adapters()).ask_materials("target", [material(i) for i in range(4)])
    assert closed.counts["direction"] == closed.counts["plan"] == closed.counts["review"] == 0
    assert closed.counts["leaf"] == 4 and closed.counts["final"] == 1
    assert result.retrieval["routing"]["model_decision"] is False


async def test_round_robin_actions_prevents_first_candidate_list_starving_second_channel():
    both = {"actions": [plan()["actions"][0], plan("web")["actions"][0]], "unplanned_goal_ids": []}
    model = SpanModel(mode="retrieve", channels=["knowledge_base", "web"], overrides={"plan": both})
    adapters = Adapters([[material(i) for i in range(20)], [material(100)]])
    result = await UnifiedRetrievalPipeline(settings(max_units=4), None, model=model, adapters=adapters).ask(
        SearchRequest(question="target", retrieval_mode="hybrid"))
    assert model.counts["leaf"] == 2
    originals = [c["data"]["original_material"]["document_id"] for c in model.calls if c["purpose"] == "leaf"]
    assert originals == ["d0", "d100"]
    assert result.generation["execution_scope"]["unread_units"] == 19


async def test_plan_exceeding_round_read_capacity_is_rejected_without_truncation():
    many = {"actions": [plan(query=str(i))["actions"][0] for i in range(3)], "unplanned_goal_ids": []}
    model = SpanModel(mode="retrieve", channels=["knowledge_base"], overrides={"plan": many})
    adapter = Adapters()
    result = await UnifiedRetrievalPipeline(settings(max_units=4, max_plan_revisions=0), None,
        model=model, adapters=adapter).ask(SearchRequest(question="target"))
    assert not adapter.calls and result.generation["status"] == "no_evidence"
    assert result.generation["execution_scope"]["control_failures"] == ["plan_failed"]
    assert model.counts["final"] == 1 and result.answer == model.calls[-1]["raw"]
    assert result.retrieval["plan_attempts"][0]["contract_error"] == "round_read_budget_exceeded"


async def test_second_round_repair_does_not_spend_reserved_final_calls():
    # Task+Direction+Plan+leaf+Review=5; next Direction+bad Plan=7;
    # max_calls=10 leaves no space for revision+leaf+Review+final, but keeps root.
    model = SpanModel(mode="retrieve", channels=["knowledge_base"], overrides={
        ("review", 1): SEARCH, ("plan", 2): "BAD"})
    result = await UnifiedRetrievalPipeline(settings(max_calls=10), None, model=model,
        adapters=Adapters()).ask(SearchRequest(question="target"))
    assert model.counts["plan_revision"] == 0 and model.counts["final"] == 1
    assert result.generation["status"] == "partial" and result.answer == model.calls[-1]["raw"]
    assert result.generation["request_budget"]["admitted_calls"] == 8


async def test_revision_timeout_preserves_audit_and_returns_no_intermediate():
    async def slow(_):
        await asyncio.sleep(10)
    model = SpanModel(mode="retrieve", channels=["knowledge_base"], overrides={
        "plan": "BAD", "plan_revision": slow})
    result = await UnifiedRetrievalPipeline(settings(timeout_seconds=0.03), None, model=model,
        adapters=Adapters()).ask(SearchRequest(question="target"))
    assert result.answer == "" and result.generation["status"] == "request_timeout"
    assert result.generation["model_calls"][-1]["status"] == "cancelled"
    assert result.generation["model_calls"][-2]["raw_text"] == "BAD"


async def test_followup_direction_can_request_clarification_without_intermediate_answer():
    model = SpanModel(mode="retrieve", channels=["knowledge_base"], overrides={
        ("review", 1): SEARCH, ("direction", 2): {"choice": "clarify"}})
    result = await UnifiedRetrievalPipeline(settings(), None, model=model, adapters=Adapters()).ask(
        SearchRequest(question="target"))
    assert result.generation["status"] == "needs_clarification"
    assert model.counts["final"] == 0 and model.counts["clarify"] == 1
    assert result.answer == model.calls[-1]["raw"]


async def test_first_round_reserves_minimum_followup_calls_not_only_read_units():
    model = SpanModel(mode="retrieve", channels=["knowledge_base"], overrides={("review", 1): SEARCH})
    adapters = Adapters([[material(i) for i in range(30)], [material(100)]])
    result = await UnifiedRetrievalPipeline(settings(max_calls=20), None, model=model, adapters=adapters).ask(
        SearchRequest(question="target"))
    assert len(adapters.calls) == 2 and model.counts["direction"] == 2
    assert model.counts["leaf"] == 9 and model.counts["final"] == 1
    assert result.generation["request_budget"]["admitted_calls"] == 20
    assert result.retrieval["execution_scope"]["round_budgets"][0]["minimum_followup_capacity_reserved"]


async def test_failed_reading_raw_is_withheld_from_redirection_and_revision():
    model = SpanModel(mode="retrieve", channels=["knowledge_base"], overrides={
        ("leaf", 1): "FAILED_READING_SENTINEL", ("review", 1): SEARCH,
        ("plan", 2): "BAD", "plan_revision": plan(query="revised")})
    result = await UnifiedRetrievalPipeline(settings(), None, model=model,
        adapters=Adapters([[material(0)], [material(1)]])).ask(SearchRequest(question="target"))
    assert result.generation["nodes"][0]["raw_text"] == "FAILED_READING_SENTINEL"
    for call in model.calls:
        if call["purpose"] in {"direction", "plan_revision", "review", "final"}:
            assert "FAILED_READING_SENTINEL" not in json.dumps(call["data"])
    assert model.counts["final"] == 1


def test_cycle_requires_span_reading_and_bounded_revisions():
    from llamaindex_retrieval.unified_contracts import UnifiedPolicy
    with pytest.raises(ValueError, match="review_cycle_requires_span_ids"):
        UnifiedPolicy(control_protocol="review_cycle_v1")
    with pytest.raises(ValueError):
        settings(max_plan_revisions=3)
