"""Engineering fixtures only, not RWKV answer-quality or training evidence."""
import json

import pytest

from llamaindex_retrieval.schemas import SearchRequest
from llamaindex_retrieval.unified_contracts import digest
from llamaindex_retrieval.unified_retrieval import UnifiedRetrievalPipeline
from test_unified_retrieval import Adapters, Model, material, settings as v2_settings
from test_unified_span_pipeline import SpanModel, settings

CONTROLS = ["single_direction_v1", "review_cycle_v1"]
LITERALS = ["[资料 99]", "[资料 x]", "[Source 99]", "[Source x]",
            "`[资料 99]`", "```text\r\n[资料 99]\r\n```", "[99]", "原样保留。" * 40]
SHORT = "# Example\n\nEnable parsing with:\n\n```text\nMODE_A()\n```\n\nRead result with:\n\n```text\nMODE_B()\n```\n"


@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize("body", LITERALS)
async def test_span_body_literals_do_not_declare_legacy_references(control, body):
    def final(data):
        return {"answer": body, "span_ids": list(data["inherited_span_bindings"]), "missing": []}
    model = SpanModel(overrides={"final": final})
    response = await UnifiedRetrievalPipeline(settings(control_protocol=control), None, model=model).ask_materials(
        "逐字返回材料中的字符串。", [material(text=body)])
    assert response.generation["status"] == "completed"
    assert response.answer == model.calls[-1]["raw"]
    assert json.loads(response.answer)["answer"] == body
    assert len(response.sources) == 1
    audit = response.generation["citation_audit"]
    assert audit["reference_protocol"] == "span_ids_v1"
    assert audit["inline_citation_interpreted"] is False
    assert audit["semantic_support_verified"] is False


@pytest.mark.parametrize("control", CONTROLS)
async def test_literal_merge_stays_eligible_without_rewriting_or_losing_span_ids(control):
    def merge(data):
        return {"answer": "`[资料 99]`", "span_ids": list(data["inherited_span_bindings"]), "missing": []}
    model = SpanModel(overrides={("merge", 1): merge})
    result = await UnifiedRetrievalPipeline(settings(control_protocol=control), None, model=model).ask_materials(
        "逐字返回示例。", [material(i, "[资料 99]") for i in range(4)])
    node = next(n for n in result.generation["nodes"] if n["kind"] == "merge")
    final = model.calls[-1]["data"]
    child = next(c for c in final["children"] if c["node_id"] == node["id"])
    assert node["status"] == "completed" and not child["raw_withheld"]
    assert child["raw_text"] == node["raw_text"]
    assert set(final["inherited_span_bindings"]) == {f"E{i}.S1" for i in range(1, 5)}
    assert result.answer == model.calls[-1]["raw"]


@pytest.mark.parametrize("bad", [
    {"answer": "[资料 99]", "span_ids": ["E99.S1"], "missing": []},
    {"answer": "[资料 99]", "span_ids": ["E1.S1", "E1.S1"], "missing": []},
    '```json\n{"answer":"[资料 99]","span_ids":["E1.S1"],"missing":[]}\n```',
])
async def test_literal_body_does_not_relax_json_or_span_reference_validation(bad):
    model = SpanModel(overrides={"final": bad})
    result = await UnifiedRetrievalPipeline(settings(), None, model=model).ask_materials(
        "目标", [material(text="[资料 99]")])
    assert result.generation["status"] == "invalid_contract"
    assert result.answer == model.calls[-1]["raw"]
    assert result.sources == [] and result.generation["span_bindings"] == {}


@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize("failure", ["task", "task_binding", "direction", "direction_permission", "plan"])
async def test_early_control_failure_uses_reserved_final_without_acquisition_or_failed_raw(control, failure):
    override = {failure: "REJECTED_CONTROL_SENTINEL"}
    if failure == "task_binding":
        override = {"task": {"goals": [{"id": "G1", "text": "目标", "refs": [
            {"message_id": "M1", "quote": "REJECTED_CONTROL_SENTINEL"}]}],
            "constraints": [], "retracted": [], "unresolved": []}}
    elif failure == "direction_permission":
        override = {"direction": {"choice": "web"}}
    raw = '  {"answer":"模型报告未能取得资料。","span_ids":[],"missing":["有效证据"]}\r\n'
    override["final"] = raw
    model = SpanModel(mode="retrieve", channels=["knowledge_base"], overrides=override)
    adapter = Adapters()
    result = await UnifiedRetrievalPipeline(settings(control_protocol=control, max_plan_revisions=0), None,
        model=model, adapters=adapter).ask(SearchRequest(question="目标", retrieval_mode="knowledge_base"))
    assert not adapter.calls and model.counts["final"] == 1
    assert result.answer == result.generation["raw_model_answer"] == raw
    assert result.generation["status"] == "no_evidence"
    stage = failure.split("_")[0]
    status = "routing_failed" if stage == "direction" else stage + "_failed"
    assert result.generation["execution_scope"]["control_failures"] == [status]
    final = model.calls[-1]
    assert "REJECTED_CONTROL_SENTINEL" not in json.dumps(final["data"])
    assert final["data"]["children"] == [] and final["data"]["inherited_span_bindings"] == {}
    assert final["options"]["structured_schema"]["properties"]["span_ids"]["maxItems"] == 0
    assert final["data"]["latest_user_message"]["content"] == "目标"
    for call in result.generation["model_calls"]:
        if isinstance(call.get("raw_text"), str):
            assert call["raw_text_sha256"] == digest(call["raw_text"])


@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize("limit", range(1, 7))
@pytest.mark.parametrize("route", ["provided", "retrieve"])
async def test_native_total_call_limit_reserves_an_actual_final_from_initial_controllers(control, limit, route):
    policy = settings(control_protocol=control)
    policy.native_request_max_calls = limit
    model = SpanModel(mode=route, channels=["knowledge_base"])
    adapter = Adapters()
    response = await UnifiedRetrievalPipeline(policy, None, model=model, adapters=adapter).ask(
        SearchRequest(question="目标"))
    assert len(model.calls) <= limit and model.counts["final"] == 1
    assert response.answer == model.calls[-1]["raw"]
    assert model.calls[-1]["purpose"] == "final"
    assert response.generation["request_budget"]["admitted_calls"] == len(model.calls)
    if limit <= 2:
        blocked = [c for c in response.generation["model_calls"] if c["status"] == "final_budget_reserved"]
        assert len(blocked) == 1 and blocked[0]["completion_attempted"] is False
        assert blocked[0]["purpose"] in {"task", "direction"}
        assert not adapter.calls
        assert response.generation["execution_scope"]["stop_reason"] == "control_call_budget_exhausted"
        if limit == 2 and control == "review_cycle_v1":
            direction = response.retrieval["directions"][-1]
            assert direction["status"] == "not_started" and not direction["completion_attempted"]
    if route == "retrieve" and limit < 6:
        assert not adapter.calls  # no provider quota without a usable read/review slot


@pytest.mark.parametrize("stage", ["task", "direction", "plan"])
async def test_fatal_control_transport_failure_never_spends_reserved_final_or_retries(stage):
    model = SpanModel(mode="retrieve", channels=["knowledge_base"], overrides={stage: ("transport_error", None)})
    adapter = Adapters()
    result = await UnifiedRetrievalPipeline(settings(control_protocol="review_cycle_v1"), None,
        model=model, adapters=adapter).ask(SearchRequest(question="目标"))
    assert result.answer == "" and result.generation["status"] == "model_failed"
    assert not adapter.calls and model.counts["final"] == model.counts["plan_revision"] == 0
    assert model.counts[stage] == 1


@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize("final_raw", ['broken final {', ('length', '{"answer":"截断')])
async def test_failed_control_and_failed_final_both_remain_immutable(control, final_raw):
    model = SpanModel(overrides={"task": ("length", "FAILED_TASK_SENTINEL"), "final": final_raw})
    result = await UnifiedRetrievalPipeline(settings(control_protocol=control, max_calls=2), None, model=model).ask(
        SearchRequest(question="目标"))
    assert model.counts["task"] == model.counts["final"] == 1
    assert result.answer == model.calls[-1]["raw"]
    assert result.generation["status"] == ("length" if isinstance(final_raw, tuple) else "invalid_contract")
    assert "FAILED_TASK_SENTINEL" not in json.dumps(model.calls[-1]["data"])


@pytest.mark.parametrize("control", CONTROLS)
async def test_clarification_remains_a_single_terminal_call_not_followed_by_final(control):
    model = SpanModel(mode="clarify")
    result = await UnifiedRetrievalPipeline(settings(control_protocol=control, max_calls=3), None, model=model).ask(
        SearchRequest(question="目标"))
    assert model.counts["clarify"] == 1 and model.counts["final"] == 0
    assert result.generation["status"] == "needs_clarification"
    assert result.answer == model.calls[-1]["raw"]


async def test_search_and_v2_do_not_gain_a_new_final_or_failed_control_fallback():
    model = SpanModel(overrides={"task": "BAD_TASK"})
    result = await UnifiedRetrievalPipeline(settings(), None, model=model).search(SearchRequest(question="目标"))
    assert result.retrieval["status"] == "task_failed" and model.counts["final"] == 0
    legacy = Model(overrides={"task": "BAD_TASK"})
    result = await UnifiedRetrievalPipeline(v2_settings(), None, model=legacy).ask(SearchRequest(question="目标"))
    assert result.generation["status"] == "task_failed" and result.answer == ""
    assert legacy.counts["final"] == 0


@pytest.mark.parametrize("control", CONTROLS)
async def test_short_structured_source_is_one_read_with_exact_full_body_and_reserved_final(control):
    model = SpanModel()
    response = await UnifiedRetrievalPipeline(settings(control_protocol=control, max_calls=2), None, model=model).ask_materials(
        "收集此材料中的调用。", [material(text=SHORT)])
    assert [c["purpose"] for c in model.calls] == ["leaf", "final"]
    original = model.calls[0]["data"]["original_material"]
    covered = set()
    for span in original["body_spans"]:
        start, end = span["span"]["start"], span["span"]["end"]
        assert SHORT[start:end] == span["text"]
        covered.update(range(start, end))
    assert covered == set(range(len(SHORT)))
    assert response.generation["execution_scope"]["unread_units"] == 0
    assert response.answer == model.calls[-1]["raw"]
    assert model.calls[-1]["options"]["evidence_ids"] == ()
    assert "body_spans" not in json.dumps(model.calls[-1]["data"])


@pytest.mark.parametrize("control", CONTROLS)
async def test_packing_never_joins_distinct_sources_and_stop_reason_keeps_budget_gap(control):
    model = SpanModel()
    response = await UnifiedRetrievalPipeline(settings(control_protocol=control, max_calls=2), None, model=model).ask_materials(
        "目标", [material(i, SHORT) for i in range(3)])
    units = response.retrieval["evidence_catalog"]["units"]
    assert len(units) == 3
    assert all(u["snippet"] == SHORT for u in units.values())
    scope = response.generation["execution_scope"]
    assert scope["unread_units"] == 2 and scope["read_units"] == 1
    assert scope["stop_reason"] == "read_budget_exhausted"
    assert scope["budget_events"][-1]["not_admitted_unit_ids"] == ["E2", "E3"]
    assert response.generation["status"] == "partial" and model.counts["final"] == 1
