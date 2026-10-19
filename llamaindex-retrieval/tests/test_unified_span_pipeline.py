"""Application wiring/provenance tests with synthetic models, not quality scores."""
import asyncio
import json

import pytest

from llamaindex_retrieval.schemas import SearchRequest
from llamaindex_retrieval.unified_contracts import reduction_calls
from llamaindex_retrieval.unified_retrieval import UnifiedRetrievalPipeline
from llamaindex_retrieval.unified_span_contract import bound_span_schema
from llamaindex_retrieval.unified_span_reading import SpanAnswer, SpanReading
from test_unified_retrieval import Adapters, Model, material, settings as old_settings


def settings(**limits):
    return old_settings(reading_protocol="span_ids_v1", **limits)


class SpanModel(Model):
    async def complete(self, messages, **options):
        purpose = options["stage"].removeprefix("unified_")
        data = json.loads(messages[0]["content"].split("\n输入数据：\n", 1)[1])
        key = (purpose, self.counts[purpose] + 1)
        if key not in self.overrides and purpose not in self.overrides:
            if purpose == "leaf":
                self.overrides[key] = {"findings": [{
                    "span_id": data["original_material"]["body_spans"][0]["id"],
                    "statement": "fixture statement"}], "missing": []}
            elif purpose in {"merge", "final"}:
                self.overrides[key] = {"answer": "fixture answer", "span_ids": list(
                    data["inherited_span_bindings"]), "missing": []}
        return await super().complete(messages, **options)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 10, 27])
async def test_v3_reaches_only_root_with_bounded_unchanged_children_and_source_closure(n):
    model = SpanModel()
    response = await UnifiedRetrievalPipeline(settings(max_units=40), None, model=model).ask_materials(
        "目标", [material(i) for i in range(n)])
    nodes = {x["id"]: x for x in response.generation["nodes"]}
    reductions = max(1, reduction_calls(n))
    assert model.counts["leaf"] == n
    assert model.counts["merge"] + model.counts["final"] == reductions
    assert model.counts["final"] == 1
    assert model.counts["task"] == 0
    assert len(nodes) == n + reductions
    for call in model.calls:
        assert "structured_schema" in call["options"]
        if call["purpose"] not in {"merge", "final"}:
            continue
        children = call["data"]["children"]
        assert (1 if n else 0) <= len(children) <= 3
        assert call["options"]["evidence_ids"] == ()
        refs = call["data"]["inherited_span_bindings"]
        assert set(refs) == {s for child in children for s in child["model_selected_span_ids"]}
        for child in children:
            assert child["raw_text"] == nodes[child["node_id"]]["raw_text"]
        for binding in refs.values():
            assert not binding["original_text_in_this_prompt"]
            assert "selected_original_text" not in binding and "context_spans" not in binding
            leaf = nodes[binding["leaf_node_id"]]
            assert binding["leaf_call_id"] == leaf["call_id"]
            assert binding["leaf_raw_sha256"] == leaf["raw_text_sha256"]
        assert call["options"]["structured_schema"] == bound_span_schema(SpanAnswer, refs)
    assert response.generation["protocol"] == "rwkv-unified-v3"
    if n:
        root = nodes[response.generation["root_node_id"]]
        assert root["kind"] == "final"
        assert response.answer == root["raw_text"] == response.generation["raw_model_answer"]
        assert response.generation["status"] == "completed"
        assert len(response.sources) == n
        assert len(response.generation["span_bindings"]) == n
        assert not response.generation["semantic_support_verified"]
    else:
        assert response.answer == model.calls[-1]["raw"]
        assert response.generation["status"] == "no_evidence"
        assert response.generation["span_bindings"] == {}


@pytest.mark.parametrize("failure", [("length", "FAILED_LEAF_SENTINEL"), "FAILED_LEAF_SENTINEL",
    {"findings": [{"span_id": "E999.S1", "statement": "FAILED_LEAF_SENTINEL"}], "missing": []}])
async def test_failed_raw_and_unadmitted_selection_are_withheld_from_merge_and_review(failure):
    model = SpanModel(mode="retrieve", channels=["knowledge_base"], overrides={("leaf", 2): failure})
    response = await UnifiedRetrievalPipeline(settings(), None, model=model,
        adapters=Adapters([[material(i) for i in range(4)]])).ask(SearchRequest(question="目标"))
    failed = response.generation["nodes"][1]
    assert "FAILED_LEAF_SENTINEL" in failed["raw_text"]
    assert failed["evidence_ids"] == [] and failed["span_ids"] == []
    if isinstance(failure, dict):
        assert failed["parsed_model_span_ids"] == ["E999.S1"]
    else:
        assert failed["parsed_model_span_ids"] is None
    for call in model.calls:
        if call["purpose"] in {"review", "merge", "final"}:
            assert "FAILED_LEAF_SENTINEL" not in json.dumps(call["data"])
            assert "E2.S1" not in call["data"]["inherited_span_bindings"]
    assert response.generation["status"] == "partial"
    assert "E2.S1" not in response.generation["span_bindings"]


async def test_intermediate_cannot_import_a_sibling_groups_selected_span():
    raw = '{"answer":"CROSS_GROUP_SENTINEL","span_ids":["E4.S1"],"missing":[]}'
    model = SpanModel(overrides={("merge", 1): raw})
    response = await UnifiedRetrievalPipeline(settings(), None, model=model).ask_materials(
        "目标", [material(i) for i in range(4)])
    bad = response.generation["nodes"][4]
    assert bad["status"] == "invalid_contract" and bad["raw_text"] == raw
    assert bad["parsed_model_span_ids"] == ["E4.S1"] and bad["span_ids"] == []
    final = next(c for c in model.calls if c["purpose"] == "final")
    assert "CROSS_GROUP_SENTINEL" not in json.dumps(final["data"])
    assert list(final["data"]["inherited_span_bindings"]) == ["E4.S1"]
    assert response.generation["status"] == "partial"


async def test_model_dropped_references_do_not_reappear_at_root():
    model = SpanModel(overrides={("merge", 1): {
        "answer": "chosen subset", "span_ids": ["E2.S1"], "missing": ["gap"]}})
    response = await UnifiedRetrievalPipeline(settings(), None, model=model).ask_materials(
        "目标", [material(i) for i in range(4)])
    final = next(c for c in model.calls if c["purpose"] == "final")
    assert set(final["data"]["inherited_span_bindings"]) == {"E2.S1", "E4.S1"}
    assert set(response.generation["span_bindings"]) == {"E2.S1", "E4.S1"}
    # The child's gap is in its unchanged raw; the program doesn't fill/erase it.
    assert "gap" in final["data"]["children"][0]["raw_text"]


async def test_single_leaf_final_call_is_reserved_and_no_intermediate_is_used_on_timeout():
    result = await UnifiedRetrievalPipeline(settings(max_calls=2), None, model=SpanModel()).ask_materials(
        "目标", [material(i) for i in range(4)])
    assert result.generation["request_budget"]["admitted_calls"] == 2
    assert len(result.generation["nodes"]) == 2
    assert result.generation["execution_scope"]["unread_units"] == 3
    assert result.generation["status"] == "partial"
    async def slow(_):
        await asyncio.sleep(10)
    model = SpanModel(overrides={"final": slow})
    result = await UnifiedRetrievalPipeline(settings(timeout_seconds=0.03), None, model=model).ask_materials(
        "目标", [material()])
    assert result.generation["nodes"][0]["raw_text"]
    assert result.answer == "" and result.generation["status"] == "request_timeout"
    assert result.generation["nodes"][-1]["status"] == "cancelled"


@pytest.mark.parametrize("n", [1, 4])
async def test_all_failed_leaves_produce_no_fake_root_answer(n):
    raw = '{"answer":"本次阅读失败，无法确认所问事实。","span_ids":[],"missing":["阅读结果"]}'
    model = SpanModel(overrides={"leaf": "NONE", "final": raw})
    result = await UnifiedRetrievalPipeline(settings(), None, model=model).ask_materials(
        "目标", [material(i) for i in range(n)])
    assert result.answer == raw and result.generation["status"] == "no_evidence"
    assert model.counts["leaf"] == n  # no repair or automatic rereading
    assert model.counts["merge"] == 0 and model.counts["final"] == 1
    final = model.calls[-1]
    assert final["data"]["inherited_span_bindings"] == {}
    assert all(c["raw_withheld"] and c["raw_text"] is None for c in final["data"]["children"])
    assert "NONE" not in json.dumps(final["data"])
    assert sum(node["raw_text"] == "NONE" for node in result.generation["nodes"]) == n


@pytest.mark.parametrize("method", ["ask", "search"])
async def test_shared_core_model_driven_followup_with_kb_scope_and_permissions(method):
    model = SpanModel(mode="retrieve", channels=["knowledge_base", "web"], overrides={
        ("review", 1): {"next": "search", "missing": ["model gap"], "reason": "model reason"}})
    adapter = Adapters([[material(0)], [material(1)]])
    result = await getattr(UnifiedRetrievalPipeline(settings(), None, model=model, adapters=adapter), method)(
        SearchRequest(question="对象", retrieval_mode="auto", knowledge_base_id="scope"))
    assert len(adapter.calls) == 2
    assert all(request.knowledge_base_id == "scope" for _, request in adapter.calls)
    assert [action.query for action, _ in adapter.calls] == ["query1", "query2"]
    assert model.counts["review"] == 2
    assert model.counts["final"] == (1 if method == "ask" else 0)
    assert result.retrieval["routing"]["model_decision"] is True
    for call in model.calls:
        assert call["data"]["latest_user_message"]["content"] == "对象"
        assert "UNSELECTED_PROVIDER_SNAPSHOT_SENTINEL" not in json.dumps(call["data"])


@pytest.mark.parametrize("failure_stage", ["review", "plan"])
async def test_controller_failure_preserved_while_reserved_root_can_report_partial(failure_stage):
    overrides = {("review", 1): {"next": "search", "missing": ["gap"], "reason": "check"},
                 (failure_stage, 2 if failure_stage == "plan" else 1): "BAD_CONTROLLER"}
    model = SpanModel(mode="retrieve", channels=["knowledge_base"], overrides=overrides)
    result = await UnifiedRetrievalPipeline(settings(), None, model=model, adapters=Adapters()).ask(
        SearchRequest(question="目标"))
    assert result.generation["status"] == "partial"
    assert result.generation["execution_scope"]["control_failures"] == [failure_stage + "_failed"]
    assert result.answer == model.calls[-1]["raw"]
    assert model.calls[-1]["purpose"] == "final"
    assert "BAD_CONTROLLER" not in json.dumps(model.calls[-1]["data"])


async def test_qa_atom_preserved_and_span_receipt_is_not_a_generated_quote():
    source = material(text="# 问题\n\n" + "内容。" * 250 + "\n# 答案\n\n答案。")
    source.metadata["kind"] = "qa-markdown"
    model = SpanModel()
    result = await UnifiedRetrievalPipeline(settings(), None, model=model).ask_materials("目标", [source])
    leaf = next(c for c in model.calls if c["purpose"] == "leaf")
    assert len(leaf["data"]["original_material"]["body_spans"]) == 1
    assert leaf["options"]["structured_schema"] == bound_span_schema(SpanReading, ["E1.S1"])
    binding = result.generation["span_bindings"]["E1.S1"]
    assert binding["selected_original_text"] == source.snippet
    assert binding["model_generated_quote"] is False


async def test_empty_findings_are_not_filled_and_final_missing_is_model_authored():
    model = SpanModel(overrides={"leaf": {"findings": [], "missing": ["not found"]},
        "final": {"answer": "无法依据本次材料回答", "span_ids": [], "missing": ["model gap"]}})
    result = await UnifiedRetrievalPipeline(settings(), None, model=model).ask_materials("目标", [material()])
    assert result.sources == [] and result.generation["span_bindings"] == {}
    assert json.loads(result.answer)["missing"] == ["model gap"]
    assert result.generation["status"] == "no_evidence"


async def test_unknown_root_citation_is_not_repaired():
    raw = '{"answer":"wrong [资料 99]","span_ids":["E1.S1"],"missing":[]}'
    model = SpanModel(overrides={"final": raw})
    result = await UnifiedRetrievalPipeline(settings(), None, model=model).ask_materials("目标", [material()])
    # Legacy-looking body text is not the span reference namespace. It is not
    # rewritten, linked, or certified as factual just because IDs bind.
    assert result.answer == raw and result.generation["status"] == "completed"
    assert list(result.generation["span_bindings"]) == ["E1.S1"]
    assert result.generation["semantic_support_verified"] is False


async def test_request_local_span_catalogs_never_cross_requests():
    model = SpanModel()
    engine = UnifiedRetrievalPipeline(settings(), None, model=model)
    a, b = await asyncio.gather(engine.ask_materials("甲", [material(1)]),
                                engine.ask_materials("乙", [material(2)]))
    one, two = [r.generation["span_bindings"]["E1.S1"] for r in (a, b)]
    assert one["source_id"] != two["source_id"] and one["snapshot_id"] != two["snapshot_id"]


async def test_kb_default_cannot_be_expanded_to_web_by_direction():
    model = SpanModel(mode="retrieve", channels=["web"])
    adapter = Adapters()
    result = await UnifiedRetrievalPipeline(settings(), None, model=model, adapters=adapter).ask(
        SearchRequest(question="目标"))
    assert result.generation["status"] == "no_evidence" and not adapter.calls
    assert result.generation["execution_scope"]["control_failures"] == ["routing_failed"]
    assert model.counts["final"] == 1 and result.answer == model.calls[-1]["raw"]
