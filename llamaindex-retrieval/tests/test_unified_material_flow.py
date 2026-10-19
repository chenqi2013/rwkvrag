"""Flow/immutability tests with fixtures, not native answer-quality evidence."""
import json

import pytest

from llamaindex_retrieval.schemas import ConversationMessage
from llamaindex_retrieval.unified_contracts import digest, dumps
from llamaindex_retrieval.unified_retrieval import UnifiedRetrievalPipeline
from llamaindex_retrieval.unified_span_contract import bound_span_schema
from llamaindex_retrieval.unified_span_reading import SpanAnswer, SpanReading
from test_unified_retrieval import Adapters, material
from test_unified_span_pipeline import SpanModel, settings


@pytest.mark.parametrize("control", ["single_direction_v1", "review_cycle_v1"])
async def test_closed_materials_need_no_derived_task_or_conversation_quotes(control):
    def forbidden(_):
        raise AssertionError("closed reading must not call a controller")
    model = SpanModel(overrides={s: forbidden for s in ("task", "direction", "plan", "review")})
    adapters = Adapters()
    history = [ConversationMessage(role="user", content="旧问题\r\n"),
               ConversationMessage(role="assistant", content="先前助手的内容，不是原文。")]
    question = "{容器}的当前问题？"
    result = await UnifiedRetrievalPipeline(settings(control_protocol=control), None,
        model=model, adapters=adapters).ask_materials(question, [material()], history=history)
    assert [call["purpose"] for call in model.calls] == ["leaf", "final"]
    for call in model.calls:
        data = call["data"]
        assert data["latest_user_message"] == {"id": "M3", "role": "user", "content": question}
        assert data["history"] == [{"id": f"M{i+1}", **m.model_dump()} for i, m in enumerate(history)]
    assert result.retrieval["task"] is None
    assert result.retrieval["task_execution"] == "not_required_closed_materials"
    assert not adapters.calls
    assert result.answer == model.calls[-1]["raw"]


FAILURES = [
    ("length", '{"findings":[{"span_id":"E1.S1","statement":"FAILED_RAW_SENTINEL'),
    "```json\r\nFAILED_RAW_SENTINEL\n```",
    {"findings": [{"span_id": "E1.S1", "statement": "first choice"},
                  {"span_id": "E1.S999", "statement": "FAILED_RAW_SENTINEL"}], "missing": []},
]


@pytest.mark.parametrize("failure", FAILURES)
@pytest.mark.parametrize("control", ["single_direction_v1", "review_cycle_v1"])
async def test_failed_original_raw_does_not_block_final_or_create_selections(failure, control):
    model = SpanModel(overrides={("leaf", 1): failure})
    result = await UnifiedRetrievalPipeline(settings(control_protocol=control), None,
        model=model).ask_materials("目标", [material(text="同一份不可改写的原文。")])
    first, final = model.calls
    failed, root = result.generation["nodes"]
    assert [c["purpose"] for c in model.calls] == ["leaf", "final"]
    assert failed["raw_text"] == first["raw"]
    assert failed["raw_text_sha256"] == digest(first["raw"])
    assert failed["span_ids"] == []  # not even the valid first finding is salvaged
    assert failed["status"] in {"length", "invalid_contract"}
    assert failed["parsed_model_span_ids"] == (["E1.S1", "E1.S999"] if isinstance(failure, dict) else None)
    assert "FAILED_RAW_SENTINEL" not in dumps(final["data"])
    assert final["data"]["inherited_span_bindings"] == {}
    child = final["data"]["children"][0]
    assert child["raw_withheld"] and child["raw_text"] is None
    assert child["call_id"] == failed["call_id"]
    assert child["raw_text_sha256"] == failed["raw_text_sha256"]
    assert root["parents"] == [failed["id"]]
    assert root["original_text_input_ids"] == []
    assert result.answer == root["raw_text"] == final["raw"]
    assert result.generation["execution_scope"]["failed_nodes"] == [failed["id"]]
    assert result.generation["span_bindings"] == {} and result.sources == []
    assert result.generation["status"] == "no_evidence"
    assert len({n["call_id"] for n in result.generation["nodes"]}) == 2


@pytest.mark.parametrize("limit", [2, 3])
async def test_failed_reading_reserves_final_without_automatic_rereading(limit):
    raw = '  {"answer":"模型说明本次阅读缺口。","span_ids":[],"missing":["有效阅读"]}\r\n'
    model = SpanModel(overrides={"leaf": FAILURES[0], "final": raw})
    result = await UnifiedRetrievalPipeline(settings(max_calls=limit), None, model=model).ask_materials(
        "目标", [material()])
    assert model.counts["leaf"] == 1 and model.counts["final"] == 1
    assert len(model.calls) == 2 <= limit
    assert result.answer == result.generation["raw_model_answer"] == raw
    assert result.sources == [] and result.generation["span_bindings"] == {}
    assert result.generation["status"] == "no_evidence"
    final = model.calls[-1]
    assert final["data"]["inherited_span_bindings"] == {}
    assert final["options"]["structured_schema"]["properties"]["span_ids"]["maxItems"] == 0
    assert "FAILED_RAW_SENTINEL" not in dumps(final["data"])


async def test_all_failed_merges_still_reach_real_final_without_leaking_failed_raw():
    model = SpanModel(overrides={"merge": "FAILED_MERGE_SENTINEL"})
    result = await UnifiedRetrievalPipeline(settings(), None, model=model).ask_materials(
        "目标", [material(i) for i in range(4)])
    assert model.counts["merge"] == 2 and model.counts["final"] == 1
    assert "FAILED_MERGE_SENTINEL" not in dumps(model.calls[-1]["data"])
    assert model.calls[-1]["data"]["inherited_span_bindings"] == {}
    assert result.answer == model.calls[-1]["raw"]
    assert result.generation["span_bindings"] == {}


async def test_transport_failure_is_not_retried_or_disguised_as_a_model_answer():
    def fail(_):
        raise ConnectionError("PRIVATE_ERROR_SENTINEL")
    model = SpanModel(overrides={"leaf": fail})
    result = await UnifiedRetrievalPipeline(settings(), None, model=model).ask_materials("目标", [material()])
    assert len(model.calls) == 1 and model.counts["final"] == 0
    assert result.answer == "" and result.generation["status"] == "model_failed"
    assert "PRIVATE_ERROR_SENTINEL" not in dumps(result.model_dump())
    assert "reading_attempts" not in result.generation["execution_scope"]


@pytest.mark.parametrize("contract", [SpanReading, SpanAnswer])
@pytest.mark.parametrize("ids", [[], ["E9.S2"], ["E2.S1", "E9.S2"]])
def test_schema_choices_are_call_local_and_do_not_change_free_text_or_global_contract(contract, ids):
    before = contract.model_json_schema()
    schema = bound_span_schema(contract, ids)
    assert contract.model_json_schema() == before
    if contract is SpanReading:
        field = schema["$defs"]["SpanFinding"]["properties"]["span_id"]
        choices = schema["properties"]["findings"]
        assert schema["$defs"]["SpanFinding"]["properties"]["statement"] == before["$defs"]["SpanFinding"]["properties"]["statement"]
    else:
        choices = schema["properties"]["span_ids"]
        field = choices["items"]
        assert schema["properties"]["answer"] == before["properties"]["answer"]
    if ids:
        assert field["enum"] == sorted(ids)
        assert "E1.S1" not in field["enum"]
    else:
        assert choices["maxItems"] == 0
    assert schema["properties"]["missing"] == before["properties"]["missing"]


async def test_generation_schema_hash_and_reducer_prompt_match_actual_namespace():
    model = SpanModel()
    result = await UnifiedRetrievalPipeline(settings(), None, model=model).ask_materials(
        "目标", [material(1), material(2)])
    for call, node in zip(model.calls, result.generation["nodes"], strict=True):
        schema = call["options"]["structured_schema"]
        assert node["span_schema_sha256"] == digest(dumps(schema))
        assert call["options"]["trace"]["span_schema_sha256"] == node["span_schema_sha256"]
        if call["purpose"] == "final":
            text = call["options"]["trace"]["messages"][0]["content"]
            printed = text.split("\n输出结构：", 1)[1].split("\n输入数据：", 1)[0]
            assert json.loads(printed) == schema
