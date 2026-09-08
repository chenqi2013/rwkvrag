import json

import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.lexical_index import LexicalResult
from llamaindex_retrieval.native_rwkv import NativeRWKVResult
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline, parse_task_selections
from llamaindex_retrieval.schemas import SearchRequest


@pytest.mark.parametrize("text, expected", [
    ("NONE", []), (" E1 \n", [1]), ("E2,E1,E2", [2, 1]),
    ("E2，E1", [2, 1]), ("E1;\nE2", [1, 2]),
])
def test_task_selection_only_interprets_actual_ids(text, expected):
    assert parse_task_selections(text, 2) == expected


@pytest.mark.parametrize("text", [
    "", "E0", "E3", "E01", "E1,", "E1,NONE", "NONE,E1", "f1: E1",
    "E1 because it is relevant", "25000", "E1E2", "E1\nE2", "E1: answer",
    "[E1]", "E1 E2", "E１", "NONE explanation",
])
def test_bad_task_selection_cannot_leak_valid_prefix_to_writer(text):
    with pytest.raises(ValueError):
        parse_task_selections(text, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ["E1", "NONE", "E1,E2"])
async def test_task_selection_binds_source_and_never_claims_field_coverage(selection):
    raw_answer = ">原样思考</think>\n原样回答，不由代码补充。"
    calls = []

    class Index:
        def search_chunks(self, query, **kwargs):
            assert query == "甲对象的完整问题"
            return [LexicalResult(node_id="source", document_id="doc", text="原文中的事实。",
                metadata={"title": "甲", "source": "fixture"}, score=1)]

    class Model:
        async def complete(self, messages, *, stage, evidence_ids, assistant_prefill, **kwargs):
            prompt = messages[0]["content"]
            calls.append({"stage": stage, "prompt": prompt, "prefill": assistant_prefill})
            body = '["甲对象的完整问题"]' if stage == "planner" else selection
            raw = ">" + body if stage != "writer" else raw_answer
            return NativeRWKVResult(status="completed", raw_text=raw, finish_reason="stop",
                trace={"stage": stage, "status": "completed", "raw_text": raw,
                       "prefill": assistant_prefill})

    settings = Settings(native_plan_protocol="shared_tasks", native_planner_prefill="<think></think",
        native_resolver_protocol="task_units", native_resolver_prefill="<think></think")
    response = await RWKVPipeline(settings, Index(), Model()).ask(SearchRequest(question="当前问题"))
    assert response.answer == raw_answer
    reader, writer = calls[1:]
    assert reader["prefill"] == "<think></think" and writer["prefill"] == "<think"
    assert json.loads(reader["prompt"].rsplit("原文：", 1)[1]) == {"E1": "原文中的事实。"}
    assert "E2" not in reader["prompt"]
    assert response.retrieval["uncovered_fields"] is None
    assert response.retrieval["field_coverage_assessed"] is False
    visible = json.loads(writer["prompt"].split("逐字证据：", 1)[1])
    if selection == "E1":
        assert [item["snippet"] for item in [source.model_dump() for source in response.sources]] == ["原文中的事实。"]
        assert response.sources[0].metadata["field_ids"] == []
        assert response.sources[0].metadata["selection_scope"] == "task_units"
        assert [item["text"] for item in visible] == ["原文中的事实。"]
        assert response.generation["model_calls"][1]["selected_units"] == ["E1"]
    else:
        assert response.sources == [] and visible == []
    assert response.generation["status"] == (
        "resolver_partial_failure" if selection == "E1,E2" else "completed")
