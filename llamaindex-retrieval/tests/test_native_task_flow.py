"""Choose one model-authored list for later stages without changing its text."""

import json

import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.lexical_index import LexicalResult
from llamaindex_retrieval.native_rwkv import NativeRWKVResult
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline
from llamaindex_retrieval.schemas import ConversationMessage, SearchRequest


@pytest.mark.asyncio
@pytest.mark.parametrize("task_source", ["fields", "queries"])
async def test_fixed_task_source_drives_reader_and_writer_preserving_plan_and_history(task_source):
    plan = {"queries": ["甲对象的当前版本支持什么接口？"], "fields": ["旧对象的价格"]}
    original_plan = json.dumps(plan, ensure_ascii=False)
    prompts = {}

    class Index:
        def search_chunks(self, query, **kwargs):
            assert query == plan["queries"][0]
            return [LexicalResult(node_id="id", document_id="doc", text="甲对象支持接口甲。",
                metadata={"title": "甲对象", "source": "fixture"}, score=1)]

    class Model:
        async def complete(self, messages, *, stage, assistant_prefill, **kwargs):
            prompts[stage] = messages[0]["content"]
            body = original_plan if stage == "planner" else "E1" if stage == "resolver" else "原样回答。"
            raw = ">思考</think>\n" + body
            return NativeRWKVResult(status="completed", raw_text=raw, finish_reason="stop",
                trace={"stage": stage, "status": "completed", "prefill": assistant_prefill,
                       "raw_text": raw})

    request = SearchRequest(question="按更正后的范围回答。", history=[
        ConversationMessage(role="user", content="之前想问旧对象价格，现撤回；只看甲对象当前接口。")])
    response = await RWKVPipeline(Settings(native_task_source=task_source,
        native_resolver_protocol="task_units"), Index(), Model()).ask(request)
    assert response.retrieval["plan"] == plan
    assert response.retrieval["active_task_source"] == task_source
    assert response.retrieval["active_tasks"] == plan[task_source]
    reader_tasks = json.loads(prompts["resolver"].split("\n子问题：", 1)[1].split("\n来源：", 1)[0])
    writer_tasks = json.loads(prompts["writer"].split("\n字段：", 1)[1].split("\n逐字证据：", 1)[0])
    assert reader_tasks == writer_tasks == plan[task_source]
    expected = {"history": [row.model_dump() for row in request.history],
                "latest_question": request.question}
    for prompt in prompts.values():
        assert json.JSONDecoder().raw_decode(prompt.split("任务：", 1)[1])[0] == expected
    assert response.generation["model_calls"][0]["raw_text"] == ">思考</think>\n" + original_plan
    assert response.answer == ">思考</think>\n原样回答。"
    assert response.retrieval["uncovered_fields"] is None
