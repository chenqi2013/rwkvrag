"""A lossy model plan must not remove the user's second comparison object."""
from unittest.mock import AsyncMock

import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline
from llamaindex_retrieval.schemas import SearchRequest
from test_rwkv_pipeline import FakeIndex, FakeModel, hit, native_result


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["global", "per_query"])
async def test_original_comparison_reaches_retrieval_and_reader_despite_lossy_plan(scope):
    question = "比较甲和乙的容量与离线能力"
    model = FakeModel(plan={"queries": ["甲容量"], "fields": ["容量"]},
                      writer_raw="模型原始答案[资料 99]")
    index = FakeIndex({question: [hit("b", "乙容量12L，不支持离线")],
                       "甲容量": [hit("a", "甲容量8L，支持离线")]})
    pipeline = RWKVPipeline(Settings(native_resolver_budget_scope=scope,
        native_preserve_original_question=True,
        native_task_source="queries", native_resolver_protocol="binary_query",
        native_resolver_task_grouping="individual"), index, model)
    pipeline._resolve = AsyncMock(return_value=([], []))
    response = await pipeline.ask(SearchRequest(question=question,
        retrieval_mode="knowledge_base", knowledge_base_id="private-kb"))
    assert response.retrieval["plan"]["queries"] == ["甲容量"]
    assert response.retrieval["retrieval_queries"] == [question, "甲容量"]
    assert response.retrieval["original_question_preserved"] is True
    assert all(call[2] == "private-kb" for call in index.calls)
    args = pipeline._resolve.call_args
    assert args.args[1] == [question, "甲容量"]
    assert {source.id for source in args.args[2]} == {"a", "b"}
    if scope == "per_query":
        assert args.kwargs["source_tasks"]["b"] == [[question]]
    # Evidence is still Reader-owned, and invalid Writer citations stay raw.
    assert response.sources == []
    assert response.answer == "模型原始答案[资料 99]"


@pytest.mark.asyncio
async def test_exact_original_query_is_not_run_twice():
    index = FakeIndex({"q": []})
    pipeline = RWKVPipeline(Settings(native_preserve_original_question=True), index,
        FakeModel(plan={"queries": ["q"], "fields": ["q"]}))
    await pipeline.ask(SearchRequest(question="q"))
    assert len(index.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("valid", [True, False])
async def test_planner_repair_is_one_new_model_call_and_preserves_both_raw_traces(valid):
    import json
    invalid = '[{"queries":["a"],"fields":["f"]}]'
    corrected = json.dumps({"queries": ["a", "b"], "fields": ["f"]}) if valid else invalid

    class Model(FakeModel):
        planner_count = 0

        async def complete(self, messages, **kwargs):
            if kwargs["stage"] == "planner":
                self.planner_count += 1
                if self.planner_count == 2:
                    assert "上一次规划未通过格式校验" in messages[0]["content"]
                    assert "latest_question" in messages[0]["content"]
                return native_result("planner", ">思考</think>" + (
                    invalid if self.planner_count == 1 else corrected))
            return await super().complete(messages, **kwargs)

    model = Model(writer_raw="原始回答")
    pipeline = RWKVPipeline(Settings(native_planner_format_repair=True),
        FakeIndex({"a": [], "b": [], "q": []}), model)
    result = await pipeline.ask(SearchRequest(question="q"))
    traces = result.generation["model_calls"]
    assert model.planner_count == 2
    assert traces[0]["raw_text"].endswith(invalid)
    assert traces[1]["raw_text"].endswith(corrected)
    assert traces[0]["format_repair_succeeded"] is valid
    assert traces[1]["purpose"] == "planner_format_repair"
    assert result.retrieval["plan"]["queries"] == (["a", "b"] if valid else ["q"])
    assert result.generation["status"] == ("completed" if valid else "planner_partial_failure")
    assert result.generation["stage_status"]["planner"] == ("completed" if valid else "failed")
    assert result.answer == "原始回答"


@pytest.mark.asyncio
async def test_incomplete_planner_generation_is_not_retried_as_a_format_error():
    model = FakeModel(planner_result=native_result("planner", None, "timeout"))
    pipeline = RWKVPipeline(Settings(native_planner_format_repair=True),
        FakeIndex({"q": []}), model)
    result = await pipeline.ask(SearchRequest(question="q"))
    assert sum(call["stage"] == "planner" for call in model.calls) == 1
    assert result.generation["status"] == "planner_partial_failure"
    assert result.retrieval["plan"]["fallback"] == "original_question"
