"""Fact lookup must not silently replace the user's selection criteria."""
import asyncio

import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline, parse_plan, planner_prompt
from llamaindex_retrieval.schemas import SearchRequest
from test_rwkv_pipeline import FakeIndex, FakeModel, hit


def settings(**kwargs):
    return Settings(_env_file=None, native_plan_protocol="fact_queries_v1", **kwargs)


@pytest.mark.parametrize("wire", [
    '["人数"]', '{"queries":[],"fields":[]}', '{"queries":[]}',
    '{"queries":["人数"],"answer":"五人"}',
    '{"queries":["a"],"queries":["b"]}', '{"queries":[null]}',
])
def test_factual_plan_rejects_invalid_shapes_without_repair(wire):
    with pytest.raises((ValueError, TypeError)):
        parse_plan(wire, settings())


def test_query_budget_is_validated_without_silently_truncating_objects():
    with pytest.raises(ValueError, match="query count"):
        parse_plan('{"queries":["甲人数","乙人数"]}', settings(native_max_queries=1))
    plan = parse_plan('{"queries":["甲人数","乙人数"]}', settings(native_max_queries=2))
    assert plan == {"queries": ["甲人数", "乙人数"], "fields": ["甲人数", "乙人数"]}


def test_final_writer_keeps_original_requirements_while_reader_looks_up_facts():
    question = "比较甲乙，必须至少五人协作，优先低内存。"
    queries = ["甲最多支持几人协作？", "乙最多支持几人协作？"]
    index = FakeIndex({q: [hit(str(i), text)] for i, (q, text) in enumerate(zip(
        queries, ["甲最多支持三人协作。", "乙最多支持八人协作。"], strict=True))})
    model = FakeModel(plan={"queries": queries}, resolver_outputs={
        "0": "f1: E1\nf2: NONE", "1": "f1: NONE\nf2: E1"},
        writer_raw=">done</think>原始回答[资料 1]")
    pipe = RWKVPipeline(settings(), index, model=model)
    result = asyncio.run(pipe.ask(SearchRequest(question=question)))
    assert [q for q, *_ in index.calls] == queries
    assert result.retrieval["active_tasks"] == queries
    assert result.generation["plan_protocol"] == "fact_queries_v1"
    writer = next(c for c in model.calls if c["stage"] == "writer")
    assert question in writer["prompt"]
    assert result.generation["model_calls"][-1]["raw_text"] == model.last_writer_raw


def test_protocol_is_opt_in_and_prompt_keeps_complete_user_task():
    assert Settings(_env_file=None).native_plan_protocol == "queries_fields"
    task = '{"history":[],"latest_question":"要求离线且至少五人"}'
    assert task in planner_prompt(task, settings())


@pytest.mark.parametrize("override", [{"native_task_matrix_enabled": True},
                                     {"native_transport": "rwkvos_batch"}])
def test_incompatible_pipeline_does_not_silently_ignore_fact_protocol(override):
    with pytest.raises(ValueError, match="Fact-query protocol"):
        settings(**override)
