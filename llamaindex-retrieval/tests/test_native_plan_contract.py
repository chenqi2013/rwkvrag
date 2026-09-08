"""The shared task list has one model-authored scope for retrieval and reading."""

import json

import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.rwkv_pipeline import conversation, parse_plan, planner_prompt
from llamaindex_retrieval.schemas import ConversationMessage


def shared_settings(**kwargs):
    return Settings(native_plan_protocol="shared_tasks", **kwargs)


def test_shared_tasks_preserve_qualified_questions_without_generating_new_fields():
    tasks = ["对象甲的2.0版本采用什么单位？", "对象乙的3.0版本是否支持接口Z？"]
    plan = parse_plan(json.dumps(tasks), shared_settings())
    assert plan == {"queries": tasks, "fields": tasks}
    assert plan["queries"] is not plan["fields"]


def test_identical_tasks_are_deduplicated_before_field_ids_are_assigned():
    plan = parse_plan('["same","same","other"]', shared_settings())
    assert plan == {"queries": ["same", "other"], "fields": ["same", "other"]}


@pytest.mark.parametrize("value", [
    {"queries": ["q"], "fields": ["changed scope"]},
    {"queries": ["q"]}, [False], "q", [" "], ["x" * 2001],
    ["a", "b", "c"], [{"question": "q"}], [], None,
])
def test_shared_contract_rejects_conflicting_fields_and_malformed_tasks(value):
    with pytest.raises(ValueError):
        parse_plan(json.dumps(value), shared_settings(native_max_queries=2))


def test_history_and_latest_question_remain_exact_data_in_shared_prompt():
    history = [ConversationMessage(role="user", content="先查甲和乙的价格及单位。"),
               ConversationMessage(role="assistant", content="可比较两个对象。"),
               ConversationMessage(role="user", content="撤回价格，也不再查甲。")]
    latest = "只保留最后确认的范围。"
    task = conversation(latest, history)
    prompt = planner_prompt(task, shared_settings())
    assert prompt.endswith("任务：" + task)
    assert json.loads(prompt.split("任务：", 1)[1]) == {
        "latest_question": latest, "history": [row.model_dump() for row in history]}
    # This tests faithful delivery, not the model's semantic understanding.


def test_old_contract_remains_explicit_and_rejects_shared_only_payload():
    settings = Settings(native_plan_protocol="queries_fields")
    with pytest.raises(ValueError):
        parse_plan('{"queries":["q"]}', settings)
    assert parse_plan('{"queries":["q"],"fields":["f"]}', settings) == {
        "queries": ["q"], "fields": ["f"]}
