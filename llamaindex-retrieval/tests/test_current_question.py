import asyncio
import json

import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.current_question import full_history_prompt, parse_question, resolve_current_question
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline, conversation
from llamaindex_retrieval.schemas import ConversationMessage, SearchRequest
from test_rwkv_pipeline import FakeIndex, FakeModel, hit, native_result


@pytest.mark.parametrize("text", [
    '{"question":"first","question":"second"}', '{"question":"", "answer":"x"}',
    '{"question":null}', '{"question":" "}', '["question"]',
    '```json\n{"question":"x"}\n```', '{"question":"x"} trailing',
])
def test_invalid_correction_is_not_repaired(text):
    with pytest.raises((ValueError, TypeError)):
        parse_question(text)


class CorrectionModel(FakeModel):
    def __init__(self, correction, **kwargs):
        super().__init__(**kwargs)
        self.correction = correction
        self.corrections = []

    async def complete(self, messages, **kwargs):
        if (messages[0]["content"].startswith("更新检索问题")
                or messages[0]["content"].startswith('只输出JSON：{"question"')):
            self.corrections.append(messages[0]["content"])
            return native_result("planner", ">done</think>" + self.correction)
        return await super().complete(messages, **kwargs)


def test_no_history_bypasses_rewrite_and_preserves_original_bytes():
    pipe = RWKVPipeline(Settings(_env_file=None, native_history_protocol="current-question-v1"),
                        None, model=CorrectionModel("not used"))
    question = "  保留 原问题\n"
    current, task, events = asyncio.run(resolve_current_question(pipe, question, []))
    assert current == question and task == conversation(question, []) and not events
    assert not pipe.model.corrections


def test_corrected_task_reaches_search_reader_writer_without_old_query_reinsertion():
    current = "甲的二进制模式如何指定存储目录？"
    old = "甲的容器模式怎么持久化？"
    latest = "不用容器，改问二进制的存储目录。"
    index = FakeIndex({current: [hit("h1", "甲二进制使用 --store 设置目录。") ]})
    model = CorrectionModel(json.dumps({"question": current}, ensure_ascii=False),
        plan={"queries": [current], "fields": [current]}, writer_raw=">done</think>答案[资料 1]")
    pipe = RWKVPipeline(Settings(_env_file=None, native_history_protocol="current-question-v1",
        native_preserve_original_question=True),
        index, model=model)
    result = asyncio.run(pipe.ask(SearchRequest(question=latest,
        history=[ConversationMessage(role="user", content=old)])))
    assert [row[0] for row in index.calls] == [current]
    for call in model.calls:
        assert old not in call["prompt"] and latest not in call["prompt"]
        assert current in call["prompt"]
    event = result.generation["model_calls"][0]
    assert old in event["original_task"] and latest in event["original_task"]
    assert event["parsed_question"] == current and event["raw_text"] == ">done</think>" + model.correction
    assert result.retrieval["current_question_preserved"] is True
    assert result.retrieval["original_question_preserved"] is False


def test_invalid_correction_does_not_search_with_withdrawn_history():
    model = CorrectionModel('{"question":null}')
    index = FakeIndex({})
    pipe = RWKVPipeline(Settings(_env_file=None, native_history_protocol="current-question-v1"),
        index, model=model)
    result = asyncio.run(pipe.ask(SearchRequest(question="改问乙", history=[
        ConversationMessage(role="user", content="先问甲") ])))
    assert result.generation["status"] == "current_question_failed"
    assert result.answer == "" and not index.calls and not model.calls
    assert "parse_error" in result.generation["model_calls"][0]


def test_raw_default_retains_history_and_does_not_call_correction():
    pipe = RWKVPipeline(Settings(_env_file=None), None, model=CorrectionModel("unused"))
    history = [ConversationMessage(role="user", content="旧问题")]
    current, task, events = asyncio.run(resolve_current_question(pipe, "新问题", history))
    assert current == "新问题" and task == conversation("新问题", history)
    assert not events and not pipe.model.corrections


def test_generic_history_prompt_preserves_all_roles_without_case_specific_rules():
    history = [ConversationMessage(role="user", content="先比较甲与乙。"),
               ConversationMessage(role="assistant", content="之前的回答仅作历史数据。")]
    question = "只展开后者的用途"
    prompt = full_history_prompt(question, history)
    instruction, payload = prompt.split("\n", 1)
    assert json.loads(payload) == {"history": [m.model_dump() for m in history],
                                  "latest_question": question}
    assert all(word not in instruction for word in ("第二是X", "OpenSearch", "MongoDB"))

    corrected = "请说明乙的用途。"
    pipe = RWKVPipeline(Settings(_env_file=None, native_history_protocol="current-question-v2"),
                        None, model=CorrectionModel(json.dumps({"question": corrected}, ensure_ascii=False)))
    current, task, events = asyncio.run(resolve_current_question(pipe, question, history))
    assert current == corrected and task == conversation(corrected, [])
    assert events[0]["history_protocol"] == "current-question-v2"
    assert events[0]["parsed_question"] == corrected
    assert question in events[0]["original_task"]


@pytest.mark.parametrize("override", [{"native_task_matrix_enabled": True},
                                     {"native_transport": "rwkvos_batch"}])
def test_unsupported_pipelines_cannot_silently_ignore_history_protocol(override):
    with pytest.raises(ValueError, match="Current-question protocol"):
        Settings(_env_file=None, native_history_protocol="current-question-v1", **override)
