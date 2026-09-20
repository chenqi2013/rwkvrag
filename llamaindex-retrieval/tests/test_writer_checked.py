"""Opt-in prompt changes preserve evidence and never repair a model's answer."""
import json

import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline, conversation
from llamaindex_retrieval.schemas import ConversationMessage, SourceItem
from llamaindex_retrieval.writer_prompt import writer_prompt_checked, writer_prompt_v2
from test_rwkv_pipeline import native_result


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [True, False])
async def test_checked_prompt_preserves_sources_history_and_even_incorrect_raw_output(empty):
    calls = []
    raw = ">thinking</think>原文：虚构摘录[资料 N]\n错误结论😀[资料 9]"

    class Model:
        async def complete(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return native_result("writer", raw)

    sources = [] if empty else [SourceItem(id="source", document_id="doc", source="test",
        title="标题不是证据", score=1, snippet="原文😀\n| A | 0 | 不支持 |", metadata={})]
    history = [ConversationMessage(role="user", content="撤回之前的价格问题，保留版本。")]
    response = await RWKVPipeline(Settings(native_writer_prompt_protocol="evidence_checked"),
                                  None, Model()).ask_materials("核对当前要求", sources, history)
    assert len(calls) == 1 and calls[0][1]["stage"] == "writer"
    prompt = calls[0][0][0]["content"]
    assert conversation("核对当前要求", history) in prompt
    for source in sources:
        assert json.dumps(source.snippet, ensure_ascii=False) in prompt
    assert response.answer == raw and response.generation["answer_modified"] is False
    assert response.generation["writer_prompt_protocol"] == "evidence_checked"
    assert response.generation["citation_audit"]["semantic_support_verified"] is False
    assert response.generation["citation_audit"]["unknown_label_ids"] == [9]
    assert response.generation["citation_audit"]["invalid_labels"] == ["[资料 N]"]
    assert response.generation["citation_audit"]["quote_audit"]["failed"] == 1


def test_candidate_is_additive_and_old_defaults_remain_explicit():
    assert Settings().native_writer_prompt_protocol == "task_first"
    assert writer_prompt_checked("task", [], ["field"]).startswith(
        writer_prompt_v2("task", [], ["field"]))


def test_checked_prompt_admits_explicit_writer_state_with_canonical_transport():
    settings = Settings(native_writer_prompt_protocol="evidence_checked",
        native_transport="rwkvos_batch", native_writer_prefill="<think></think",
        rwkvos_writer_state_id="selected-writer-state",
        rwkvos_writer_prompt_protocol="rwkv_g1j_no_think_v1")
    assert settings.rwkvos_writer_state_id == "selected-writer-state"
