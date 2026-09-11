import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.native_rwkv import NativeRWKVResult
from llamaindex_retrieval.reader_prompt import parse_binary_decision
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline
from llamaindex_retrieval.schemas import SourceItem


@pytest.mark.parametrize("raw", [
    '{"answer":"YES","answer":"NO"}', '{"answer":"YES","extra":0}',
    '[{"answer":"YES"}]', '[["answer","YES"]]', '{"answer":true}',
    '{"result":"YES"}', 'YES', '{"answer":"yes"}',
    '{"answer":"YES"} because relevant', '```json\n{"answer":"YES"}\n```',
])
def test_binary_selection_rejects_ambiguous_or_unrequested_formats(raw):
    with pytest.raises(ValueError):
        parse_binary_decision(raw)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "length"])
async def test_binary_reader_binds_one_original_unit_and_query_and_keeps_raw(status):
    calls = []
    class Model:
        async def complete(self, messages, *, stage, assistant_prefill, **kwargs):
            prompt = messages[0]["content"]
            calls.append(prompt)
            assert '待查问题：["模型生成的独立问题"]' in prompt
            assert "完整对话任务：" not in prompt
            raw = '>{"answer":"YES"}' if "甲事实" in prompt else '>{"answer":"NO"}'
            return NativeRWKVResult(status=status, raw_text=raw, finish_reason="stop",
                trace={"stage":stage,"status":status,"raw_text":raw,"prefill":assistant_prefill})

    settings = Settings(native_resolver_protocol="binary_query", native_task_source="queries",
        native_resolver_task_grouping="individual", native_resolver_prefill="<think></think",
        native_resolver_window_characters=256, native_resolver_overlap_characters=20)
    sources = [SourceItem(id="a", document_id="a", source="fixture", title="甲", score=1,
        snippet="甲事实。"), SourceItem(id="b", document_id="b", source="fixture", title="乙", score=1,
        snippet="乙旁支。")]
    evidence, events = await RWKVPipeline(settings, None, Model())._resolve(
        "原始历史仍由上游Planner保存", ["模型生成的独立问题"], sources)
    assert len(calls) == len(events) == 2
    assert events[0]["raw_text"] == '>{"answer":"YES"}'
    if status == "completed":
        assert len(evidence) == 1 and evidence[0].snippet == sources[0].snippet
        assert evidence[0].metadata["span_start"] == 0
        assert evidence[0].metadata["span_end"] == len(sources[0].snippet)
        assert events[0]["decision"] == "YES" and events[1]["decision"] == "NO"
    else:
        assert evidence == [] and all("parse_error" in event for event in events)


@pytest.mark.parametrize("invalid", [
    {"native_task_source":"fields"}, {"native_resolver_task_grouping":"joint"},
    {"rwkvos_reader_input_layout":"task_last"}, {"rwkvos_reader_state_id":"old-state"},
    {"rwkvos_state_id":"old-global-state"},
    {"native_resolver_format_repair":True},
])
def test_binary_reader_cannot_reuse_incompatible_task_or_state_protocol(invalid):
    settings = dict(native_resolver_protocol="binary_query", native_task_source="queries",
        native_resolver_task_grouping="individual")
    with pytest.raises(ValueError):
        Settings(**(settings | invalid))
