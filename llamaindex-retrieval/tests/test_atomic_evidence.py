import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import json

import pytest

from llamaindex_retrieval.atomic_evidence import (
    AtomicEvidenceService, AtomicRequest, candidates, source_claim, parse_selection, short_spans,
)
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.schemas import SourceItem


def source(text="设备甲功率 18 W。", kb="kb", identity="s1"):
    return SourceItem(id=identity, document_id="d1", source="upload", title="记录", score=1,
        snippet=text, metadata={"knowledge_base_id": kb, "source_sha256": "rev-a"})


def record():
    return {"id": "run", "knowledge_base_id": "kb", "claims": [], "sources": [], "calls": [], "issues": [], "coverage": {}, "index_version": "physical-v1"}


class Model:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.prompts = []

    async def complete(self, messages, **kwargs):
        self.prompts.append(messages)
        value = next(self.replies)
        raw, status = value if isinstance(value, tuple) else (value, "completed")
        trace = kwargs["trace"]
        trace.update(call_id=str(len(self.prompts)), transport="rwkvos_batch", provider_finish_reason="stop" if status == "completed" else "length",
            envelope={"valid": True, "answer_span": {"start": 0, "end": len(raw), "unit": "unicode_code_points"}})
        return SimpleNamespace(raw_text=raw, trace=trace, status=status)


def service(replies, **settings):
    return AtomicEvidenceService(Settings(_env_file=None, **settings), AsyncMock(), Mock(), Model(replies), reader=Model(['{"answer":"YES"}'] * 64))


def test_exact_offsets_emoji_table_headers_and_qa_parent():
    text = "# 型号甲😀\n| 模式 | 功率 |\n| --- | --- |\n| 标准 | 18 W |\n\n问：甲支持离线吗？\n答：不支持。\n"
    spans = short_spans(text)
    for s in spans:
        assert text[s["start"]:s["end"]] == s["text"]
        for c in s["context"]:
            assert text[c["start"]:c["end"]] == c["text"]
    row = next(s for s in spans if "| 标准" in s["text"])
    assert any("模式 | 功率" in c["text"] for c in row["context"])
    answer = next(s for s in spans if "答：" in s["text"])
    assert answer["text"] == "问：甲支持离线吗？\n答：不支持。\n"
    assert sum("问：甲支持离线吗？" in s["text"] for s in spans) == 1


def test_long_records_rank_relevant_short_span_without_rewriting():
    text = "\n".join(f"维护记录{i}：外壳完好。" for i in range(200)) + "\n最终核准：设备甲功率 18 W。"
    spans, coverage = candidates(source(text), AtomicRequest(object="设备甲", attribute="功率"))
    assert len(spans) <= 6 and any("18 W" in s["text"] for s in spans)
    assert coverage["unexamined_spans"] > 0
    assert all(text[s["start"]:s["end"]] == s["text"] for s in spans)


@pytest.mark.parametrize("raw", ['[true]', '[0]', '[1,1]', '[9]', '{"ids":[1]}'])
def test_invalid_selections_are_not_coerced(raw):
    with pytest.raises(ValueError):
        parse_selection(raw, 2)


def test_zero_negation_and_missing_text_are_saved_without_value_inference():
    for text in ["实际交付 0 台，未量产。", "实际交付未记载。", "正式更正：功率25 W，旧值18 W撤回。"]:
        span = short_spans(text)[0]
        for supported in [True, False]:
            claim = source_claim(span, supported)
            assert claim["statement_quote"] == text
            assert claim["value_quote"] is None
            assert claim["normalization_status"] == "not_performed"
            assert claim["kind"] == ("reader_supported" if supported else "unconfirmed")


def test_parent_metadata_offsets_are_not_misrepresented_as_chunk_offsets():
    s = source("| 标准 | 18 W |")
    s.metadata["context_spans"] = [{"text": "功率单位 W"}]
    spans, _ = candidates(s, AtomicRequest(object="设备甲", attribute="功率"))
    context = spans[0]["context"][0]
    assert context["origin"] == "saved_source_metadata" and context["context_index"] == 0
    assert context["start"] == 0 and context["end"] == 6
    assert source_claim(spans[0], True)["statement_position"] == {"start": 0, "end": len(s.snippet)}


@pytest.mark.asyncio
async def test_conflicting_source_ids_cannot_bind_to_another_snapshot():
    svc = service([])
    with pytest.raises(ValueError, match="conflicting_source_identity"):
        await svc.extract(record(), AtomicRequest(object="设备甲", attribute="功率"), [source(), source("其他值")])
    assert not svc.model.prompts


@pytest.mark.asyncio
async def test_conflicting_sources_remain_separate_and_raw_outputs_immutable():
    texts = [source(), source("设备甲功率 25 W。", identity="s2")]
    originals = deepcopy([s.model_dump() for s in texts])
    svc = service(['[1]', '[1]'])
    run = await svc.extract(record(), AtomicRequest(object="设备甲", attribute="功率"), texts)
    assert [c["statement_quote"] for c in run["claims"]] == ["设备甲功率 18 W。", "设备甲功率 25 W。"]
    assert [c["binding"]["source_id"] for c in run["claims"]] == ["s1", "s2"]
    assert run["calls"][0]["raw_text"] == '[1]'
    assert all(c["semantic_verified"] is False for c in run["claims"])
    assert originals == [s.model_dump() for s in texts]
    # The extractor sees only the selected short evidence, not all candidates.
    assert "25 W" not in svc.reader.prompts[0][0]["content"]


@pytest.mark.asyncio
async def test_length_and_invalid_schema_are_failures_not_no_evidence():
    for reply in [('[1]', 'length'), '{"ids":[1]}']:
        svc = service([reply])
        run = await svc.extract(record(), AtomicRequest(object="设备甲", attribute="功率"), [source()])
        assert run["status"] == "incomplete" and run["issues"] and not run["claims"]
        assert run["calls"][0]["raw_text"]


@pytest.mark.asyncio
async def test_call_budget_keeps_prior_claims_and_records_incomplete():
    svc = service(['[1]'], atomic_max_calls=2)
    run = await svc.extract(record(), AtomicRequest(object="设备甲", attribute="功率"), [source(), source(identity="s2")])
    assert len(run["calls"]) == 2 and len(run["claims"]) == 1
    assert run["status"] == "incomplete" and run["issues"][-1]["detail"] == "call_budget_exhausted"


@pytest.mark.asyncio
async def test_cross_knowledge_base_input_is_rejected():
    svc = service([])
    with pytest.raises(ValueError, match="knowledge_base_mismatch"):
        await svc.extract(record(), AtomicRequest(object="设备甲", attribute="功率"), [source(kb="other")])
    assert not svc.model.prompts


@pytest.mark.asyncio
async def test_reader_rejection_cannot_be_overridden_by_locator():
    svc = service(['[1]'])
    svc.reader = Model(['{"answer":"NO"}', '{"answer":"NO"}'])
    run = await svc.extract(record(), AtomicRequest(object="设备甲", attribute="实际交付"), [source("设备甲预约80台。")])
    assert run["claims"][0]["kind"] == "unconfirmed" and len(svc.model.prompts) == 1
    assert run["status"] == "completed" and run["coverage"]["retrieval_exhaustive"] is False


@pytest.mark.asyncio
async def test_rejected_absence_text_remains_visible_without_inventing_zero():
    svc = service(['[1]'])
    svc.reader = Model(['{"answer":"NO"}', '{"answer":"YES"}'])
    run = await svc.extract(record(), AtomicRequest(object="设备甲", attribute="实际交付"), [source("设备甲实际交付未记载。")])
    assert run["claims"][0]["kind"] == "unconfirmed"
    assert run["claims"][0]["value_quote"] is None
    assert run["claims"][0]["statement_quote"] == "设备甲实际交付未记载。"


@pytest.mark.asyncio
async def test_index_is_pinned_and_completion_is_persisted():
    svc = service([], atomic_timeout_seconds=1)
    svc.index.index_name = "alias"
    svc.index.versions.current.return_value = "physical-v2"
    def slow(*args, **kwargs):
        assert kwargs["knowledge_base_id"] == "kb"
        return []
    svc.index.search_chunks.side_effect = slow
    run = await svc.inspect("kb", AtomicRequest(object="设备甲", attribute="功率"))
    assert svc.index.index_name == "alias" and run["index_version"] == "physical-v2"
    svc.repo.finish_atomic_run.assert_awaited_once()
    assert run["coverage"]["retrieval_exhaustive"] is False


@pytest.mark.asyncio
async def test_cancellation_saves_incomplete_history():
    svc = service([])
    svc.index.versions.current.return_value = "v1"
    svc.index.search_chunks.return_value = []
    async def cancel(*args):
        raise asyncio.CancelledError
    svc.extract = cancel
    with pytest.raises(asyncio.CancelledError):
        await svc.inspect("kb", AtomicRequest(object="设备甲", attribute="功率"))
    saved = svc.repo.finish_atomic_run.call_args.args[0]
    assert saved["status"] == "incomplete" and saved["issues"][0]["kind"] == "cancelled"


@pytest.mark.asyncio
async def test_actual_timeout_saves_incomplete_without_inventing_absence():
    svc = service([], atomic_timeout_seconds=1)
    svc.index.versions.current.return_value = "v1"
    svc.index.search_chunks.return_value = []
    async def slow(*args):
        await asyncio.sleep(10)
    svc.extract = slow
    run = await svc.inspect("kb", AtomicRequest(object="设备甲", attribute="功率"))
    assert run["status"] == "incomplete"
    assert run["issues"] == [{"kind": "timeout", "remote_execution_cancelled": False}]
    assert not run["claims"]
    svc.repo.finish_atomic_run.assert_awaited_once()
