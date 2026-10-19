"""RWKV-only service contracts. No real model, database or index calls.

Legacy rule-answering tests retired with their implementation; their original
bytes and the pre-change test results remain in the local removal audit.
"""
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline
from llamaindex_retrieval.schemas import SearchRequest, SourceItem
from llamaindex_retrieval.service import SearchService
from test_rwkv_pipeline import FakeIndex, FakeModel, hit


@pytest.mark.parametrize("override", [
    {"rag_pipeline": "existing"}, {"generation_output_mode": "legacy"},
    {"native_history_protocol": "current-question-g1k-v1"},
])
def test_removed_modes_are_rejected_not_silently_aliased(override):
    with pytest.raises(ValueError):
        Settings(_env_file=None, **override)


@pytest.mark.parametrize("field,value", [
    ("rag_pipeline", "existing"), ("generation_output_mode", "legacy"),
])
def test_service_rejects_mutated_legacy_settings_before_creating_pipeline(field, value):
    settings = Settings(_env_file=None)
    setattr(settings, field, value)
    with pytest.raises(ValueError, match="immutable"):
        SearchService(settings, None)


def test_rule_modules_and_fallback_entry_points_are_removed():
    import llamaindex_retrieval.service as module
    from llamaindex_retrieval.lexical_index import LexicalIndex

    root = Path(module.__file__).parent
    for name in ("active_retrieval", "document_reranking", "evidence_extraction",
                 "evidence_gate", "evidence_utils", "generation", "qa_analysis",
                 "query_planning", "question_patterns", "semantic_query_planning"):
        assert not (root / (name + ".py")).exists()
    assert not hasattr(SearchService, "_deterministic_answer")
    assert not hasattr(LexicalIndex, "search_plain")
    assert not hasattr(LexicalIndex, "station_list_chunks")


def make_service(monkeypatch, *, model, index=None, **options):
    monkeypatch.setattr("llamaindex_retrieval.service.RWKVPipeline",
        lambda settings, index, recorder: RWKVPipeline(settings, index, model=model))
    return SearchService(Settings(_env_file=None, **options), index)


async def test_reader_none_stays_empty_even_for_a_cause_question(monkeypatch):
    question = "为何发生该事件？"
    model = FakeModel(plan={"queries": [question], "fields": ["原因"]},
                      resolver_outputs={"candidate": "f1: NONE"})
    index = FakeIndex({question: [hit("candidate", "事件由于设备故障而发生。") ]})
    service = make_service(monkeypatch, model=model, index=index, native_empty_evidence_policy="fail")
    result = await service.ask(SearchRequest(question=question))
    assert result.sources == [] and result.answer == ""
    assert result.generation["status"] == "no_evidence"
    assert not any(call["stage"] == "writer" for call in model.calls)
    assert "NONE" in result.generation["model_calls"][-1]["raw_text"]
    await service.aclose()
    assert model.closed


@pytest.mark.parametrize("raw", [
    ">思考</think>根据检索到的资料，无法确定。",  # no rule-based refusal replacement
    ">思考</think>没有引用的原始答案。",          # no fabricated citation
    ">思考</think>答案[资料 99]",                # no citation reassignment
    ">思考</think>数值为987654，无法支持的句子也保留。\n",
])
async def test_writer_raw_output_is_never_replaced_deleted_or_repaired(monkeypatch, raw):
    material = SourceItem(id="m", document_id="d", source="fixture", title="原文",
                          score=1, snippet="实体甲于某年成立。")
    model = FakeModel(writer_raw=raw)
    service = make_service(monkeypatch, model=model, native_answer_quality_policy="fail_citation")
    result = await service.ask_materials("实体甲何时成立？", [material])
    assert result.answer == result.generation["raw_model_answer"] == raw
    assert result.generation["answer_modified"] is False
    assert result.generation["status"] == "answer_quality_failed"
    assert [call["stage"] for call in model.calls] == ["writer"]
    await service.aclose()


async def test_search_only_limits_display_and_preserves_ranked_sources(monkeypatch):
    native = AsyncMock()
    sources = [SourceItem(id=str(i), document_id="shared", source="fixture", title="",
                          score=1, snippet=f"原文{i}") for i in range(6)]
    native.search.return_value = sources
    monkeypatch.setattr("llamaindex_retrieval.service.RWKVPipeline", lambda *a, **k: native)
    service = SearchService(Settings(_env_file=None), None)
    request = SearchRequest(question="查询", top_k=2)
    result = await service.search(request)
    native.search.assert_awaited_once_with(request)
    assert result.results == sources[:2]
    assert result.retrieval["candidate_count"] == 6
    assert result.retrieval["per_document_limit"] is None
    await service.aclose()
    native.aclose.assert_awaited_once()
