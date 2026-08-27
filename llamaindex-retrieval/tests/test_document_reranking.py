import json

import httpx
import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.document_reranking import LanguageModelDocumentReranker
from llamaindex_retrieval.query_planning import build_query_plan
from llamaindex_retrieval.schemas import SourceItem


def stream_response(content: str) -> httpx.Response:
    event = json.dumps(
        {"choices": [{"delta": {"content": content}}]},
        ensure_ascii=False,
    )
    return httpx.Response(
        200,
        text=f"data: {event}\n\ndata: [DONE]\n\n",
        headers={"content-type": "text/event-stream"},
    )


def source(document_id: str, title: str, snippet: str) -> SourceItem:
    return SourceItem(
        id=f"{document_id}-node",
        document_id=document_id,
        source="wiki",
        title=title,
        score=1.0,
        snippet=snippet,
    )


@pytest.mark.asyncio
async def test_reranker_selects_relevant_document_without_rewriting_evidence() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["contents"][0]
        if "文档标题：长城" in prompt:
            payload = {"relevant": True, "score": 3, "reason": "直接列出长城关隘"}
        else:
            payload = {"relevant": False, "score": 0, "reason": "泛指所有历史关隘"}
        return stream_response(json.dumps(payload, ensure_ascii=False))

    reranker = LanguageModelDocumentReranker(
        Settings(
            generation_password="secret",
            document_reranking_enabled=True,
        ),
        transport=httpx.MockTransport(handler),
    )
    sources = [
        source("passes", "关隘", "历史著名关隘包括函谷关、潼关。"),
        source("wall", "长城", "长城著名关隘包括山海关、嘉峪关。"),
    ]

    result = await reranker.rerank(
        "中国有哪些著名的长城关隘？",
        build_query_plan("中国有哪些著名的长城关隘？"),
        sources,
    )

    assert [item.document_id for item in result.sources] == ["wall"]
    assert result.sources[0].snippet == sources[1].snippet
    assert {decision.document_id for decision in result.decisions} == {"passes", "wall"}


@pytest.mark.asyncio
async def test_semantic_reranker_selects_documents_in_one_batch() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        prompt = json.loads(request.content)["contents"][0]
        assert "[d1] 标题：关隘" in prompt
        assert "[d2] 标题：长城" in prompt
        return stream_response(">\nd2\n只选择直接证据")

    reranker = LanguageModelDocumentReranker(
        Settings(
            generation_password="secret",
            semantic_pipeline_enabled=True,
            document_reranking_enabled=True,
        ),
        transport=httpx.MockTransport(handler),
    )
    sources = [
        source("passes", "关隘", "历史著名关隘包括函谷关、潼关。"),
        source("wall", "长城", "长城著名关隘包括山海关、嘉峪关。"),
    ]

    result = await reranker.rerank(
        "中国有哪些著名的长城关隘？",
        build_query_plan("中国有哪些著名的长城关隘？"),
        sources,
    )

    assert calls == 1
    assert result.strategy == "model_batch_with_lexical_safety"
    assert [item.document_id for item in result.sources] == ["wall", "passes"]
    assert result.sources[0].snippet == sources[1].snippet


@pytest.mark.asyncio
async def test_semantic_reranker_drops_rejected_extracted_evidence() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return stream_response("d2")

    reranker = LanguageModelDocumentReranker(
        Settings(
            generation_password="secret",
            semantic_pipeline_enabled=True,
            document_reranking_enabled=True,
        ),
        transport=httpx.MockTransport(handler),
    )
    sources = [
        source("history", "线路历史", "曾经调整过若干站名。").model_copy(
            update={"metadata": {"evidence_span_hashes": ["history-hash"]}}
        ),
        source("stations", "车站列表", "车站包括甲站、乙站。").model_copy(
            update={"metadata": {"evidence_span_hashes": ["station-hash"]}}
        ),
    ]

    result = await reranker.rerank(
        "线路有哪些站点？",
        build_query_plan("线路有哪些站点？"),
        sources,
    )

    assert result.strategy == "model_batch_evidence"
    assert [item.document_id for item in result.sources] == ["stations"]


def test_reranker_rejects_inconsistent_contract() -> None:
    raw = '{"relevant":true,"score":0,"reason":"无关"}'

    with pytest.raises(ValueError, match="disagree"):
        LanguageModelDocumentReranker._parse(raw)


def test_reranker_accepts_compact_small_model_contract() -> None:
    assert LanguageModelDocumentReranker._parse(
        ">\n3|直接给出所求事实\n后续解释不参与协议"
    ) == (True, 3, "直接给出所求事实")
