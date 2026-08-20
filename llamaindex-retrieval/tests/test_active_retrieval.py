import json

import httpx
import pytest

from llamaindex_retrieval.active_retrieval import ActiveRetrievalAgent
from llamaindex_retrieval.config import Settings
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


def source() -> SourceItem:
    return SourceItem(
        id="source-1",
        document_id="document-1",
        source="finewiki-zh",
        title="深圳地铁1号线",
        score=1.0,
        snippet="该线路共设30个车站，但当前片段没有列出站名。",
    )


@pytest.mark.asyncio
async def test_active_retrieval_requests_new_bm25_queries() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert "BM25 补充查询规划器" in payload["contents"][0]
        return stream_response(
            json.dumps(
                {
                    "queries": ["深圳地铁车站列表 1号线", "罗宝线 全部车站"],
                },
                ensure_ascii=False,
            )
        )

    settings = Settings(generation_password="secret")
    agent = ActiveRetrievalAgent(settings, transport=httpx.MockTransport(handler))

    result = await agent.decide(
        "深圳地铁1号线所有站点",
        [source()],
        used_queries=("深圳地铁1号线所有站点",),
        round_number=1,
    )

    assert result.error is None
    assert result.decision is not None
    assert result.decision.action == "bm25_search"
    assert result.decision.queries == (
        "深圳地铁车站列表 1号线",
        "罗宝线 全部车站",
    )


@pytest.mark.asyncio
async def test_active_retrieval_rejects_unapproved_tool_action() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return stream_response(
            '{"action":"bash","queries":["rg 北京"],"reason":"尝试文件搜索"}'
        )

    settings = Settings(generation_password="secret")
    agent = ActiveRetrievalAgent(settings, transport=httpx.MockTransport(handler))

    result = await agent.decide(
        "中国的首都是哪里？",
        [source()],
        used_queries=(),
        round_number=1,
    )

    assert result.decision is None
    assert result.error is not None
    assert "only queries" in result.error
