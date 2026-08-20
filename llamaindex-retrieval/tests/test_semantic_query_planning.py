import json

import httpx
import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.query_planning import build_query_plan
from llamaindex_retrieval.semantic_query_planning import LanguageModelQueryPlanner


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


@pytest.mark.asyncio
async def test_model_planner_uses_structured_queries_and_keeps_original_question() -> None:
    requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        payload = json.loads(request.content)
        assert "不是回答问题" in payload["contents"][0]
        return stream_response(
            json.dumps(
                {
                    "subject": "深圳地铁1号线",
                    "relations": ["车站", "站点列表"],
                    "queries": [
                        "深圳地铁1号线 车站列表",
                        "深圳地铁一号线 沿途车站",
                        "罗宝线 站点",
                    ],
                },
                ensure_ascii=False,
            )
        )

    settings = Settings(
        generation_password="secret",
        generation_base_url="https://generation.example/v1",
    )
    planner = LanguageModelQueryPlanner(settings, transport=httpx.MockTransport(handler))
    fallback = build_query_plan("深圳一号线都停靠哪些地方？")

    result = await planner.plan("深圳一号线都停靠哪些地方？", fallback)

    assert result.strategy == "model"
    assert result.plan.subject == "深圳地铁1号线"
    assert result.plan.relations == ("车站", "站点列表")
    assert result.plan.queries == (
        "深圳地铁1号线 车站列表",
        "深圳地铁一号线 沿途车站",
        "罗宝线 站点",
        fallback.normalized_question,
    )
    assert result.model_queries == result.plan.queries[:-1]
    assert result.error is None

    cached = await planner.plan("深圳一号线都停靠哪些地方？", fallback)

    assert cached == result
    assert requests == 1


@pytest.mark.asyncio
async def test_model_planner_falls_back_when_response_is_not_valid_json() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return stream_response("我建议搜索深圳地铁和罗宝线。")

    settings = Settings(generation_password="secret")
    planner = LanguageModelQueryPlanner(settings, transport=httpx.MockTransport(handler))
    fallback = build_query_plan("深圳一号线都停靠哪些地方？")

    result = await planner.plan("深圳一号线都停靠哪些地方？", fallback)

    assert result.strategy == "deterministic_fallback"
    assert result.plan == fallback
    assert result.model_queries == ()
    assert result.error is not None


@pytest.mark.asyncio
async def test_model_planner_can_be_disabled() -> None:
    settings = Settings(
        generation_password="secret",
        model_query_planning_enabled=False,
    )
    planner = LanguageModelQueryPlanner(settings)
    fallback = build_query_plan("中国的首都是哪里？")

    result = await planner.plan("中国的首都是哪里？", fallback)

    assert result.strategy == "deterministic_fallback"
    assert result.plan == fallback
    assert result.error == "disabled"
