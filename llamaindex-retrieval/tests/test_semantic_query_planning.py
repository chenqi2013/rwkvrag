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
        assert "资料中可能出现的 2 到 6 个简短同义表达" in payload["contents"][0]
        return stream_response(
            json.dumps(
                {
                    "subject": "深圳地铁1号线",
                    "intent": "list",
                    "answer_shape": "list",
                    "set_semantics": "partial",
                    "fields": [
                        {
                            "field_id": "f1",
                            "question": "沿途停靠哪些车站",
                            "relations": ["车站", "站点列表"],
                        }
                    ],
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
    assert result.plan.analysis.intent == "list"
    assert result.plan.answer_shape == "list"
    assert result.plan.set_semantics == "partial"
    assert result.plan.fields[0].question == "沿途停靠哪些车站"
    assert result.plan.queries[:3] == (
        "深圳地铁1号线 车站列表",
        "深圳地铁一号线 沿途车站",
        "罗宝线 站点",
    )
    assert fallback.normalized_question in result.plan.queries
    assert any(query in result.plan.queries for query in fallback.queries)
    assert result.model_queries == (
        "深圳地铁1号线 车站列表",
        "深圳地铁一号线 沿途车站",
        "罗宝线 站点",
    )
    assert result.error is None

    cached = await planner.plan("深圳一号线都停靠哪些地方？", fallback)

    assert cached == result
    assert requests == 1


@pytest.mark.asyncio
async def test_deterministic_planner_handles_role_attribution_without_model() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return stream_response(json.dumps({
            "subject": "西游记",
            "intent": "fact",
            "answer_shape": "single_fact",
            "set_semantics": "specific",
            "fields": [{
                "field_id": "f1",
                "question": "西游记的作者是谁？",
                "relations": ["作者", "作者姓名"],
            }],
            "queries": [
                "西游记的作者是谁？",
                "西游记的作者是哪位作家？",
                "西游记的作者是谁？",
            ],
        }, ensure_ascii=False))

    settings = Settings(
        generation_password="secret",
        generation_base_url="https://generation.example/v1",
    )
    planner = LanguageModelQueryPlanner(settings, transport=httpx.MockTransport(handler))
    fallback = build_query_plan("西游记是哪个作者写的？")

    result = await planner.plan("西游记是哪个作者写的？", fallback)

    assert result.strategy == "deterministic_fallback"
    assert result.plan.subject == "西游记"
    assert result.plan.relations == ("作者", "写")
    assert result.plan.fields[0].relations == ("作者", "写")
    assert result.model_queries == ()


@pytest.mark.asyncio
async def test_semantic_pipeline_uses_model_for_structured_role_question() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return stream_response(json.dumps({
            "subject": "西游记",
            "intent": "agent",
            "answer_shape": "single_fact",
            "set_semantics": "specific",
            "fields": [{
                "field_id": "f1",
                "question": "西游记的作者是谁",
                "relations": ["作者", "创作者"],
            }],
            "relations": ["作者", "创作者"],
            "queries": ["西游记 作者", "西游记 创作者"],
        }, ensure_ascii=False))

    planner = LanguageModelQueryPlanner(
        Settings(
            generation_password="secret",
            semantic_pipeline_enabled=True,
        ),
        transport=httpx.MockTransport(handler),
    )
    fallback = build_query_plan("西游记是哪个作者写的？")

    result = await planner.plan(fallback.original_question, fallback)

    assert result.strategy == "model"
    assert result.plan.relations == ("作者", "创作者")
    assert result.plan.analysis.intent == "agent"


@pytest.mark.asyncio
async def test_model_planner_repairs_answer_used_as_subject() -> None:
    requests = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            payload = {
                "subject": "吴承恩",
                "intent": "definition",
                "answer_shape": "single_fact",
                "set_semantics": "specific",
                "fields": [{
                    "field_id": "f1",
                    "question": "《西游记》的作者是谁？",
                    "relations": ["作者", "作者姓名"],
                }],
                "queries": ["《西游记》的作者是谁？"],
            }
        else:
            payload = {
                "subject": "西游记",
                "intent": "agent",
                "answer_shape": "single_fact",
                "set_semantics": "specific",
                "fields": [{
                    "field_id": "f1",
                    "question": "《西游记》的作者是谁？",
                    "relations": ["作者", "创作者"],
                }],
                "relations": ["作者", "创作者"],
                "queries": ["西游记 作者", "西游记 创作者"],
            }
        return stream_response(json.dumps(payload, ensure_ascii=False))

    settings = Settings(
        generation_password="secret",
        generation_base_url="https://generation.example/v1",
    )
    planner = LanguageModelQueryPlanner(
        settings,
        transport=httpx.MockTransport(handler),
    )

    result = await planner.plan(
        "《西游记》出自谁之手？",
        build_query_plan("《西游记》出自谁之手？"),
    )

    assert requests == 1
    assert result.strategy == "model"
    assert result.plan.subject == "西游记"
    assert result.plan.relations == ("作者", "作者姓名")
    assert result.model_queries == ("《西游记》的作者是谁？",)


@pytest.mark.parametrize(
    "question",
    (
        "中国四大名著是哪四个？",
        "中国历史上的四大名著是哪四个？",
    ),
)
@pytest.mark.asyncio
async def test_model_planner_preserves_structural_list_constraint(question: str) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return stream_response(json.dumps({
            "subject": "四大名著",
            "intent": "definition",
            "answer_shape": "single_fact",
            "set_semantics": "specific",
            "fields": [{
                "field_id": "f1",
                "question": "四大名著的具体名称",
                "relations": ["四大名著的名称", "完整列表"],
            }],
            "queries": [
                "中国四大名著的全称是什么",
                "中国四大名著的完整列表是什么",
                "中国四大名著的标准名称有哪些",
            ],
        }, ensure_ascii=False))

    settings = Settings(
        generation_password="secret",
        generation_base_url="https://generation.example/v1",
    )
    planner = LanguageModelQueryPlanner(settings, transport=httpx.MockTransport(handler))
    fallback = build_query_plan(question)

    result = await planner.plan(question, fallback)

    assert result.strategy == "model"
    assert result.plan.analysis.intent == "list"
    assert result.plan.answer_shape == "list"
    assert result.plan.set_semantics == "all"
    assert result.plan.subject == fallback.subject
    assert result.plan.relations == fallback.relations
    assert result.plan.queries[:2] == fallback.queries[:2]


@pytest.mark.asyncio
async def test_model_planner_cannot_change_list_question_into_single_fact() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return stream_response(json.dumps({
            "subject": "秦始皇",
            "intent": "fact",
            "answer_shape": "single_fact",
            "set_semantics": "specific",
            "fields": [{
                "field_id": "f1",
                "question": "秦始皇的主要功绩是什么",
                "relations": ["功绩"],
            }],
            "relations": ["功绩"],
            "queries": ["秦始皇的主要功绩是什么"],
        }, ensure_ascii=False))

    planner = LanguageModelQueryPlanner(
        Settings(generation_password="secret"),
        transport=httpx.MockTransport(handler),
    )
    fallback = build_query_plan("秦始皇有哪些丰功伟绩")

    result = await planner.plan(fallback.original_question, fallback)

    assert result.plan.analysis.intent == "list"
    assert result.plan.answer_shape == "list"
    assert result.plan.subject == fallback.subject
    assert result.plan.relations == fallback.relations


@pytest.mark.asyncio
async def test_model_planner_cannot_downgrade_narrative_request() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return stream_response(json.dumps({
            "subject": "草船借箭",
            "intent": "fact",
            "answer_shape": "single_fact",
            "set_semantics": "specific",
            "fields": [{
                "field_id": "f1",
                "question": "草船借箭的经过",
                "relations": ["经过", "过程"],
            }],
            "relations": ["经过", "过程"],
            "queries": ["草船借箭 经过", "草船借箭 过程"],
        }, ensure_ascii=False))

    settings = Settings(
        generation_password="secret",
        generation_base_url="https://generation.example/v1",
    )
    planner = LanguageModelQueryPlanner(settings, transport=httpx.MockTransport(handler))
    fallback = build_query_plan("讲讲草船借箭的故事")

    result = await planner.plan("讲讲草船借箭的故事", fallback)

    assert result.plan.answer_shape == "narrative"


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


def test_model_planner_rejects_a_guessed_answer_as_subject() -> None:
    question = "水浒传里赤手空拳打死老虎的是谁？"

    assert LanguageModelQueryPlanner._subject_is_supported(question, "水浒传") is True
    assert LanguageModelQueryPlanner._subject_is_supported(question, "宋江") is False
    assert LanguageModelQueryPlanner._subject_matches_fallback("水浒传", "赤手空拳打死老虎的人物") is False
    assert LanguageModelQueryPlanner._subject_matches_fallback("马斯克", "埃隆·马斯克") is True


def test_narrative_agent_relation_still_uses_model_query_expansion() -> None:
    simple = build_query_plan("宇树科技创始人是谁？")
    narrative = build_query_plan("水浒传里赤手空拳打死老虎的是谁？")

    assert LanguageModelQueryPlanner._fallback_contract_is_sufficient(simple) is True
    assert LanguageModelQueryPlanner._fallback_contract_is_sufficient(narrative) is False


def test_model_planner_requires_specific_hypothesis_for_event_agent_question() -> None:
    prompt = LanguageModelQueryPlanner._prompt("水浒传里赤手空拳打死老虎的是谁？")

    assert "至少一条必须包含你推测的具体人物姓名" in prompt
    assert "人物、英雄、主角、人名" in prompt
