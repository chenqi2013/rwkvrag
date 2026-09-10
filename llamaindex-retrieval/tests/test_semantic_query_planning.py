import json

import httpx
import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.query_planning import build_query_plan
from llamaindex_retrieval.semantic_query_planning import LanguageModelQueryPlanner


def contract(**updates: object) -> dict[str, object]:
    return {
        "subject": "示例项目",
        "intent": "fact",
        "answer_shape": "single_fact",
        "set_semantics": "specific",
        "fields": [{"field_id": "f1", "question": "创建者是谁？", "relations": ["创建者"]}],
        "relations": ["创建者"],
        "queries": ["示例项目 创建者", "示例项目 创立人"],
        **updates,
    }


def stream_response(content: str) -> httpx.Response:
    event = json.dumps({"choices": [{"delta": {"content": content}}]}, ensure_ascii=False)
    return httpx.Response(
        200,
        text=f"data: {event}\n\ndata: [DONE]\n\n",
        headers={"content-type": "text/event-stream"},
    )


@pytest.mark.asyncio
async def test_planner_uses_model_review_and_caches_reviewed_contract() -> None:
    prompts = []
    reviewed = contract(subject="模型确认的对象", relations=["发起人"])

    async def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(json.loads(request.content)["contents"][0])
        return stream_response(
            json.dumps(
                contract() if len(prompts) == 1 else reviewed,
                ensure_ascii=False,
            )
        )

    planner = LanguageModelQueryPlanner(
        Settings(generation_password="secret", model_query_planning_cache_ttl=300),
        transport=httpx.MockTransport(handler),
    )
    question = "这个项目是谁建立的？"
    fallback = build_query_plan(question)
    result = await planner.plan(question, fallback)
    cached = await planner.plan(question, fallback)

    assert len(prompts) == 2
    assert question in prompts[0] and question in prompts[1]
    assert "示例项目" in prompts[1]
    assert result.strategy == "model"
    assert result.plan.subject == "模型确认的对象"
    assert result.plan.relations == ("发起人",)
    assert result.plan.queries == ("示例项目 创建者", "示例项目 创立人", question)
    assert cached == result


@pytest.mark.parametrize(
    ("question", "shape", "semantics"),
    [
        ("中国四大名著是哪几个？", "list", "all"),
        ("中国四大名著是哪四个？", "single_fact", "specific"),
        ("秦始皇有哪些伟大成就？", "narrative", "partial"),
        ("一个未知对象有哪些成员？", "summary", "latest"),
    ],
)
@pytest.mark.asyncio
async def test_immutable_planner_keeps_model_semantics_without_wording_overrides(
    question,
    shape,
    semantics,
) -> None:
    payload = contract(answer_shape=shape, set_semantics=semantics, subject="模型指定对象")
    raw = json.dumps(payload, ensure_ascii=False)
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return stream_response(raw)

    planner = LanguageModelQueryPlanner(
        Settings(generation_password="secret"),
        transport=httpx.MockTransport(handler),
    )
    result = await planner.plan_immutable(question)

    assert len(calls) == 1
    assert result.strategy == "model"
    assert result.plan.subject == "模型指定对象"
    assert result.plan.answer_shape == shape
    assert result.plan.set_semantics == semantics
    assert result.plan.analysis.expects_list == (shape == "list")
    assert result.plan.analysis.expects_complete_list == (shape == "list" and semantics == "all")
    assert result.raw_output == raw
    assert question in result.prompt
    assert question in result.plan.queries
    assert result.plan.fields[0].field_id == "f1"


@pytest.mark.parametrize("immutable", [False, True])
@pytest.mark.parametrize("failure", ["invalid_json", "invalid_contract", "http", "timeout"])
@pytest.mark.asyncio
async def test_planner_failure_uses_neutral_original_query(immutable, failure) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if failure == "http":
            return httpx.Response(503)
        if failure == "timeout":
            raise httpx.ReadTimeout("model timed out", request=request)
        return stream_response("not JSON" if failure == "invalid_json" else "{}")

    planner = LanguageModelQueryPlanner(
        Settings(generation_password="secret"),
        transport=httpx.MockTransport(handler),
    )
    question = "深圳地铁一号线有哪些站点？"
    fallback = build_query_plan(question)
    result = (
        await planner.plan_immutable(question)
        if immutable
        else await planner.plan(question, fallback)
    )
    assert result.strategy == "deterministic_fallback"
    assert result.plan == fallback
    assert result.plan.subject == ""
    assert result.plan.relations == ()
    assert result.error


@pytest.mark.parametrize(
    ("settings", "reason"),
    [
        ({"model_query_planning_enabled": False}, "disabled"),
        ({"generation_password": ""}, "generation_password_not_configured"),
    ],
)
@pytest.mark.asyncio
async def test_optional_planner_can_be_disabled_without_network(settings, reason) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("disabled planner must not make a request")

    planner = LanguageModelQueryPlanner(
        Settings(**settings),
        transport=httpx.MockTransport(handler),
    )
    fallback = build_query_plan("任意问题？")
    result = await planner.plan(fallback.original_question, fallback)
    assert result.plan == fallback
    assert result.error == reason


@pytest.mark.parametrize(
    "updates",
    [
        {"subject": ""},
        {"intent": "invented"},
        {"answer_shape": "invented"},
        {"set_semantics": "invented"},
        {"fields": []},
        {"fields": [{"field_id": "bad", "question": "问题"}]},
        {"queries": []},
    ],
)
def test_parser_rejects_invalid_protocol(updates) -> None:
    with pytest.raises(ValueError):
        LanguageModelQueryPlanner._parse(json.dumps(contract(**updates)))


def test_parser_deduplicates_queries_fields_and_keeps_field_identity() -> None:
    payload = contract(
        queries=["查询甲", "查询甲", "", "查询乙"],
        fields=[
            {"field_id": "f1", "question": "第一字段", "relations": ["创建", "创建"]},
            {"field_id": "f1", "question": "重复编号", "relations": []},
            {"field_id": "f2", "question": "第二字段", "relations": ["时间"]},
        ],
    )
    parsed = LanguageModelQueryPlanner._parse(json.dumps(payload))
    assert [field.field_id for field in parsed[4]] == ["f1", "f2"]
    assert parsed[4][0].relations == ("创建",)
    assert parsed[-1] == ("查询甲", "查询乙")


@pytest.mark.asyncio
async def test_query_limit_reserves_original_question_after_model_expansion() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(json.dumps(contract(queries=["查询甲", "查询乙", "查询丙"])))

    planner = LanguageModelQueryPlanner(
        Settings(generation_password="secret", model_query_planning_max_queries=3),
        transport=httpx.MockTransport(handler),
    )
    question = "任意原问题？"
    result = await planner.plan(question, build_query_plan(question))
    assert result.plan.queries == ("查询甲", "查询乙", question)
