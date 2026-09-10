import json

import httpx
import pytest


from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.evidence_quality import is_repetitive_garbage
from llamaindex_retrieval.generation import AnswerGenerationError, EvidenceAnswerGenerator
from llamaindex_retrieval.schemas import SourceItem


@pytest.mark.parametrize(
    "raw",
    [
        "北京。",
        "北京。[资料 99]",
        "根据检索到的资料，无法确定。",
        "Assistant: <think>推理</think>北京。\n用户：继续",
        "[资料 1] 标题：首都\n北京。\n北京。\n北京。",
        "  **北京** 😀\n\n",
        "",
    ],
)
@pytest.mark.asyncio
async def test_default_writer_preserves_every_streamed_output_character(raw: str) -> None:
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        events = [
            json.dumps({"choices": [{"delta": {"content": part}}]}, ensure_ascii=False)
            for part in (raw[:3], raw[3:])
        ]
        return httpx.Response(
            200, text="".join(f"data: {event}\n\n" for event in events) + "data: [DONE]\n\n"
        )

    generator = EvidenceAnswerGenerator(
        Settings(generation_password="secret"),
        transport=httpx.MockTransport(handler),
    )
    result = await generator.generate_with_trace("任意问题", [source()])

    assert len(calls) == 1
    assert result.answer.encode("utf-8") == raw.encode("utf-8")
    assert result.raw_output == raw
    assert result.trace["request"]["payload"]["password"] == "***"
    assert "secret" not in json.dumps(result.trace)


@pytest.mark.parametrize("failure", ["http", "timeout"])
@pytest.mark.asyncio
async def test_generation_failure_retains_diagnostic_trace(failure) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ReadTimeout("model timed out", request=request)
        return httpx.Response(503, headers={"x-request-id": "request-123"})

    generator = EvidenceAnswerGenerator(
        Settings(generation_password="secret"),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(AnswerGenerationError) as caught:
        await generator.generate_with_trace("任意问题", [source()])
    assert caught.value.trace["error"]["type"] in {"HTTPStatusError", "ReadTimeout"}
    assert caught.value.trace["request"]["payload"]["password"] == "***"
    if failure == "http":
        assert caught.value.trace["response"]["status_code"] == 503
        assert caught.value.trace["response"]["headers"]["x-request-id"] == "request-123"


def source() -> SourceItem:
    return SourceItem(
        id="capital-1",
        document_id="capital",
        source="finewiki-zh",
        title="首都",
        score=1.0,
        snippet="中华人民共和国的首都是北京。",
    )


def test_semantic_writer_puts_evidence_before_question_with_minimal_controls() -> None:
    generator = EvidenceAnswerGenerator(Settings(generation_output_mode="immutable"))

    prompt = generator._prompt(
        "换一种说法也要找到这条事实",
        [source()],
        subject="中华人民共和国",
        relations=("首都", "国都"),
        answer_shape="single_fact",
        set_semantics="specific",
        fields=(("f1", "首都城市", ("首都", "国都")),),
    )

    assert prompt.startswith("system:\n知识库问答助手；只能依据资料")
    assert "任务契约" not in prompt
    assert "answer_shape" not in prompt
    assert prompt.index("资料：") < prompt.index("中华人民共和国的首都是北京。")
    assert prompt.index("中华人民共和国的首都是北京。") < prompt.index(
        "问题：换一种说法也要找到这条事实"
    )
    assert prompt.rstrip().endswith("assistant:")


def test_detects_periodic_parser_garbage_without_rejecting_normal_prose() -> None:
    assert is_repetitive_garbage("唐朝 > 军定\n\n" + "整須教定開領思" * 40) is True
    assert (
        is_repetitive_garbage(
            "唐高祖李渊是唐朝开国皇帝。他于618年称帝并建立唐朝，此后逐步完成统一。" * 4
        )
        is False
    )


@pytest.mark.asyncio
async def test_semantic_generator_does_not_rewrite_model_answer() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=('data: {"choices":[{"delta":{"content":"北京。"}}]}\n\ndata: [DONE]\n\n'),
        )

    generator = EvidenceAnswerGenerator(
        Settings(
            generation_base_url="http://rwkv.test/v1",
            generation_password="test-password",
            semantic_pipeline_enabled=True,
        ),
        transport=httpx.MockTransport(handler),
    )

    answer = await generator.generate(
        "中国的首都是哪个城市？",
        [source()],
        trusted_evidence=True,
    )

    assert answer == "北京。"


@pytest.mark.asyncio
async def test_generator_skips_model_call_without_sources() -> None:
    generator = EvidenceAnswerGenerator(Settings())

    assert await generator.generate("没有资料怎么办？", []) == "未检索到可用于回答该问题的资料。"


@pytest.mark.asyncio
async def test_immutable_generator_returns_byte_exact_model_output_and_trace() -> None:
    raw = "Assistant: <think>推理</think>北京。[资料 2]\n用户：继续"

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                "data: "
                + json.dumps({"choices": [{"delta": {"content": raw}}]}, ensure_ascii=False)
                + "\n\ndata: [DONE]\n\n"
            ),
        )

    generator = EvidenceAnswerGenerator(
        Settings(
            generation_password="test-password",
            generation_output_mode="immutable",
        ),
        transport=httpx.MockTransport(handler),
    )

    result = await generator.generate_with_trace(
        "中国的首都是哪个城市？",
        [source()],
        trusted_evidence=True,
    )

    assert result.answer == raw
    assert result.raw_output == raw
    assert result.trace["request"]["payload"]["password"] == "***"
    assert result.trace["request"]["password_present"] is True
    assert result.trace["response"]["end_reason"] == "done"
    assert result.prompt.index("资料：") < result.prompt.index("问题：中国的首都是哪个城市？")
    assert result.prompt.rstrip().endswith("assistant:")


@pytest.mark.asyncio
async def test_generator_reads_current_model_from_models_endpoint() -> None:
    request_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        assert request.url == "http://models.test/v1/models"
        return httpx.Response(200, json={"data": [{"id": "rwkv-current"}]})

    generator = EvidenceAnswerGenerator(
        Settings(generation_models_url="http://models.test/v1/models"),
        transport=httpx.MockTransport(handler),
    )

    assert await generator.current_model() == "rwkv-current"
    assert await generator.current_model() == "rwkv-current"
    assert request_count == 1


@pytest.mark.asyncio
async def test_generator_allows_model_list_failure() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    generator = EvidenceAnswerGenerator(
        Settings(generation_models_url="http://models.test/v1/models"),
        transport=httpx.MockTransport(handler),
    )

    assert await generator.current_model() is None
