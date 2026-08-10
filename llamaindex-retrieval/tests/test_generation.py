import json

import httpx
import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.generation import EvidenceAnswerGenerator
from llamaindex_retrieval.schemas import SourceItem


def source() -> SourceItem:
    return SourceItem(
        id="capital-1",
        document_id="capital",
        source="finewiki-zh",
        title="首都",
        score=1.0,
        snippet="中华人民共和国的首都是北京。",
    )


def test_generator_removes_bracketed_chat_protocol_markers() -> None:
    assert (
        EvidenceAnswerGenerator._clean_answer("[助手 1] 北京。[资料 1]\n[用户] 中国首都是哪里？")
        == "北京。[资料 1]"
    )


def test_generator_removes_bracketed_thinking_and_answer_markers() -> None:
    raw = "[思考] 用户问首都，资料说北京。\n[回答] 中国的首都是北京。[资料 1]"

    assert EvidenceAnswerGenerator._clean_answer(raw) == "中国的首都是北京。[资料 1]"


def test_generator_extracts_answer_after_echoed_evidence_block() -> None:
    raw = (
        "[资料 1] 标题：中国首都\n"
        "中国首都 > 古都列表 > 北京\n\n"
        "北京为五朝帝都，中華民國北洋政府和中華人民共和國的首都。\n\n"
        "Assistant: 根据资料，中国的首都是北京。[资料 1]"
    )

    assert EvidenceAnswerGenerator._clean_answer(raw) == "根据资料，中国的首都是北京。[资料 1]"


def test_generator_rejects_protocol_payloads_and_evidence_echo() -> None:
    assert EvidenceAnswerGenerator._clean_answer('{"status":"ok","evidence":[]}') == ""
    assert EvidenceAnswerGenerator._clean_answer("[资料 1] 标题：中国首都\n北京") == ""
    assert EvidenceAnswerGenerator._clean_answer("根据资料回答问题，只回答答案。") == ""


@pytest.mark.asyncio
async def test_generator_uses_evidence_and_truncates_rwkv_continuation() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url == "http://rwkv.test/v1/chat/completions"
        assert payload["contents"][0].find("中华人民共和国的首都是北京。") >= 0
        assert payload["temperature"] == 0.8
        assert payload["top_k"] == 50
        assert payload["top_p"] == 0.6
        assert payload["alpha_presence"] == 1.0
        assert payload["alpha_frequency"] == 0.1
        assert payload["alpha_decay"] == 0.99
        assert payload["stream"] is True
        return httpx.Response(
            200,
            content=(
                'data: {"choices":[{"delta":{"content":"Assistant: <think>推理</think>北京。[资料 1]\\n用户："}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"续写内容"}}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    generator = EvidenceAnswerGenerator(
        Settings(generation_base_url="http://rwkv.test/v1", generation_password="test-password"),
        transport=httpx.MockTransport(handler),
    )

    assert await generator.generate("中国的首都是哪个城市？", [source()]) == "北京。[资料 1]"


@pytest.mark.asyncio
async def test_generator_cleans_echoed_evidence_from_model_output() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                'data: {"choices":[{"delta":{"content":"[资料 1] 标题：中国首都\\n北京为中华人民共和国的首都。\\n\\nAssistant: 中国的首都是北京。[资料 1]"}}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    generator = EvidenceAnswerGenerator(
        Settings(generation_base_url="http://rwkv.test/v1", generation_password="test-password"),
        transport=httpx.MockTransport(handler),
    )

    assert await generator.generate("中国的首都是哪个城市？", [source()]) == "中国的首都是北京。[资料 1]"


@pytest.mark.asyncio
async def test_generator_skips_model_call_without_sources() -> None:
    generator = EvidenceAnswerGenerator(Settings())

    assert await generator.generate("没有资料怎么办？", []) == "未检索到可用于回答该问题的资料。"


@pytest.mark.asyncio
async def test_generator_skips_model_call_when_evidence_does_not_cover_question() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        pytest.fail("the generation model must not be called for unrelated evidence")

    generator = EvidenceAnswerGenerator(
        Settings(generation_password="test-password"),
        transport=httpx.MockTransport(handler),
    )

    answer = await generator.generate(
        "中国有多少个民族，分别是哪些？",
        [
            SourceItem(
                id="china-1",
                document_id="china",
                source="finewiki-zh",
                title="中国",
                score=1.0,
                snippet="中华人民共和国是位于东亚的国家。",
            )
        ],
    )

    assert answer == "根据检索到的资料，无法确定。"


@pytest.mark.asyncio
async def test_generator_skips_model_call_when_subject_anchor_is_absent() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        pytest.fail("the generation model must not be called for subject-mismatched evidence")

    generator = EvidenceAnswerGenerator(
        Settings(generation_password="test-password"),
        transport=httpx.MockTransport(handler),
    )

    answer = await generator.generate(
        "秦始皇有哪些丰功伟绩？",
        [
            SourceItem(
                id="henan-1",
                document_id="henan",
                source="finewiki-zh",
                title="河南省",
                score=1.0,
                snippet="河南省拥有龙门石窟、殷墟等历史文化遗产。",
            )
        ],
    )

    assert answer == "根据检索到的资料，无法确定。"


@pytest.mark.asyncio
async def test_generator_adds_a_citation_when_model_omits_one() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content='data: {"choices":[{"delta":{"content":"北京。"}}]}\n\ndata: [DONE]\n\n',
        )

    generator = EvidenceAnswerGenerator(
        Settings(generation_password="test-password"),
        transport=httpx.MockTransport(handler),
    )

    assert await generator.generate("中国的首都是哪个城市？", [source()]) == "北京。 [资料 1]"


@pytest.mark.asyncio
async def test_generator_removes_invalid_citation_and_adds_valid_one() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content='data: {"choices":[{"delta":{"content":"北京。[资料 2]"}}]}\n\ndata: [DONE]\n\n',
        )

    generator = EvidenceAnswerGenerator(
        Settings(generation_password="test-password"),
        transport=httpx.MockTransport(handler),
    )

    assert await generator.generate("中国的首都是哪个城市？", [source()]) == "北京。 [资料 1]"


@pytest.mark.asyncio
async def test_generator_reads_current_model_from_models_endpoint() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://models.test/v1/models"
        return httpx.Response(200, json={"data": [{"id": "rwkv-current"}]})

    generator = EvidenceAnswerGenerator(
        Settings(generation_models_url="http://models.test/v1/models"),
        transport=httpx.MockTransport(handler),
    )

    assert await generator.current_model() == "rwkv-current"


@pytest.mark.asyncio
async def test_generator_allows_model_list_failure() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    generator = EvidenceAnswerGenerator(
        Settings(generation_models_url="http://models.test/v1/models"),
        transport=httpx.MockTransport(handler),
    )

    assert await generator.current_model() is None
