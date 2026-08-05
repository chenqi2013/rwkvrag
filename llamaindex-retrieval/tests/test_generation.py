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
                'data: {"choices":[{"delta":{"content":"Assistant: <think>推理</think>北京。\\n用户："}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"续写内容"}}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    generator = EvidenceAnswerGenerator(
        Settings(generation_base_url="http://rwkv.test/v1", generation_password="test-password"),
        transport=httpx.MockTransport(handler),
    )

    assert await generator.generate("中国的首都是哪个城市？", [source()]) == "北京。"


@pytest.mark.asyncio
async def test_generator_skips_model_call_without_sources() -> None:
    generator = EvidenceAnswerGenerator(Settings())

    assert await generator.generate("没有资料怎么办？", []) == "未检索到可用于回答该问题的资料。"
