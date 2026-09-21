import json
import httpx
import pytest

from llamaindex_retrieval.native_rwkv import NativeRWKVClient
from llamaindex_retrieval.model_client import model_answer_bounds


@pytest.mark.parametrize("finish", ["stop", "length"])
async def test_plain_transport_matches_frozen_envelope_and_preserves_raw(finish):
    requests = []
    raw = "  1. 甲[资料 1]\n2. 甲[资料 1]\n"
    def handler(request):
        body = json.loads(request.content); requests.append(body)
        assert body["prompt"] == "User: 原文\n不改变\n\nAssistant: <think></think>\n"
        assert body["add_special_tokens"] is False
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count":3,"tokens":[4,5,6],"max_model_len":16384})
        assert body["top_k"] == 1 and body["ignore_eos"] is False
        return httpx.Response(200, json={"choices":[{"text":raw,"finish_reason":finish,
            "prompt_token_ids":[4,5,6],"token_ids":[7,8]}],"usage":{"prompt_tokens":3}})
    async with NativeRWKVClient(base_url="http://test/v1",model="test",prompt_protocol="g1j_plain",
                               transport=httpx.MockTransport(handler)) as client:
        result = await client.complete([{"role":"user","content":"原文\n不改变"}],
                                       assistant_prefill="<think></think",top_k=1,temperature=1,seed=11)
    assert result.status == ("completed" if finish == "stop" else "length")
    assert result.raw_text == raw
    assert model_answer_bounds(raw,result.trace) == (0,len(raw))
    assert result.trace["input_token_ids"] == [4,5,6]
    assert len(requests) == 2


async def test_plain_transport_rejects_input_token_mismatch_without_repair():
    raw="保留原输出"
    def handler(request):
        if request.url.path=="/tokenize":
            return httpx.Response(200,json={"count":2,"tokens":[4,5],"max_model_len":16384})
        return httpx.Response(200,json={"choices":[{"text":raw,"finish_reason":"stop","prompt_token_ids":[4,6]}],
                                      "usage":{"prompt_tokens":2}})
    async with NativeRWKVClient(base_url="http://test/v1",model="test",prompt_protocol="g1j_plain",
                               transport=httpx.MockTransport(handler)) as client:
        result=await client.complete([{"role":"user","content":"问题"}],assistant_prefill="<think></think")
    assert result.status=="invalid_response"
    assert result.raw_text==raw
    assert model_answer_bounds(raw,result.trace) is None


async def test_plain_transport_does_not_silently_convert_open_thinking_protocol():
    def handler(request):
        pytest.fail("invalid template must not reach HTTP")
    async with NativeRWKVClient(base_url="http://test/v1",model="test",prompt_protocol="g1j_plain",
                               transport=httpx.MockTransport(handler)) as client:
        result=await client.complete([{"role":"user","content":"问题"}],assistant_prefill="<think")
    assert result.status=="invalid_request"
