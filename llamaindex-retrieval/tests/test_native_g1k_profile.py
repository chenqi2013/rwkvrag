"""G1K uses explicit native settings; no inference or quality claims in mocks."""
import json

import httpx
import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.model_client import model_client_options
from llamaindex_retrieval.structured_native import StructuredNativeRWKVClient

MODEL = "rwkv7-g1k-7.2b-20260930-ctx25600"
RAW = ' {"value":null,"status":"unknown","evidence_ids":[]}\n'


def options():
    settings = Settings(
        _env_file=None,
        rag_pipeline="rwkv", native_transport="native",
        native_base_url="http://native.test/v1", native_model=MODEL,
        native_context_window_tokens=25600, native_require_model_identity=True,
        native_completion_protocol="g1j_plain", native_state_routing=None,
        native_planner_prefill="<think></think", native_resolver_prefill="<think></think",
        native_writer_prefill="<think></think",
    )
    return model_client_options(settings)


@pytest.mark.parametrize("count,server_limit,output,status", [
    (25568, 25600, 32, "completed"),
    (25569, 25600, 32, "budget_exceeded"),
    (16000, 16384, 512, "budget_exceeded"),
])
async def test_g1k_profile_reserves_output_and_obeys_lower_server_limit(count, server_limit, output, status):
    calls = []
    tokens = [1] * count

    def handle(request):
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        assert body["model"] == MODEL
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": count, "tokens": tokens, "max_model_len": server_limit})
        return httpx.Response(200, json={"model": MODEL, "choices": [{
            "text": RAW, "finish_reason": "stop", "prompt_token_ids": tokens, "token_ids": [2],
        }], "usage": {"prompt_tokens": count, "completion_tokens": 1}})

    async with StructuredNativeRWKVClient(**options(), transport=httpx.MockTransport(handle)) as client:
        result = await client.complete([{"role": "user", "content": "结构测试"}],
                                       assistant_prefill="<think></think", max_tokens=output)
    assert result.status == status
    assert len(calls) == (2 if status == "completed" else 1)
    assert result.trace["budget"]["effective_context_window"] == min(25600, server_limit)
    if status == "completed":
        assert result.raw_text == RAW
        assert "vllm_xargs" not in calls[-1][1]


@pytest.mark.parametrize("order", [
    ["value", "status", "evidence_ids"],
    ["evidence_ids", "status", "value"],
])
async def test_structured_native_preserves_caller_property_order_and_raw_answer(order):
    schema = {"type": "object", "properties": {key: {"type": "string"} for key in order}}
    seen = []

    def handle(request):
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 1, "tokens": [1], "max_model_len": 25600})
        seen.append(body)
        return httpx.Response(200, json={"model": MODEL, "choices": [{
            "text": RAW, "finish_reason": "stop", "prompt_token_ids": [1], "token_ids": [2],
        }], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    async with StructuredNativeRWKVClient(**options(), transport=httpx.MockTransport(handle)) as client:
        result = await client.complete([{"role": "user", "content": "结构测试"}],
                                       assistant_prefill="<think></think", structured_schema=schema)
    assert result.status == "completed"
    assert list(seen[0]["structured_outputs"]["json"]["properties"]) == order
    assert list(result.trace["parameters"]["structured_outputs"]["json"]["properties"]) == order
    assert list(schema["properties"]) == order
    # Even a schema-invalid fake response is retained, not repaired by transport.
    assert result.raw_text == result.text == RAW
