import asyncio
import base64
from hashlib import sha256
import json

import httpx
import pytest

from llamaindex_retrieval.native_rwkv import (
    NativeRWKVClient,
    inspect_envelope,
    render_native_prompt,
)


MESSAGES = [{"role": "user", "content": "  原始问题\n保留尾部空白  "}]
RAW = ">推理内容</think>\n  原始答案 [S7]  \n"


def token_response(**changes):
    return {"count": 4, "tokens": [0, 12, 34, 56], "max_model_len": 16384, **changes}


def completion_response(text=RAW, finish_reason="stop", **changes):
    return {"choices": [{"text": text, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 9, "total_tokens": 13}, **changes}


def client(handler, **kwargs):
    return NativeRWKVClient(
        base_url="http://rwkv.test/v1", model="rwkv-test",
        transport=httpx.MockTransport(handler), **kwargs,
    )


def assert_wire(entry, request=None, response=None):
    for side, expected in (("request", request), ("response", response)):
        key = f"{side}_body_base64"
        if key in entry:
            data = base64.b64decode(entry[key], validate=True)
            assert sha256(data).hexdigest() == entry[f"{side}_body_sha256"]
            if expected is not None:
                assert data == expected


@pytest.mark.parametrize("configured,transmitted", [
    ("<think", "<think"), ("<think>", "<think"),
    ("<think></think", "<think></think"), ("<think></think>", "<think></think"),
])
def test_native_template_preserves_real_history_and_prefill(configured, transmitted):
    messages = [
        {"role": "system", "content": "  原规则\n"},
        {"role": "user", "content": "旧提问"},
        {"role": "assistant", "content": "旧回答\n"},
        {"role": "user", "content": "  更正：新版本\nUser: 引文  "},
    ]
    prompt = render_native_prompt(messages, configured)
    assert prompt.prompt == (
        "System✿  原规则\n✿\nUser✿旧提问✿\nBot✿旧回答\n✿\n"
        "User✿  更正：新版本\nUser: 引文  ✿\nBot✿" + transmitted
    )
    assert prompt.prefill == transmitted
    assert "<|endoftext|>" not in prompt.prompt
    assert messages[-1]["content"] == "  更正：新版本\nUser: 引文  "


def test_literal_delimiter_encoding_has_no_collision_or_backslash_decoding():
    original = "✿ literal \\u273f __RWKVRAG_LITERAL_U273F_0__"
    result = render_native_prompt([{"role": "user", "content": original}])
    assert result.delimiter_escape == "__RWKVRAG_LITERAL_U273F_1__"
    encoded = result.prompt.removeprefix("User✿").removesuffix("✿\nBot✿<think")
    assert encoded.replace(result.delimiter_escape, "✿") == original
    raw = f">想</think>{result.delimiter_escape} \\u273f"
    bounds = inspect_envelope(raw)
    assert bounds is not None
    assert raw[bounds[0]:bounds[1]] == f"{result.delimiter_escape} \\u273f"


@pytest.mark.parametrize("raw,prefill,expected", [
    (RAW, "<think", "\n  原始答案 [S7]  \n"),
    (">\n{}  ", "<think></think", "\n{}  "),
    ("{}", "<think></think", None),
    (">没有结束", "<think", None),
    (">想<think>嵌套</think>答案", "<think", None),
    (">想</think> \n ", "<think", None),
    (">想</think>原生✿分隔", "<think", None),
    (">想</think><think>第二段", "<think", None),
    (">想</think>答案", "<think>", None),
])
def test_envelope_returns_exact_bounds_without_cleanup(raw, prefill, expected):
    bounds = inspect_envelope(raw, prefill)
    assert (raw[bounds[0]:bounds[1]] if bounds else None) == expected


async def test_real_entrypoint_has_exact_payload_trace_and_immutable_text():
    requests = []
    def handler(request):
        requests.append(request)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json=token_response())
        return httpx.Response(200, json=completion_response())
    async with client(handler, api_key="test-secret") as model:
        result = await model.complete(MESSAGES, evidence_ids=["node-full-hash"], stage="resolver")
    assert result.status == "completed"
    assert result.text == result.raw_text == RAW
    assert result.finish_reason == "stop"
    assert len(requests) == 2
    tokenize, completion = (json.loads(request.content) for request in requests)
    assert tokenize == {"model": "rwkv-test", "prompt": "User✿  原始问题\n保留尾部空白  ✿\nBot✿<think",
                        "add_special_tokens": True}
    assert completion == {**tokenize, "max_tokens": 2048, "temperature": 0.0,
                          "top_p": 1.0, "top_k": 0, "presence_penalty": 0.0,
                          "frequency_penalty": 0.0, "stream": False,
                          "stop": ["✿"], "stop_token_ids": [0], "ignore_eos": False}
    trace = result.trace
    assert trace["evidence_ids"] == ["node-full-hash"]
    assert trace["prompt_sha256"] == sha256(tokenize["prompt"].encode()).hexdigest()
    assert trace["raw_text_sha256"] == sha256(RAW.encode()).hexdigest()
    assert trace["envelope"]["raw_text_modified"] is False
    assert trace["messages"] == MESSAGES
    assert "test-secret" not in json.dumps(trace)
    for entry, request in zip(trace["http"], requests, strict=True):
        assert_wire(entry, request=request.content)
        assert entry["response_body_complete"] is True
    assert trace["elapsed_ms"] >= trace["active_ms"] >= 0
    assert trace["queue_ms"] >= 0


@pytest.mark.parametrize("settings_limit,server_limit,output,expected", [
    (10, 30, 6, "completed"), (30, 10, 7, "budget_exceeded"),
    (10, 30, 7, "budget_exceeded"),
])
async def test_exact_whole_prompt_budget_with_server_and_config_limits(
    settings_limit, server_limit, output, expected,
):
    requests = []
    def handler(request):
        requests.append(request)
        data = (token_response(max_model_len=server_limit)
                if request.url.path == "/tokenize" else completion_response())
        return httpx.Response(200, json=data)
    async with client(handler, context_window_tokens=settings_limit) as model:
        result = await model.complete(MESSAGES, max_tokens=output)
    assert result.status == expected
    assert len(requests) == (2 if expected == "completed" else 1)
    assert result.trace["completion_attempted"] == (expected == "completed")
    assert result.trace["budget"]["effective_context_window"] == min(settings_limit, server_limit)


@pytest.mark.parametrize("changes", [
    {"count": True}, {"count": 3}, {"tokens": [0, 0, 2, 3]},
    {"tokens": [1, 2, 3, 4]}, {"tokens": [0, 2, 0, 4]},
    {"tokens": [0, True, 2, 3]}, {"tokens": None}, {"max_model_len": None},
])
async def test_invalid_tokenizer_never_calls_generation(changes):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=token_response(**changes))
    async with client(handler) as model:
        result = await model.complete(MESSAGES)
    assert result.status == "invalid_response"
    assert len(requests) == 1
    assert result.raw_text is None
    assert_wire(result.trace["http"][0])


@pytest.mark.parametrize("payload,expected,raw", [
    ({"choices": []}, "invalid_response", None),
    (completion_response(text=None), "invalid_response", None),
    (completion_response(finish_reason="length"), "length", RAW),
    (completion_response(finish_reason="length", text=">未完成推理"), "length", ">未完成推理"),
    (completion_response(finish_reason=None), "invalid_response", RAW),
    (completion_response(text="{}"), "invalid_response", "{}"),
    (completion_response(text=""), "invalid_response", ""),
    (completion_response(usage={"prompt_tokens": 5}), "invalid_response", RAW),
])
async def test_completion_failures_never_masquerade_as_success_or_destroy_raw(payload, expected, raw):
    def handler(request):
        return httpx.Response(200, json=token_response() if request.url.path == "/tokenize" else payload)
    async with client(handler) as model:
        result = await model.complete(MESSAGES)
    assert result.status == expected
    assert result.raw_text == raw
    assert result.text == (raw if raw is not None else "")
    assert len(result.trace["http"]) == 2
    assert_wire(result.trace["http"][-1])
    assert result.trace["http"][-1]["http_status"] == 200


@pytest.mark.parametrize("stage,status,body,expected", [
    ("tokenize", 503, b"unavailable", "http_error"),
    ("completion", 502, b"broken gateway", "http_error"),
    ("completion", 200, b"not JSON", "invalid_response"),
    ("completion", 200, b"\xff", "invalid_response"),
    ("completion", 200, b"[]", "invalid_response"),
])
async def test_http_and_malformed_responses_retain_exact_body_without_retry(stage, status, body, expected):
    requests = []
    def handler(request):
        requests.append(request)
        current = "tokenize" if request.url.path == "/tokenize" else "completion"
        return (httpx.Response(status, content=body) if current == stage
                else httpx.Response(200, json=token_response()))
    async with client(handler) as model:
        result = await model.complete(MESSAGES)
    assert result.status == expected
    assert len(requests) == (1 if stage == "tokenize" else 2)
    assert_wire(result.trace["http"][-1], response=body)


@pytest.mark.parametrize("error,expected", [(httpx.ConnectError, "transport_error"),
                                            (httpx.ReadTimeout, "timeout")])
async def test_unknown_transport_outcome_is_recorded_once(error, expected):
    requests = []
    def handler(request):
        requests.append(request)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json=token_response())
        raise error("simulated", request=request)
    async with client(handler) as model:
        result = await model.complete(MESSAGES)
    assert result.status == expected
    assert len(requests) == 2
    assert result.trace["completion_attempted"] is True
    assert result.trace["http"][-1]["error_type"] == error.__name__
    assert_wire(result.trace["http"][-1])


class InterruptedBody(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'{"choices":['
        raise httpx.ReadTimeout("body interrupted")


async def test_partial_200_body_survives_read_timeout():
    def handler(request):
        return (httpx.Response(200, json=token_response()) if request.url.path == "/tokenize"
                else httpx.Response(200, stream=InterruptedBody()))
    async with client(handler) as model:
        result = await model.complete(MESSAGES)
    assert result.status == "timeout"
    entry = result.trace["http"][-1]
    assert entry["http_status"] == 200
    assert entry["response_body_complete"] is False
    assert_wire(entry, response=b'{"choices":[')


async def test_overall_deadline_and_cancellation_preserve_attempts_and_release_slot():
    entered = asyncio.Event()
    blocker = asyncio.Event()
    requests = []
    async def handler(request):
        requests.append(request)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json=token_response())
        entered.set()
        await blocker.wait()
        return httpx.Response(200, json=completion_response())
    async with client(handler, timeout_seconds=0.02, max_concurrency=1) as model:
        result = await model.complete(MESSAGES)
        assert result.status == "timeout"
        assert len(result.trace["http"]) == 2
        trace = {}
        entered.clear()
        task = asyncio.create_task(model.complete(MESSAGES, trace=trace))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert trace["status"] == "cancelled"
        assert trace["completion_attempted"] is True
        assert trace["http"][-1]["error_type"] == "CancelledError"
        blocker.set()
        assert (await model.complete(MESSAGES)).status == "completed"
    assert len(requests) == 6


async def test_shared_client_concurrency_is_bounded_across_stages_and_requests():
    active = maximum = 0
    requests = []
    both_entered = asyncio.Event()
    release = asyncio.Event()
    async def handler(request):
        nonlocal active, maximum
        requests.append(request)
        active += 1
        maximum = max(maximum, active)
        if len(requests) >= 2:
            both_entered.set()
        await release.wait()
        await asyncio.sleep(0)
        active -= 1
        data = token_response() if request.url.path == "/tokenize" else completion_response()
        return httpx.Response(200, json=data)
    async with client(handler, max_concurrency=2) as model:
        tasks = [asyncio.create_task(model.complete(MESSAGES, stage=f"stage-{i}")) for i in range(6)]
        await both_entered.wait()
        assert len(requests) == 2
        release.set()
        results = await asyncio.gather(*tasks)
    assert maximum == 2
    assert len(requests) == 12
    assert all(result.status == "completed" for result in results)
    assert len({result.trace["call_id"] for result in results}) == 6
    assert all(len(result.trace["http"]) == 2 for result in results)


@pytest.mark.parametrize("kwargs", [{"max_tokens": 0}, {"max_tokens": True},
                                   {"temperature": float("nan")}, {"top_p": 0},
                                   {"assistant_prefill": "arbitrary answer"}])
async def test_invalid_request_is_rejected_without_network(kwargs):
    def handler(request):
        pytest.fail("invalid request must not send HTTP")
    async with client(handler) as model:
        result = await model.complete(MESSAGES, **kwargs)
    assert result.status == "invalid_request"
    assert result.trace["http"] == []


async def test_closed_client_returns_diagnostic_trace_without_losing_request():
    def handler(request):
        pytest.fail("closed client must not send HTTP")
    model = client(handler)
    await model.aclose()
    result = await model.complete(MESSAGES)
    assert result.status == "transport_error"
    assert result.trace["http"][0]["error_type"] == "RuntimeError"
    assert result.trace["completion_attempted"] is False
    assert_wire(result.trace["http"][0])
