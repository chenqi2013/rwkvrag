import asyncio
import base64
from hashlib import sha256
import json

import httpx
import pytest

from llamaindex_retrieval.rwkvos_batch import (
    RwkvosBatchClient, inspect_batch_envelope, render_batch_prompt,
)


MODEL = "rwkv7-g1j-2.9b-20260831-ctx16384"
MESSAGES = [{"role": "user", "content": "完整原文 \nUser: 引文 ✿  "}]


def response(contents, **changes):
    return {"object": "chat.completion", "model": MODEL,
            "choices": [{"index": i, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"} for i, text in enumerate(contents)], **changes}


def client(handler, **kwargs):
    return RwkvosBatchClient(base_url="https://batch.test/v1", model=MODEL,
                             transport=httpx.MockTransport(handler), **kwargs)


def assert_bytes(entry, side):
    raw = base64.b64decode(entry[f"{side}_body_base64"], validate=True)
    assert sha256(raw).hexdigest() == entry[f"{side}_body_sha256"]
    return raw


def test_complete_history_plain_template_preserves_every_content_character():
    messages = [{"role": "system", "content": " 规则 "}, *MESSAGES,
                {"role": "assistant", "content": "旧回答"}, {"role": "user", "content": "更正"}]
    prompt, prefill = render_batch_prompt(messages, "<think></think")
    assert prompt == ("System:  规则 \n\nUser: 完整原文 \nUser: 引文 ✿  \n\n"
                      "Assistant: 旧回答\n\nUser: 更正\n\nAssistant: <think></think>")
    assert prefill == "<think></think>"
    assert messages[1]["content"] == MESSAGES[0]["content"]


@pytest.mark.parametrize("raw,prefill,expected", [
    ("  答案 \n", "<think></think>", "  答案 \n"),
    (">答案", "<think></think>", ">答案"),
    ("思考</think>\n答案", "<think>", "\n答案"),
    ("思考未闭合", "<think>", None),
    ("<think>嵌套</think>答案", "<think>", None),
    ("</think>答案", "<think></think>", None),
    ("<think>推理</think>答案", "<think></think>", None),
    ("  ", "<think></think>", None),
])
def test_envelope_uses_only_actual_output_without_rewriting(raw, prefill, expected):
    envelope = inspect_batch_envelope(raw, prefill)
    assert envelope["raw_text_modified"] is False
    assert envelope["valid"] is (expected is not None)
    span = envelope["answer_span"]
    assert (raw[span["start"]:span["end"]] if span else None) == expected


async def test_real_batch_reordered_indices_original_bytes_unknown_termination_and_private_auth():
    seen, recorded = [], []
    async def recorder(event, row):
        recorded.append((event, row))
    def handler(request):
        assert recorded[0][0] == "batch_started"
        assert request.headers["CF-Access-Client-Secret"] == "credential-secret"
        seen.append(request)
        data = response(["第一答", "  第二答\n"])
        data["choices"].reverse()
        return httpx.Response(200, json=data)
    async with client(handler, state_id="state-explicit", recorder=recorder,
                      headers={"CF-Access-Client-Secret": "credential-secret"}) as model:
        answers = await asyncio.gather(model.complete(MESSAGES, stage="planner", temperature=0),
                                       model.complete(MESSAGES, stage="writer", temperature=0))
    assert [r.text for r in answers] == ["第一答", "  第二答\n"]
    assert len(seen) == 1
    payload = json.loads(seen[0].content)
    assert len(payload["contents"]) == 2
    assert payload == {"model": MODEL, "contents": payload["contents"], "max_tokens": 2048,
                       "temperature": .001, "top_k": 20, "top_p": 0,
                       "alpha_presence": 0, "alpha_frequency": 0, "alpha_decay": .99,
                       "chunk_size": 8, "stream": False, "state_id": "state-explicit"}
    assert [e for e, _ in recorded] == ["batch_started", "batch_completed"]
    for index, result in enumerate(answers):
        t = result.trace
        assert result.status == "completed" and result.finish_reason == "stop"
        assert t["batch_index"] == index and t["state_id"] == "state-explicit"
        assert t["termination_verified"] is False and t["termination"] == "unknown"
        assert t["provider_finish_reason"] == "stop" and t["usage"] is None
        assert t["budget"]["fits"] is None and t["budget"]["single_bos_verified"] is None
        assert t["requested_temperature"] == 0
        assert t["parameters"]["temperature"] == .001
        assert t["messages"] == MESSAGES
        assert t["raw_text_sha256"] == sha256(result.text.encode()).hexdigest()
        assert t["elapsed_ms"] >= t["queue_ms"] >= 0
        assert_bytes(t["http"][0], "request")
        assert_bytes(t["http"][0], "response")
    assert "credential-secret" not in json.dumps([r.trace for r in answers])
    assert "credential-secret" not in json.dumps(recorded)


async def test_distinct_parameters_never_share_http_and_batch_capacity_is_real():
    requests, active, maximum = [], 0, 0
    async def handler(request):
        nonlocal active, maximum
        body = json.loads(request.content)
        n = len(body["contents"])
        active += n
        maximum = max(maximum, active)
        requests.append(body)
        await asyncio.sleep(.01)
        active -= n
        return httpx.Response(200, json=response([str(body["max_tokens"])] * n))
    async with client(handler, max_concurrency=3, batch_size=2, batch_wait_ms=1) as model:
        results = await asyncio.gather(*[model.complete(MESSAGES, max_tokens=100 + i % 2)
                                         for i in range(9)])
    assert [r.text for r in results] == [str(100 + i % 2) for i in range(9)]
    assert maximum <= 3
    assert all(len(r["contents"]) <= 2 for r in requests)
    assert sum(len(r["contents"]) for r in requests) == 9


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "negative", "out_of_range", "boolean",
                                        "unknown_finish", "non_string_finish", "surrogate", "missing_content", "wrong_model", "error"])
async def test_malformed_batch_never_silently_reassigns_or_retries(mutation):
    calls = []
    def handler(request):
        calls.append(request)
        data = response(["A", "B"])
        if mutation == "missing":
            data["choices"].pop()
        elif mutation == "duplicate":
            data["choices"][1]["index"] = 0
        elif mutation in {"negative", "out_of_range", "boolean"}:
            data["choices"][1]["index"] = {"negative": -1, "out_of_range": 2, "boolean": True}[mutation]
        elif mutation == "unknown_finish":
            data["choices"][1]["finish_reason"] = "unknown"
        elif mutation == "non_string_finish":
            data["choices"][1]["finish_reason"] = ["stop"]
        elif mutation == "surrogate":
            data["choices"][1]["message"]["content"] = "\ud800"
            return httpx.Response(200, content=json.dumps(data).encode())
        elif mutation == "missing_content":
            del data["choices"][1]["message"]["content"]
        elif mutation == "wrong_model":
            data["model"] = "other-model"
        else:
            data = {"error": "provider error"}
        return httpx.Response(200, json=data)
    async with client(handler) as model:
        results = await asyncio.gather(model.complete(MESSAGES), model.complete(MESSAGES))
    assert len(calls) == 1
    assert all(r.status == "invalid_response" for r in results)
    assert all(r.raw_text is None for r in results)
    assert all(assert_bytes(r.trace["http"][0], "response") for r in results)


@pytest.mark.parametrize("body", [b'{"model":"a","model":"b"}', b'{"bad":NaN}', b'\xff', b'[]'])
async def test_invalid_json_is_retained_without_retry(body):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, content=body)
    async with client(handler) as model:
        result = await model.complete(MESSAGES)
    assert result.status == "invalid_response" and len(requests) == 1
    assert assert_bytes(result.trace["http"][0], "response") == body


async def test_explicit_length_preserves_whole_raw_but_does_not_claim_success():
    def handler(request):
        data = response(["看似完整答案"])
        data["choices"][0]["finish_reason"] = "length"
        return httpx.Response(200, json=data)
    async with client(handler) as model:
        result = await model.complete(MESSAGES)
    assert result.status == "length" and result.text == "看似完整答案"
    assert result.trace["provider_finish_reason"] == "length"
    assert result.trace["termination_verified"] is False


@pytest.mark.parametrize("kind,expected", [("http", "http_error"), ("timeout", "timeout"),
                                            ("network", "transport_error")])
async def test_http_network_and_bounded_timeout_fail_without_retry(kind, expected):
    requests = []
    async def handler(request):
        requests.append(request)
        if kind == "timeout":
            await asyncio.sleep(.2)
        if kind == "network":
            raise httpx.ConnectError("secret URL must not become error text")
        return httpx.Response(503, content=b'{"error":"unavailable"}')
    async with client(handler, timeout_seconds=.03, batch_wait_ms=0) as model:
        result = await model.complete(MESSAGES)
    assert result.status == expected and len(requests) == 1
    assert "secret URL" not in json.dumps(result.trace)
    if kind == "http":
        assert result.trace["http"][0]["http_status"] == 503
        assert assert_bytes(result.trace["http"][0], "response") == b'{"error":"unavailable"}'


async def test_partial_body_failure_retains_observed_bytes():
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"choices":'
            raise httpx.ReadError("broken")
    def handler(request):
        return httpx.Response(200, stream=BrokenStream())
    async with client(handler) as model:
        result = await model.complete(MESSAGES)
    assert result.status == "transport_error"
    entry = result.trace["http"][0]
    assert entry["response_body_complete"] is False
    assert assert_bytes(entry, "response") == b'{"choices":'


@pytest.mark.parametrize("event", ["batch_started", "batch_completed"])
async def test_recorder_failure_prevents_unrecorded_success(event):
    requests = []
    def recorder(name, row):
        if name == event:
            raise OSError("credential must not be included")
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=response(["A"]))
    async with client(handler, recorder=recorder) as model:
        result = await model.complete(MESSAGES)
    assert result.status == "transport_error"
    assert len(requests) == (0 if event == "batch_started" else 1)
    assert "credential must" not in json.dumps(result.trace)
    if event == "batch_completed":
        assert result.raw_text == "A" and result.finish_reason == "stop"


async def test_pending_cancellation_never_sends_and_dispatched_cancellation_keeps_peers():
    entered, release = asyncio.Event(), asyncio.Event()
    seen = []
    async def handler(request):
        payload = json.loads(request.content)
        seen.append(payload)
        entered.set()
        await release.wait()
        return httpx.Response(200, json=response(["A", "B"]))
    async with client(handler, batch_wait_ms=20, max_concurrency=2) as model:
        pending_trace = {}
        pending = asyncio.create_task(model.complete(MESSAGES, trace=pending_trace))
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        first_trace = {}
        first = asyncio.create_task(model.complete(MESSAGES, trace=first_trace))
        peer = asyncio.create_task(model.complete(MESSAGES))
        await entered.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert model._active == 2
        assert not peer.done()
        release.set()
        result = await peer
    assert len(seen) == 1 and len(seen[0]["contents"]) == 2
    assert pending_trace["completion_attempted"] is False
    assert pending_trace["cancellation_effect"] == "removed_before_dispatch"
    assert first_trace["status"] == "cancelled" and first_trace["raw_text"] == "A"
    assert first_trace["cancellation_effect"] == "shared_batch_continues"
    assert first_trace["batch_item_status"] == "completed"
    assert result.text == "B" and result.status == "completed"


async def test_close_fails_pending_and_active_calls_and_leaves_no_background_workers():
    entered = asyncio.Event()
    async def handler(request):
        entered.set()
        await asyncio.Event().wait()
    model = client(handler, max_concurrency=1, batch_size=1, batch_wait_ms=0)
    active = asyncio.create_task(model.complete(MESSAGES))
    await entered.wait()
    pending = asyncio.create_task(model.complete(MESSAGES))
    await asyncio.sleep(0)
    await model.aclose()
    results = await asyncio.gather(active, pending)
    assert [r.status for r in results] == ["transport_error", "transport_error"]
    assert results[0].trace["completion_attempted"] is True
    assert results[1].trace["completion_attempted"] is False
    assert not model._workers and model._active == 0
    assert (await model.complete(MESSAGES)).status == "invalid_request"


@pytest.mark.parametrize("kwargs", [{"seed": 1}, {"temperature": -1}, {"temperature": 1001}, {"top_p": 2},
                                    {"max_tokens": True}, {"top_k": -1},
                                    {"assistant_prefill": "arbitrary"}])
async def test_unsupported_request_configuration_does_not_send(kwargs):
    def handler(request):
        pytest.fail("invalid request must not be sent")
    async with client(handler) as model:
        result = await model.complete(MESSAGES, **kwargs)
    assert result.status == "invalid_request" and not result.trace["http"]


@pytest.mark.parametrize("kwargs", [{"base_url": "https://user:secret@host/v1"},
                                    {"base_url": "https://host/v1?key=secret"},
                                    {"max_concurrency": 0}, {"batch_size": True},
                                    {"headers": {"Host": "other.test"}}, {"state_id": ""}])
def test_invalid_client_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RwkvosBatchClient(**({"base_url": "https://host/v1", "model": MODEL} | kwargs))


@pytest.mark.parametrize("stops", [None, [], [0], [0, 261, 24281]])
async def test_explicit_stop_tokens_preserve_empty_and_zero_while_default_is_omitted(stops):
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response(["原样答案"]))
    async with client(handler, stop_tokens=stops) as model:
        result = await model.complete(MESSAGES)
    assert result.status == "completed"
    if stops is None:
        assert "stop_tokens" not in requests[0]
        assert "stop_tokens" not in result.trace["parameters"]
    else:
        assert requests[0]["stop_tokens"] == stops
        assert result.trace["parameters"]["stop_tokens"] == stops
    assert result.trace["stop_tokens_requested"] == stops
    assert result.trace["provider_default_stop_tokens"] == {
        "source": "upstream_reported", "tokens": [0, 261, 24281], "deployment_verified": False}


@pytest.mark.parametrize("stops", [[True], [-1], [0, 0], "0", [1.0]])
def test_invalid_stop_tokens_rejected_without_network(stops):
    with pytest.raises(ValueError):
        RwkvosBatchClient(base_url="https://host/v1", model=MODEL, stop_tokens=stops)


async def test_stop_configuration_is_copied_and_distinct_parameters_cannot_coalesce():
    seen = []
    stops = [0]
    def handler(request):
        payload = json.loads(request.content)
        seen.append(payload)
        return httpx.Response(200, json=response(["原答"] * len(payload["contents"])))
    async with client(handler, stop_tokens=stops, batch_wait_ms=20) as model:
        stops.append(261)
        first = asyncio.create_task(model.complete(MESSAGES))
        await asyncio.sleep(0)
        # Explicit reconfiguration between calls remains isolated in the queued payloads.
        model.stop_tokens = []
        second = asyncio.create_task(model.complete(MESSAGES))
        await asyncio.gather(first, second)
    assert [row["stop_tokens"] for row in seen] == [[0], []]
    assert all(len(row["contents"]) == 1 for row in seen)
