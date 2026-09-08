"""Actual HTTP boundary mocks for optional server input counts; no live network."""
import asyncio
import base64
from hashlib import sha256
import json
from time import sleep

import httpx
import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.model_client import model_client_options
from llamaindex_retrieval.rwkvos_batch import RwkvosBatchClient, render_batch_prompt

MODEL = "rwkv7-g1j-2.9b-20260831-ctx16384"
MESSAGES = [{"role": "user", "content": "完整证据\n\n中文😀 ✿ User: 原文  "},
            {"role": "assistant", "content": "历史答案"}, {"role": "user", "content": "更正范围"}]


def answer(payload, text="结果[资料 1]"):
    return {"model": MODEL, "object": "chat.completion", "choices": [
        {"index": i, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        for i in range(len(payload["contents"]))]}


def client(handler, **kwargs):
    return RwkvosBatchClient(base_url="https://model.test/v1", model=MODEL,
                             transport=httpx.MockTransport(handler), **kwargs)


def raw(entry, side):
    body = base64.b64decode(entry[side + "_body_base64"], validate=True)
    assert sha256(body).hexdigest() == entry[side + "_body_sha256"]
    return body


async def test_default_disabled_has_no_new_count_request_or_application_gate():
    seen = []
    def handler(req):
        seen.append(req.url.path)
        return httpx.Response(200, json=answer(json.loads(req.content)))
    async with client(handler, context_window_tokens=1) as model:
        result = await model.complete(MESSAGES)
    assert result.status == "completed"
    assert seen == ["/v1/batch/completions"]
    assert "token_count_id" not in result.trace
    assert result.trace["budget"]["input_tokens"] is None


@pytest.mark.parametrize("mode", ["complete", "continuation"])
async def test_exact_final_prompt_count_raw_receipts_and_no_inferred_server_window(mode):
    seen, events = [], []
    async def recorder(event, record):
        events.append((event, record))
    def handler(req):
        seen.append(req)
        assert req.headers["CF-Access-Client-Secret"] == "private-header-value"
        if req.url.path.endswith("/tokens/count"):
            assert events[0][0] == "token_count_started"
            return httpx.Response(200, content=b'{"tokens":32624}')
        return httpx.Response(200, json=answer(json.loads(req.content), ">原始答案" if mode == "continuation" else "原始答案"))
    async with client(handler, count_input_tokens=True, prefill_mode=mode, recorder=recorder,
                      headers={"CF-Access-Client-Secret": "private-header-value"}) as model:
        result = await model.complete(MESSAGES, assistant_prefill="<think></think", stage="writer", evidence_ids=["src1"])
    expected = render_batch_prompt(MESSAGES, "<think></think", mode)[0]
    assert json.loads(seen[0].content) == {"text": expected}
    assert json.loads(seen[1].content)["contents"] == [expected]
    assert [e for e, _ in events] == ["token_count_started", "token_count_completed", "batch_started", "batch_completed"]
    count, generation = result.trace["http"]
    assert "batch_id" not in count and "count_id" not in generation
    assert count["count_id"] == result.trace["token_count_id"]
    assert count["call_id"] == result.trace["call_id"] and count["model_stage"] == "writer"
    assert count["evidence_ids"] == ["src1"]
    assert count["prompt_sha256"] == sha256(expected.encode()).hexdigest()
    assert raw(count, "request") == seen[0].content
    assert raw(count, "response") == b'{"tokens":32624}'
    assert count["http_attempted"] and count["response_body_complete"]
    budget = result.trace["budget"]
    assert budget["input_tokens"] == 32624 and budget["counted_prompt_sha256"] == count["prompt_sha256"]
    assert budget["application_input_limit"] is None and budget["application_policy_fits"] is None
    assert budget["server_context_window"] is None and budget["fits"] is None
    assert budget["single_bos_verified"] is None
    assert result.trace["usage"] is None and result.trace["termination_verified"] is False
    assert result.status == "completed"
    assert "private-header-value" not in json.dumps([result.trace, events])


@pytest.mark.parametrize("limit_count,status,generations", [(100, "completed", 1), (101, "budget_exceeded", 0)])
async def test_explicit_input_only_application_limit_has_inclusive_boundary(limit_count, status, generations):
    seen = []
    def handler(req):
        seen.append(req.url.path)
        data = {"tokens": limit_count} if req.url.path.endswith("/count") else answer(json.loads(req.content))
        return httpx.Response(200, json=data)
    async with client(handler, count_input_tokens=True, input_token_limit=100) as model:
        result = await model.complete(MESSAGES, max_tokens=2048)
    assert result.status == status
    assert seen.count("/v1/batch/completions") == generations
    assert result.trace["completion_attempted"] is bool(generations)
    assert result.trace["budget"]["application_policy_fits"] is bool(generations)
    assert result.trace["budget"]["reserved_output_tokens"] == 2048


@pytest.mark.parametrize("body", [b'{"tokens":true}', b'{"tokens":12.0}', b'{"tokens":0}',
    b'{"tokens":-1}', b'{"tokens":"12"}', b'{"tokens":12,"error":"failed"}',
    b'{"tokens":12,"tokens":13}', b'{"tokens":NaN}', b'[]', b'{}', b'\xff'])
async def test_bad_count_retains_actual_body_and_never_generates_or_retries(body):
    seen = []
    def handler(req):
        seen.append(req.url.path)
        return httpx.Response(200, content=body)
    async with client(handler, count_input_tokens=True) as model:
        result = await model.complete(MESSAGES)
    assert seen == ["/v1/tokens/count"]
    assert result.status == "token_count_failed" and not result.trace["completion_attempted"]
    assert raw(result.trace["http"][0], "response") == body
    assert result.trace["budget"]["input_tokens"] is None


@pytest.mark.parametrize("failure", ["http", "timeout", "connect"])
async def test_count_network_failures_have_independent_terminal_receipts(failure):
    seen, events = [], []
    async def handler(req):
        seen.append(req.url.path)
        if failure == "timeout":
            await asyncio.sleep(.2)
        if failure == "connect":
            raise httpx.ConnectError("secret error detail")
        return httpx.Response(503, content=b'not available')
    async with client(handler, count_input_tokens=True, timeout_seconds=.025,
                      recorder=lambda e, r: events.append((e, r))) as model:
        result = await model.complete(MESSAGES)
    assert result.status == "token_count_failed"
    assert seen == ["/v1/tokens/count"]
    assert [e for e, _ in events] == ["token_count_started", "token_count_completed"]
    assert events[1][1]["ended_at"] and events[1][1]["http_attempted"]
    assert "secret error detail" not in json.dumps(events)
    if failure == "http":
        assert raw(result.trace["http"][0], "response") == b'not available'


async def test_partial_count_response_and_recorder_failure_are_not_success():
    class Partial(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"tokens":'
            raise httpx.ReadError("private diagnostic")
    async with client(lambda req: httpx.Response(200, stream=Partial()), count_input_tokens=True) as model:
        result = await model.complete(MESSAGES)
    assert result.status == "token_count_failed"
    entry = result.trace["http"][0]
    assert raw(entry, "response") == b'{"tokens":' and not entry["response_body_complete"]


@pytest.mark.parametrize("event_to_fail,expected_count_calls", [("token_count_started", 0), ("token_count_completed", 1)])
async def test_failed_durable_recorder_prevents_generation(event_to_fail, expected_count_calls):
    seen = []
    def handler(req):
        seen.append(req.url.path)
        return httpx.Response(200, json={"tokens": 99})
    def recorder(event, record):
        if event == event_to_fail:
            raise OSError("private storage path")
    async with client(handler, count_input_tokens=True, recorder=recorder) as model:
        result = await model.complete(MESSAGES)
    assert result.status == "token_count_failed" and len(seen) == expected_count_calls
    assert all(p.endswith("/count") for p in seen)
    assert "private storage path" not in json.dumps(result.trace)


async def test_count_and_generation_share_global_slots_with_mixed_oversize_items():
    active, peak, generations = 0, 0, []
    async def handler(req):
        nonlocal active, peak
        payload = json.loads(req.content)
        n = 1 if "text" in payload else len(payload["contents"])
        active += n
        peak = max(peak, active)
        await asyncio.sleep(.004)
        active -= n
        if "text" in payload:
            return httpx.Response(200, json={"tokens": 101 if "oversize" in payload["text"] else 50})
        generations.extend(payload["contents"])
        return httpx.Response(200, json=answer(payload))
    async with client(handler, count_input_tokens=True, input_token_limit=100, max_concurrency=3,
                      batch_size=2, batch_wait_ms=0) as model:
        results = await asyncio.gather(*[model.complete([{"role": "user", "content": "oversize" if i % 3 == 0 else f"ok{i}"}]) for i in range(12)])
    assert peak <= 3 and len(generations) == 8
    assert sum(r.status == "budget_exceeded" for r in results) == 4
    assert sum(r.status == "completed" for r in results) == 8
    assert not any("oversize" in prompt for prompt in generations)


async def test_total_deadline_includes_count_and_batch_queue_no_ghost_generation():
    seen = []
    async def handler(req):
        seen.append(req.url.path)
        await asyncio.sleep(.02)
        return httpx.Response(200, json={"tokens": 50})
    async with client(handler, count_input_tokens=True, timeout_seconds=.04, batch_wait_ms=60) as model:
        result = await model.complete(MESSAGES)
    assert seen == ["/v1/tokens/count"]
    assert result.status == "timeout" and not result.trace["completion_attempted"]
    assert result.trace["budget"]["input_tokens"] == 50


async def test_cancel_count_and_close_do_not_dispatch_phantom_batches():
    entered = asyncio.Event()
    seen, events = [], []
    async def handler(req):
        seen.append(req.url.path)
        entered.set()
        await asyncio.sleep(10)
    model = client(handler, count_input_tokens=True, max_concurrency=1,
                   recorder=lambda e, r: events.append((e, r)))
    first = asyncio.create_task(model.complete(MESSAGES))
    await entered.wait()
    second = asyncio.create_task(model.complete(MESSAGES))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    await model.aclose()
    result = await second
    assert result.status == "token_count_failed"
    assert all(path.endswith("/tokens/count") for path in seen)
    assert not model._pending and not model._workers and not model._count_jobs
    assert model._slots._value == 1
    assert sum(e == "token_count_completed" for e, _ in events) == 2


@pytest.mark.parametrize("kwargs", [{"input_token_limit": 100}, {"count_input_tokens": True, "input_token_limit": True},
    {"count_input_tokens": True, "input_token_limit": 0}, {"count_input_tokens": "yes"}])
def test_bad_policy_is_rejected_before_client_creation(kwargs):
    with pytest.raises(ValueError):
        client(lambda req: None, **kwargs)


def test_settings_factory_wires_only_explicit_count_policy():
    settings = Settings(_env_file=None, native_transport="rwkvos_batch")
    assert model_client_options(settings)["count_input_tokens"] is False
    assert model_client_options(settings)["input_token_limit"] is None
    enabled = Settings(_env_file=None, native_transport="rwkvos_batch",
                       rwkvos_count_input_tokens=True, rwkvos_input_token_limit=8000)
    assert model_client_options(enabled)["count_input_tokens"] is True
    assert model_client_options(enabled)["input_token_limit"] == 8000
    with pytest.raises(ValueError):
        Settings(_env_file=None, rwkvos_input_token_limit=8000)
    with pytest.raises(ValueError):
        Settings(_env_file=None, rwkvos_count_input_tokens=True, rwkvos_input_token_limit=True)


@pytest.mark.parametrize("stage", ["token_count_started", "batch_started"])
async def test_blocking_recorder_cannot_dispatch_http_after_deadline(stage):
    seen = []
    def recorder(event, record):
        if event == stage:
            sleep(.035)  # Simulate a slow synchronous fsync without yielding.
    def handler(req):
        seen.append(req.url.path)
        return httpx.Response(200, json={"tokens": 5})
    async with client(handler, count_input_tokens=True, timeout_seconds=.02,
                      recorder=recorder, batch_wait_ms=0) as model:
        result = await model.complete(MESSAGES)
    expected = [] if stage == "token_count_started" else ["/v1/tokens/count"]
    assert seen == expected
    assert result.status == ("token_count_failed" if not expected else "timeout")
    assert not result.trace["completion_attempted"]


async def test_count_deadline_already_elapsed_before_counter_task_has_no_http():
    seen = []
    async with client(lambda req: seen.append(req), count_input_tokens=True,
                      timeout_seconds=.000001) as model:
        result = await model.complete(MESSAGES)
    assert seen == []
    assert result.status == "token_count_failed"
    assert not result.trace["http"][0]["http_attempted"]
    assert model._slots._value == model.max_concurrency
