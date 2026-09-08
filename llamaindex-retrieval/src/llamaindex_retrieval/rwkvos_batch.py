"""Indexed raw-content RWKVoS transport. No retries or output rewriting.

``completed`` means a valid response and answer boundary, not verified natural
termination: the observed provider reports ``stop`` even at its output limit.
Authentication is private; an optional awaited recorder receives exact HTTP bytes
before dispatch and after completion, once per shared batch or token-count call.
"""

from __future__ import annotations

import asyncio
import base64
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import inspect
import json
import math
from time import perf_counter
from typing import Callable, Sequence
from uuid import uuid4

import httpx

from .native_rwkv import NativeRWKVResult


def render_batch_prompt(messages: list[dict[str, str]], assistant_prefill: str,
                        prefill_mode: str = "complete") -> tuple[str, str]:
    """Render all original message characters using the observed plain template."""
    aliases = {"<think": "<think>", "<think>": "<think>",
               "<think></think": "<think></think>", "<think></think>": "<think></think>"}
    if assistant_prefill not in aliases:
        raise ValueError("unsupported assistant prefill")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    labels = {"system": "System", "user": "User", "assistant": "Assistant"}
    parts = []
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or not isinstance(message["role"], str) or message["role"] not in labels
                or not isinstance(message["content"], str)):
            raise ValueError("unsupported message structure")
        parts.append(f"{labels[message['role']]}: {message['content']}")
    if prefill_mode not in {"complete", "continuation"}:
        raise ValueError("unsupported prefill mode")
    prefill = aliases[assistant_prefill]
    if prefill_mode == "continuation":
        prefill = prefill[:-1]
    parts.append(f"Assistant: {prefill}")
    return "\n\n".join(parts), prefill


def inspect_batch_envelope(raw: str, prefill: str) -> dict:
    """Locate the actual answer without synthesizing a native leading ``>``."""
    start = 0
    valid = isinstance(raw, str) and bool(raw.strip())
    if prefill in {"<think", "<think></think"}:
        # The provider returns the missing final '>' as actual output. Locate
        # it in raw text; never synthesize, prepend or strip output characters.
        combined = prefill + raw
        close = combined.find("</think>", len("<think>"))
        valid = (valid and raw.startswith(">") and close >= 0
                 and "<think" not in combined[len("<think>"):close])
        start = close + len("</think>") - len(prefill) if close >= 0 else 0
    elif prefill == "<think>":
        close = raw.find("</think>")
        valid = valid and close >= 0 and "<think" not in raw[:close]
        start = close + len("</think>") if close >= 0 else 0
    elif prefill != "<think></think>":
        valid = False
    answer = raw[start:]
    valid = valid and bool(answer.strip()) and not answer.lstrip().startswith(("<think", "</think"))
    return {"valid": bool(valid), "answer_span": (
        {"start": start, "end": len(raw), "unit": "unicode_code_points"} if valid else None),
        "raw_text_modified": False}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _number(value: object, name: str, low: float, high: float | None = None) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < low or (high is not None and value > high)):
        raise ValueError(f"invalid {name}")


def _json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


@dataclass
class _Call:
    future: asyncio.Future
    trace: dict
    parameters: dict
    started: float
    prefill: str
    prompt: str = ""
    cancelled: bool = False
    dispatched: bool = False
    slot_owned: bool = False
    deadline: float | None = None


class RwkvosBatchClient:
    """Coalesce equal-parameter calls, preserving every slot's identity.

    ``max_concurrency`` bounds actual in-flight contents, including slots whose
    caller cancelled after dispatch. A cancelled caller never cancels its peers.
    ``aclose`` explicitly cancels active HTTP batches and fails unsent calls.
    ``recorder(event, record)`` may be sync or async; exceptions fail the batch.
    The recorder must provide its own durable storage (e.g. exclusive + fsync).
    No server tokenizer or state is inferred or implicitly reused.
    Enabled counting shares the concurrency bound and a deadline measured from
    ``complete`` entry, including queues. Completed-receipt cleanup is separately
    bounded by ``timeout_seconds``; counting does not establish a server limit.
    """

    def __init__(
        self, *, base_url: str, model: str, headers: dict[str, str] | None = None,
        timeout_seconds: float = 120, context_window_tokens: int = 16384,
        max_concurrency: int = 32, transport: httpx.AsyncBaseTransport | None = None,
        batch_size: int = 8, batch_wait_ms: float = 5, state_id: str | None = None,
        prefill_mode: str = "complete",
        recorder: Callable | None = None, alpha_decay: float = .99, chunk_size: int = 8,
        stop_tokens: list[int] | None = None,
        count_input_tokens: bool = False, input_token_limit: int | None = None,
    ) -> None:
        url = httpx.URL(base_url)
        if (url.scheme not in {"http", "https"} or not url.host or url.userinfo
                or url.query or url.fragment):
            raise ValueError("base_url must be a credential-free HTTP(S) endpoint")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model is required")
        _number(timeout_seconds, "timeout_seconds", .000001)
        _number(batch_wait_ms, "batch_wait_ms", 0)
        _number(alpha_decay, "alpha_decay", 0, 1)
        for name, value in (("context_window_tokens", context_window_tokens),
                            ("max_concurrency", max_concurrency), ("batch_size", batch_size),
                            ("chunk_size", chunk_size)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if state_id is not None and (not isinstance(state_id, str) or not state_id.strip()):
            raise ValueError("state_id must be a nonempty string or None")
        if type(count_input_tokens) is not bool:
            raise ValueError("count_input_tokens must be boolean")
        if input_token_limit is not None and (
            type(input_token_limit) is not int or input_token_limit < 1 or not count_input_tokens
        ):
            raise ValueError("input_token_limit requires enabled counting and a positive integer")
        if stop_tokens is not None and (not isinstance(stop_tokens, list) or any(
            type(token) is not int or token < 0 for token in stop_tokens
        ) or len(set(stop_tokens)) != len(stop_tokens)):
            raise ValueError("stop_tokens must be unique nonnegative integers or None")
        if headers is not None and (not isinstance(headers, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in headers.items()
        )):
            raise ValueError("headers must contain strings")
        reserved = {"host", "content-length", "transfer-encoding", "content-type"}
        if any(k.lower() in reserved for k in (headers or {})):
            raise ValueError("transport headers cannot be overridden")
        if recorder is not None and not callable(recorder):
            raise ValueError("recorder must be callable")
        self.base_url = str(url).rstrip("/")
        self.url = (self.base_url if self.base_url.endswith("/batch/completions")
                    else self.base_url + "/batch/completions")
        self.count_url = self.url.removesuffix("/batch/completions") + "/tokens/count"
        self.count_input_tokens = count_input_tokens
        self.input_token_limit = input_token_limit
        self.model = model
        self.context_window_tokens = context_window_tokens
        self.timeout_seconds = timeout_seconds
        self.max_concurrency = max_concurrency
        self.batch_size = min(batch_size, max_concurrency)
        self.batch_wait_ms = batch_wait_ms
        if prefill_mode not in {"complete", "continuation"}:
            raise ValueError("unsupported prefill mode")
        self.prefill_mode = prefill_mode
        self.state_id = state_id
        self.alpha_decay = alpha_decay
        self.chunk_size = chunk_size
        self.stop_tokens = deepcopy(stop_tokens)
        self._headers = {"Content-Type": "application/json", **(headers or {})}
        self._recorder = recorder
        self._pending: dict[str, deque[_Call]] = {}
        self._wake = asyncio.Event()
        self._dispatcher: asyncio.Task | None = None
        self._workers: set[asyncio.Task] = set()
        self._count_jobs: set[asyncio.Task] = set()
        # Count-enabled calls retain one slot through counting, queueing and
        # generation. A dispatched cancelled slot is released by the batch.
        self._slots = asyncio.Semaphore(max_concurrency)
        self._active = 0
        self._closed = False
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds, transport=transport, trust_env=False,
            follow_redirects=False, limits=httpx.Limits(max_connections=max_concurrency,
                                                       max_keepalive_connections=max_concurrency))

    async def __aenter__(self) -> RwkvosBatchClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        count_jobs = list(self._count_jobs)
        for job in count_jobs:
            job.cancel()
        if count_jobs:
            await asyncio.gather(*count_jobs, return_exceptions=True)
        if self._dispatcher is not None:
            self._dispatcher.cancel()
            await asyncio.gather(self._dispatcher, return_exceptions=True)
        for queue in self._pending.values():
            for call in queue:
                call.trace["error_type"] = "ClientClosedBeforeDispatch"
                self._finish(call, "transport_error")
        self._pending.clear()
        workers = list(self._workers)
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        await self._client.aclose()

    async def _record(self, event: str, record: dict) -> None:
        if self._recorder is not None:
            result = self._recorder(event, deepcopy(record))
            if inspect.isawaitable(result):
                await result

    def _finish(self, call: _Call, status: str, raw: str | None = None,
                finish: str | None = None) -> None:
        self._release_slot(call)
        record = call.trace
        record.update(raw_text=raw, finish_reason=finish, provider_finish_reason=finish)
        if raw is not None:
            record["raw_text_sha256"] = sha256(raw.encode("utf-8")).hexdigest()
        record["batch_item_status"] = status
        if call.cancelled:
            record["shared_batch_received_at"] = _now()
            return
        record.update(status=status, ended_at=_now(), elapsed_ms=(perf_counter() - call.started) * 1000)
        if not call.future.done():
            call.future.set_result(NativeRWKVResult(status, raw, finish, record))

    def _release_slot(self, call: _Call) -> None:
        if call.slot_owned:
            call.slot_owned = False
            self._slots.release()

    async def _count_input(self, call: _Call) -> bool:
        """Count exact rendered input; never infer a provider context limit."""
        began = perf_counter()
        entry = {"count_id": str(uuid4()), "call_id": call.trace["call_id"],
                 "stage": "token_count", "model_stage": call.trace["stage"],
                 "evidence_ids": deepcopy(call.trace["evidence_ids"]),
                 "url": self.count_url, "started_at": _now(), "http_attempted": False,
                 "response_body_complete": False, "payload": {"text": call.prompt},
                 "prompt_sha256": call.trace["prompt_sha256"]}
        call.trace["token_count_id"] = entry["count_id"]
        call.trace["http"].append(entry)
        chunks: list[bytes] = []
        status, tokens, response, cancelled = "transport_error", None, None, False
        try:
            remaining = call.deadline - perf_counter()
            if remaining <= 0:
                raise TimeoutError
            async with asyncio.timeout(remaining):
                request = self._client.build_request("POST", self.count_url,
                                                     headers=self._headers, json=entry["payload"])
                entry.update(request_body_base64=base64.b64encode(request.content).decode("ascii"),
                             request_body_sha256=sha256(request.content).hexdigest())
                entry["recorder_stage"] = "token_count_started"
                await self._record("token_count_started", entry)
                entry["recorder_stage"] = None
                await self._slots.acquire()
                call.slot_owned = True
                if perf_counter() >= call.deadline:
                    raise TimeoutError
                if self._closed:
                    raise RuntimeError("client closed")
                entry["http_attempted"] = True
                response = await self._client.send(request, stream=True)
                entry["http_status"] = response.status_code
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)
                entry["response_body_complete"] = True
                response.raise_for_status()
                data = json.loads(b"".join(chunks).decode("utf-8"), object_pairs_hook=_json_object,
                                  parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
                entry["response_json"] = data
                if (not isinstance(data, dict) or "error" in data
                        or type(data.get("tokens")) is not int or data["tokens"] <= 0):
                    raise ValueError("invalid token count for nonempty prompt")
                tokens, status = data["tokens"], "completed"
        except asyncio.CancelledError:
            cancelled = not self._closed
            status = "cancelled"
            entry["error_type"] = "ClientClosedDuringTokenCount" if self._closed else "CancelledError"
        except (TimeoutError, httpx.TimeoutException) as error:
            status, entry["error_type"] = "timeout", type(error).__name__
        except httpx.HTTPStatusError as error:
            status, entry["error_type"] = "http_error", type(error).__name__
        except (ValueError, UnicodeDecodeError) as error:
            status = "transport_error" if entry.get("recorder_stage") else "invalid_response"
            entry["error_type"] = type(error).__name__
        except Exception as error:
            entry["error_type"] = type(error).__name__
        finally:
            body = b"".join(chunks)
            entry.update(response_body_base64=base64.b64encode(body).decode("ascii"),
                         response_body_sha256=sha256(body).hexdigest())
            if response is not None:
                try:
                    await response.aclose()
                except Exception as error:
                    status, entry["close_error_type"] = "transport_error", type(error).__name__
            entry.update(ended_at=_now(), elapsed_ms=(perf_counter() - began) * 1000, status=status)
            try:
                # Complete the durable receipt even when the request deadline
                # or caller cancellation ended the network operation.
                async with asyncio.timeout(self.timeout_seconds):
                    await self._record("token_count_completed", entry)
            except BaseException as error:
                status = "transport_error"
                entry.update(status=status, recorder_error_type=type(error).__name__)
        if cancelled:
            raise asyncio.CancelledError
        if status != "completed":
            call.trace["error_type"] = entry.get("recorder_error_type", entry.get("error_type"))
            return False
        call.trace["budget"].update(
            input_tokens=tokens, input_token_count_source="server_tokens_count",
            counted_prompt_sha256=entry["prompt_sha256"], verification="server_count_only",
            application_input_limit=self.input_token_limit,
            application_policy_fits=(None if self.input_token_limit is None
                                     else tokens <= self.input_token_limit))
        return True

    async def complete(
        self, messages: list[dict[str, str]], *, max_tokens: int = 2048,
        assistant_prefill: str = "<think></think>", temperature: float = .001,
        top_p: float = 0, top_k: int = 20, presence_penalty: float = 0,
        frequency_penalty: float = 0, seed: int | None = None,
        stage: str = "reader", evidence_ids: Sequence[str] = (), trace: dict | None = None,
    ) -> NativeRWKVResult:
        started = perf_counter()
        record = trace if trace is not None else {}
        record.update(call_id=str(uuid4()), transport="rwkvos_batch", stage=stage, model=self.model,
                      state_id=self.state_id, messages=deepcopy(messages), evidence_ids=list(evidence_ids),
                      started_at=_now(), status="pending", http=[], raw_text=None, finish_reason=None,
                      completion_attempted=False, usage=None, termination="unknown",
                      termination_verified=False, provider_finish_reason=None,
                      provider_default_stop_tokens_verified=False,
                      provider_default_stop_tokens={"source": "upstream_reported",
                                                    "tokens": [0, 261, 24281],
                                                    "deployment_verified": False},
                      stop_tokens_requested=deepcopy(self.stop_tokens),
                      budget={"input_tokens": None, "reserved_output_tokens": max_tokens,
                              "configured_context_window": self.context_window_tokens,
                              "server_context_window": None, "fits": None,
                              "single_bos_verified": None, "verification": "unknown"})
        call = _Call(asyncio.get_running_loop().create_future(), record, {}, started, "")
        try:
            if self._closed:
                raise ValueError("client is closed")
            prompt, prefill = render_batch_prompt(messages, assistant_prefill, self.prefill_mode)
            if type(max_tokens) is not int or max_tokens < 1:
                raise ValueError("max_tokens must be positive")
            if type(top_k) is not int or top_k < 0:
                raise ValueError("top_k must be nonnegative")
            _number(temperature, "temperature", 0, 1000)
            _number(top_p, "top_p", 0, 1)
            _number(presence_penalty, "presence_penalty", -float("inf"))
            _number(frequency_penalty, "frequency_penalty", -float("inf"))
            if seed is not None:
                raise ValueError("seed is not supported by this provider contract")
            parameters = {"model": self.model, "max_tokens": max_tokens,
                          "temperature": max(.001, temperature), "top_k": top_k, "top_p": top_p,
                          "alpha_presence": presence_penalty, "alpha_frequency": frequency_penalty,
                          "alpha_decay": self.alpha_decay, "chunk_size": self.chunk_size,
                          "stream": False}
            if self.stop_tokens is not None:
                parameters["stop_tokens"] = deepcopy(self.stop_tokens)
            if self.state_id is not None:
                parameters["state_id"] = self.state_id
            record.update(prompt=prompt, prompt_sha256=sha256(prompt.encode()).hexdigest(),
                          requested_prefill=assistant_prefill, prefill=prefill, prefill_mode=self.prefill_mode,
                          requested_temperature=temperature, parameters=deepcopy(parameters))
            call.parameters, call.prefill, call.prompt = parameters, prefill, prompt
            if self.count_input_tokens:
                call.deadline = started + self.timeout_seconds
                record["timeout_scope"] = "complete_including_count_and_queues"
                record["budget"].update(application_input_limit=self.input_token_limit,
                                        application_policy_fits=None)
                counter = asyncio.create_task(self._count_input(call), name="rwkvos-token-count")
                self._count_jobs.add(counter)
                counter.add_done_callback(self._count_jobs.discard)
                if not await counter:
                    self._finish(call, "token_count_failed")
                    return call.future.result()
                if self._closed:
                    record["error_type"] = "ClientClosedBeforeDispatch"
                    self._finish(call, "transport_error")
                    return call.future.result()
                if record["budget"]["application_policy_fits"] is False:
                    record["error_type"] = "ApplicationInputTokenLimitExceeded"
                    self._finish(call, "budget_exceeded")
                    return call.future.result()
            key = json.dumps(parameters, sort_keys=True, allow_nan=False)
            self._pending.setdefault(key, deque()).append(call)
            if self._dispatcher is None:
                self._dispatcher = asyncio.create_task(self._dispatch(), name="rwkvos-batch-dispatch")
            self._wake.set()
            return await asyncio.shield(call.future)
        except asyncio.CancelledError:
            call.cancelled = True
            call.future.cancel()
            record.update(status="cancelled", error_type="CancelledError", ended_at=_now(),
                          elapsed_ms=(perf_counter() - started) * 1000,
                          cancellation_effect=("shared_batch_continues" if call.dispatched
                                               else "removed_before_dispatch"))
            self._wake.set()
            if not call.dispatched:
                self._release_slot(call)
            raise
        except (ValueError, TypeError) as error:
            record["error_type"] = type(error).__name__
            self._finish(call, "invalid_request")
            return call.future.result()

    async def _dispatch(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            await asyncio.sleep(self.batch_wait_ms / 1000)
            while self._pending and self._active < self.max_concurrency:
                key = next(iter(self._pending))
                queue = self._pending[key]
                batch = []
                capacity = min(self.batch_size, self.max_concurrency - self._active)
                while queue and len(batch) < capacity:
                    call = queue.popleft()
                    if not call.cancelled:
                        batch.append(call)
                if not queue:
                    del self._pending[key]
                if not batch:
                    continue
                self._active += len(batch)
                for call in batch:
                    call.dispatched = True
                worker = asyncio.create_task(self._batch(batch), name="rwkvos-batch-http")
                self._workers.add(worker)
                worker.add_done_callback(self._workers.discard)

    @staticmethod
    def _choices(data: dict, count: int, model: str) -> list[dict]:
        if (not isinstance(data, dict) or data.get("object") != "chat.completion"
                or "error" in data or data.get("model") != model
                or not isinstance(data.get("choices"), list)
                or len(data["choices"]) != count):
            raise ValueError("invalid batch response identity or count")
        slots = {}
        for choice in data["choices"]:
            if not isinstance(choice, dict):
                raise ValueError("invalid batch choice")
            index, message = choice.get("index"), choice.get("message")
            if (type(index) is not int or index < 0 or index >= count or index in slots
                    or not isinstance(message, dict) or message.get("role") != "assistant"
                    or not isinstance(message.get("content"), str)
                    or "error" in choice or not isinstance(choice.get("finish_reason"), str)
                    or choice["finish_reason"] not in {"stop", "length"}):
                raise ValueError("invalid indexed batch choice")
            message["content"].encode("utf-8")  # Reject unpaired JSON surrogates.
            slots[index] = choice
        return [slots[index] for index in range(count)]

    async def _batch(self, batch: list[_Call]) -> None:
        began = perf_counter()
        batch_id = str(uuid4())
        entry = {"batch_id": batch_id, "stage": "batch_completions", "url": self.url,
                 "started_at": _now(), "http_attempted": False, "response_body_complete": False,
                 "slots": [{"index": i, "call_id": c.trace["call_id"], "stage": c.trace["stage"],
                            "evidence_ids": deepcopy(c.trace["evidence_ids"])} for i, c in enumerate(batch)],
                 "payload": {**batch[0].parameters, "contents": [c.prompt for c in batch]}}
        for index, call in enumerate(batch):
            call.trace.update(batch_id=batch_id, batch_index=index, batch_size=len(batch),
                              queue_ms=(began - call.started) * 1000,
                              provider_item_latency_ms=None, latency_scope="shared_batch_round_trip")
            call.trace["http"].append(entry)
        status, choices, response = "transport_error", None, None
        chunks: list[bytes] = []
        try:
            deadlines = [call.deadline for call in batch if call.deadline is not None]
            timeout = (max(0, min(deadlines) - perf_counter()) if deadlines else self.timeout_seconds)
            if timeout <= 0:
                raise TimeoutError
            async with asyncio.timeout(timeout):
                request = self._client.build_request("POST", self.url, headers=self._headers,
                                                     json=entry["payload"])
                entry.update(request_body_base64=base64.b64encode(request.content).decode("ascii"),
                             request_body_sha256=sha256(request.content).hexdigest())
                entry["recorder_stage"] = "batch_started"
                await self._record("batch_started", entry)
                entry["recorder_stage"] = None
                if deadlines and perf_counter() >= min(deadlines):
                    raise TimeoutError
                entry["http_attempted"] = True
                for call in batch:
                    call.trace["completion_attempted"] = True
                response = await self._client.send(request, stream=True)
                entry["http_status"] = response.status_code
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)
                entry["response_body_complete"] = True
                response.raise_for_status()
                data = json.loads(b"".join(chunks).decode("utf-8"), object_pairs_hook=_json_object,
                                  parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
                entry["response_json"] = data
                choices = self._choices(data, len(batch), self.model)
                entry["items"] = []
                for index, (call, choice) in enumerate(zip(batch, choices, strict=True)):
                    envelope = inspect_batch_envelope(choice["message"]["content"], call.prefill)
                    finish = choice["finish_reason"]
                    item_status = ("length" if finish == "length" else
                                   "completed" if envelope["valid"] else "invalid_response")
                    entry["items"].append({"index": index, "call_id": call.trace["call_id"],
                                           "status": item_status, "provider_finish_reason": finish,
                                           "termination_verified": False, "envelope": envelope,
                                           "provider_item_latency_ms": None})
                status = "completed"
        except asyncio.CancelledError:
            entry["error_type"] = "ClientClosedDuringBatch" if self._closed else "BatchCancelled"
        except (TimeoutError, httpx.TimeoutException) as error:
            status, entry["error_type"] = "timeout", type(error).__name__
        except httpx.HTTPStatusError as error:
            status, entry["error_type"] = "http_error", type(error).__name__
        except (ValueError, UnicodeDecodeError) as error:
            status = "transport_error" if entry.get("recorder_stage") else "invalid_response"
            entry["error_type"] = type(error).__name__
        except Exception as error:
            entry["error_type"] = type(error).__name__
        finally:
            body = b"".join(chunks)
            entry.update(response_body_base64=base64.b64encode(body).decode("ascii"),
                         response_body_sha256=sha256(body).hexdigest(), ended_at=_now(),
                         elapsed_ms=(perf_counter() - began) * 1000, status=status)
            if response is not None:
                try:
                    await response.aclose()
                except Exception as error:
                    entry["close_error_type"] = type(error).__name__
                    status = "transport_error"
            entry["status"] = status
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    await self._record("batch_completed", entry)
            except BaseException as error:
                entry["recorder_error_type"] = type(error).__name__
                status = "transport_error"
            entry["status"] = status
            for index, call in enumerate(batch):
                call.trace["active_ms"] = (perf_counter() - began) * 1000
                if status == "completed" and choices is not None:
                    choice = choices[index]
                    raw, finish = choice["message"]["content"], choice["finish_reason"]
                    call.trace["provider_choice"] = deepcopy(choice)
                    item = entry["items"][index]
                    call.trace["envelope"] = deepcopy(item["envelope"])
                    self._finish(call, item["status"], raw, finish)
                else:
                    call.trace["error_type"] = entry.get("recorder_error_type", entry.get("error_type"))
                    if choices is not None:
                        choice = choices[index]
                        call.trace["provider_choice"] = deepcopy(choice)
                        self._finish(call, status, choice["message"]["content"], choice["finish_reason"])
                    else:
                        self._finish(call, status)
            self._active -= len(batch)
            self._wake.set()
