"""RWKV native HTTP transport; model text is immutable, including protocol markup.

Template contract: vllm-rwkv 67f0c5996c50 ``rwkv_defaults._render_plain_chat``.
One server-tokenizer request precedes at most one completion. There are no retries,
truncation, answer cleanup, citation repair, or implicit state reuse in this module.
"""

from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from time import perf_counter
from typing import Literal, Sequence
from uuid import uuid4

import httpx


NativeStatus = Literal[
    "completed", "length", "invalid_response", "budget_exceeded", "timeout",
    "http_error", "transport_error", "invalid_request",
]


@dataclass(frozen=True)
class NativePrompt:
    prompt: str
    prefill: str
    delimiter_escape: str | None


def render_native_prompt(
    messages: list[dict[str, str]], assistant_prefill: str = "<think",
) -> NativePrompt:
    """Keep every message character; escape only the native role delimiter.

    ``messages`` contains actual conversation turns, including assistant history.
    The separate prefill argument is configuration, never a history message.
    """
    aliases = {"<think": "<think", "<think>": "<think",
               "<think></think": "<think></think", "<think></think>": "<think></think"}
    if assistant_prefill not in aliases:
        raise ValueError("assistant_prefill must select open or empty native thinking")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    labels = {"system": "System", "user": "User", "assistant": "Bot"}
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or not isinstance(message["role"], str) or message["role"] not in labels
                or not isinstance(message["content"], str)):
            raise ValueError("each message must contain a supported role and string content")
    contents = [message["content"] for message in messages]
    escape = None
    if any("✿" in content for content in contents):
        number = 0
        while True:
            candidate = f"__RWKVRAG_LITERAL_U273F_{number}__"
            if all(candidate not in content for content in contents):
                escape = candidate
                break
            number += 1
    rendered = []
    for message in messages:
        content = message["content"]
        if escape is not None:
            content = content.replace("✿", escape)
        rendered.append(f"{labels[message['role']]}✿{content}✿")
    prefill = aliases[assistant_prefill]
    rendered.append(f"Bot✿{prefill}")
    return NativePrompt("\n".join(rendered), prefill, escape)


def inspect_envelope(
    raw_text: str, prefill: str = "<think",
) -> tuple[int, int] | None:
    """Return answer bounds in *raw_text*, without changing a single character.

    This checks only the native thinking boundary. Callers must additionally
    require result.status == 'completed' before accepting structured selections.
    Literal delimiter escape markers stay literal in both raw and returned span.
    """
    if (prefill not in {"<think", "<think></think"}
            or not isinstance(raw_text, str) or not raw_text.startswith(">")):
        return None
    combined = prefill + raw_text
    close = combined.find("</think>", len("<think>"))
    if close < 0 or "<think" in combined[len("<think>"):close]:
        return None
    start = close + len("</think>") - len(prefill)
    answer = raw_text[start:]
    if not answer.strip() or "✿" in answer or answer.lstrip().startswith(("<think", "</think")):
        return None
    return start, len(raw_text)


@dataclass(frozen=True)
class NativeRWKVResult:
    status: NativeStatus
    raw_text: str | None
    finish_reason: str | None
    trace: dict

    @property
    def text(self) -> str:
        """The exact choice.text, including thinking and surrounding whitespace."""
        return self.raw_text if self.raw_text is not None else ""


class _ProtocolError(ValueError):
    pass


class _BudgetError(ValueError):
    pass


class NativeRWKVClient:
    """Application-scoped client; all stages share one concurrency limit.

    Use one instance for the service lifetime and close it during shutdown.
    ``timeout_seconds`` bounds tokenization plus generation after a slot is
    acquired. Queue time is recorded separately. Cancellation is propagated;
    pass ``trace={}`` if its record must remain reachable after cancellation.
    """

    def __init__(
        self, *, base_url: str, model: str, api_key: str = "",
        timeout_seconds: float = 120, context_window_tokens: int = 16384,
        max_concurrency: int = 32, transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        url = httpx.URL(base_url)
        if (url.scheme not in {"http", "https"} or not url.host or url.userinfo
                or url.query or url.fragment):
            raise ValueError("base_url must be a credential-free HTTP(S) endpoint")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model is required")
        if not isinstance(api_key, str):
            raise ValueError("api_key must be a string")
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be finite and positive")
        for name, value in (("context_window_tokens", context_window_tokens),
                            ("max_concurrency", max_concurrency)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.base_url = str(url).rstrip("/")
        self.model = model
        self.context_window_tokens = context_window_tokens
        self.timeout_seconds = timeout_seconds
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds, transport=transport, trust_env=False,
            follow_redirects=False,
            limits=httpx.Limits(max_connections=max_concurrency,
                               max_keepalive_connections=max_concurrency),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> NativeRWKVClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def _post(self, url: str, payload: dict, stage: str, trace: dict) -> dict:
        entry = {"stage": stage, "url": url, "payload": deepcopy(payload)}
        trace["http"].append(entry)
        started = perf_counter()
        try:
            request = self._client.build_request("POST", url, headers=self._headers, json=payload)
            entry.update(
                request_body_base64=base64.b64encode(request.content).decode("ascii"),
                request_body_sha256=sha256(request.content).hexdigest(),
            )
            response = await self._client.send(request, stream=True)
            entry.update(http_status=response.status_code, response_body_complete=False)
            chunks: list[bytes] = []
            try:
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)
                entry["response_body_complete"] = True
            finally:
                body = b"".join(chunks)
                entry.update(
                    response_body_base64=base64.b64encode(body).decode("ascii"),
                    response_body_sha256=sha256(body).hexdigest(),
                )
                await response.aclose()
            response.raise_for_status()
            try:
                data = json.loads(body.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as error:
                raise _ProtocolError("response must contain UTF-8 JSON") from error
            if not isinstance(data, dict):
                raise _ProtocolError("response must contain a JSON object")
            return data
        except BaseException as error:
            entry["error_type"] = type(error).__name__
            raise
        finally:
            entry["elapsed_ms"] = (perf_counter() - started) * 1000

    def _check_budget(self, data: dict, output_tokens: int, trace: dict) -> int:
        count, tokens, server_limit = data.get("count"), data.get("tokens"), data.get("max_model_len")
        if (type(count) is not int or count < 1 or not isinstance(tokens, list)
                or len(tokens) != count or any(type(token) is not int or token < 0 for token in tokens)):
            raise _ProtocolError("tokenizer count must match its complete token ID list")
        if tokens[0] != 0 or tokens.count(0) != 1:
            raise _ProtocolError("tokenizer must return exactly one BOS/EOS 0 at the start")
        if type(server_limit) is not int or server_limit < 1:
            raise _ProtocolError("tokenizer must report a positive server context limit")
        effective_limit = min(server_limit, self.context_window_tokens)
        trace["budget"] = {
            "input_tokens": count, "reserved_output_tokens": output_tokens,
            "configured_context_window": self.context_window_tokens,
            "server_context_window": server_limit, "effective_context_window": effective_limit,
            "single_bos_verified": True, "fits": count + output_tokens <= effective_limit,
        }
        if count + output_tokens > effective_limit:
            raise _BudgetError("complete prompt plus reserved output exceeds context window")
        return count

    async def complete(
        self, messages: list[dict[str, str]], *, max_tokens: int = 2048,
        assistant_prefill: str = "<think", temperature: float = 0.0,
        top_p: float = 1.0, top_k: int = 0, presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0, seed: int | None = None,
        stage: str = "reader", evidence_ids: Sequence[str] = (), trace: dict | None = None,
    ) -> NativeRWKVResult:
        record = trace if trace is not None else {}
        record.update(
            call_id=str(uuid4()), stage=stage, model=self.model, http=[],
            messages=deepcopy(messages), evidence_ids=list(evidence_ids),
            started_at=datetime.now(timezone.utc).isoformat(), status="pending",
            raw_text=None, finish_reason=None, completion_attempted=False,
        )
        started = perf_counter()
        acquired = None
        status: NativeStatus = "invalid_request"
        raw_text = None
        finish_reason = None
        try:
            native = render_native_prompt(messages, assistant_prefill)
            if type(max_tokens) is not int or max_tokens < 1:
                raise ValueError("max_tokens must be a positive integer")
            for name, value, low, high in (
                ("temperature", temperature, 0, math.inf), ("top_p", top_p, 0, 1),
                ("presence_penalty", presence_penalty, -2, 2),
                ("frequency_penalty", frequency_penalty, -2, 2),
            ):
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or not low <= value <= high):
                    raise ValueError(f"invalid {name}")
            if top_p == 0 or type(top_k) is not int or top_k < -1:
                raise ValueError("top_p must be positive; top_k must be -1 or nonnegative")
            if seed is not None and type(seed) is not int:
                raise ValueError("seed must be an integer or None")
            if (not isinstance(stage, str) or isinstance(evidence_ids, (str, bytes))
                    or not all(isinstance(x, str) for x in evidence_ids)):
                raise ValueError("stage and evidence IDs must contain strings")
            payload = {
                "model": self.model, "prompt": native.prompt, "max_tokens": max_tokens,
                "temperature": temperature, "top_p": top_p, "top_k": top_k,
                "presence_penalty": presence_penalty, "frequency_penalty": frequency_penalty,
                "stream": False, "stop": ["✿"], "stop_token_ids": [0],
                "ignore_eos": False, "add_special_tokens": True,
            }
            if seed is not None:
                payload["seed"] = seed
            record.update(
                prompt=native.prompt, prompt_sha256=sha256(native.prompt.encode("utf-8")).hexdigest(),
                prefill=native.prefill, delimiter_escape=native.delimiter_escape,
                parameters={k: v for k, v in payload.items() if k != "prompt"},
            )
            async with self._semaphore:
                acquired = perf_counter()
                record["queue_ms"] = (acquired - started) * 1000
                async with asyncio.timeout(self.timeout_seconds):
                    token_data = await self._post(
                        self.base_url.removesuffix("/v1") + "/tokenize",
                        {"model": self.model, "prompt": native.prompt, "add_special_tokens": True},
                        "tokenize", record,
                    )
                    input_tokens = self._check_budget(token_data, max_tokens, record)
                    record["completion_attempted"] = True
                    data = await self._post(self.base_url + "/completions", payload, "completion", record)
                    record["usage"] = data.get("usage")
                    choices = data.get("choices")
                    if (not isinstance(choices, list) or len(choices) != 1
                            or not isinstance(choices[0], dict)):
                        raise _ProtocolError("completion must return exactly one choice")
                    choice = choices[0]
                    record["finish_reason"] = choice.get("finish_reason")
                    record["raw_text"] = choice.get("text")
                    if isinstance(choice.get("text"), str):
                        raw_text = choice["text"]
                        record["raw_text_sha256"] = sha256(raw_text.encode("utf-8")).hexdigest()
                    if isinstance(choice.get("finish_reason"), str):
                        finish_reason = choice["finish_reason"]
                    if raw_text is None:
                        raise _ProtocolError("completion choice.text must be a string")
                    usage = data.get("usage")
                    if (not isinstance(usage, dict) or type(usage.get("prompt_tokens")) is not int
                            or usage["prompt_tokens"] != input_tokens):
                        raise _ProtocolError("completion prompt_tokens must equal server tokenizer count")
                    bounds = inspect_envelope(raw_text, native.prefill)
                    record["envelope"] = {
                        "valid": bounds is not None,
                        "answer_span": {"start": bounds[0], "end": bounds[1],
                                        "unit": "unicode_code_points"} if bounds else None,
                        "raw_text_modified": False,
                    }
                    if finish_reason == "length":
                        status = "length"
                    elif finish_reason != "stop" or bounds is None:
                        raise _ProtocolError("completion did not naturally finish a valid native envelope")
                    else:
                        status = "completed"
        except asyncio.CancelledError:
            record.update(status="cancelled", error_type="CancelledError")
            raise
        except (TimeoutError, httpx.TimeoutException) as error:
            status = "timeout"
            record.update(error_type=type(error).__name__, error="model operation timed out")
        except httpx.HTTPStatusError as error:
            status = "http_error"
            record.update(error_type=type(error).__name__, error=f"HTTP {error.response.status_code}")
        except httpx.RequestError as error:
            status = "transport_error"
            record.update(error_type=type(error).__name__, error="model HTTP transport failed")
        except _BudgetError as error:
            status = "budget_exceeded"
            record.update(error_type=type(error).__name__, error=str(error))
        except _ProtocolError as error:
            status = "invalid_response"
            record.update(error_type=type(error).__name__, error=str(error))
        except ValueError as error:
            status = "invalid_request"
            record.update(error_type=type(error).__name__, error=str(error))
        except Exception as error:
            status = "transport_error"
            record.update(error_type=type(error).__name__, error="model client operation failed")
        finally:
            ended = perf_counter()
            record.update(
                ended_at=datetime.now(timezone.utc).isoformat(), elapsed_ms=(ended - started) * 1000,
                active_ms=(ended - acquired) * 1000 if acquired is not None else 0,
            )
            record.setdefault("queue_ms", (ended - started) * 1000 if acquired is None else 0)
            if record["status"] != "cancelled":
                record["status"] = status
        return NativeRWKVResult(status, raw_text, finish_reason, record)
