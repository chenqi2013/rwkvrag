#!/usr/bin/env python3
"""Fixed-material development smoke, never a blind or database retrieval test.

Example (the output directory must not already exist):
  python eval/native-smoke/run_smoke.py --base-url http://127.0.0.1:18421/v1 \
    --model rwkv7-g1j-13.3b-zero-state-capability-ctx16384 --output /new/smoke-run

--validate-only and --self-test make no network requests. The latter exercises
the real pipeline/native client through an in-process httpx MockTransport.
No automatic semantic score is produced. The separate expected_* oracle is for
explicit review of the final answer, actual citation support and extra claims.
"""

import argparse
import asyncio
import base64
import contextvars
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone

CASE_ID = contextvars.ContextVar("native_smoke_case_id")
MATERIAL_KEYS = {"id", "document_id", "source", "title", "uri", "score", "snippet", "metadata"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"invalid JSON number: {value}")

    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def write_new(path, data):
    """Exclusive creation plus file and directory fsync; never replace receipts."""
    data = data if isinstance(data, bytes) else encode(data) + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def load_fixtures(path):
    raw = path.read_bytes()
    rows = [strict_json(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 8:
        raise ValueError("the fixed development smoke requires all 8 cases")
    ids = set()
    for row in rows:
        case_id = row["id"]
        if not isinstance(case_id, str) or not re.fullmatch(r"smoke_[0-9]{3}", case_id):
            raise ValueError("invalid case ID")
        if case_id in ids:
            raise ValueError("duplicate case ID")
        ids.add(case_id)
        if row.get("phase") != "development_smoke_not_blind":
            raise ValueError("fixtures must explicitly identify their exposed development status")
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError("empty question")
        if not isinstance(row["history"], list) or not isinstance(row["materials"], list):
            raise ValueError("history and materials must be lists")
        for message in row["history"]:
            if (
                not isinstance(message, dict)
                or set(message) != {"role", "content"}
                or message["role"] not in {"user", "assistant"}
                or not isinstance(message["content"], str)
            ):
                raise ValueError("invalid original history")
        materials = {}
        for material in row["materials"]:
            if not isinstance(material, dict) or set(material) != MATERIAL_KEYS:
                raise ValueError("material must have exactly the 8 SourceItem fields")
            for key in ["id", "document_id", "source", "title", "snippet"]:
                if not isinstance(material[key], str) or not material[key]:
                    raise ValueError(f"invalid material {key}")
            if material["id"] in materials:
                raise ValueError("duplicate material ID")
            if material["uri"] is not None and not isinstance(material["uri"], str):
                raise ValueError("invalid material URI")
            if (
                type(material["score"]) not in {int, float}
                or not math.isfinite(material["score"])
                or not isinstance(material["metadata"], dict)
            ):
                raise ValueError("invalid material score or metadata")
            materials[material["id"]] = material
        if row["provenance"]["materials_sha256"] != sha256(encode(row["materials"])):
            raise ValueError("material bytes differ from their fixture provenance")
        if row["expected_decision"] not in {"answer", "partial_abstain", "abstain", "clarify"}:
            raise ValueError("invalid expected decision")
        fact_ids = set()
        for fact in row["expected_facts"]:
            if fact["id"] in fact_ids or not fact["claim"].strip():
                raise ValueError("duplicate or empty expected fact")
            fact_ids.add(fact["id"])
            if not fact["supporting_quotes"]:
                raise ValueError(
                    "a positive fact needs an independently specified supporting quote"
                )
            for evidence in fact["supporting_quotes"]:
                text = materials[evidence["material_id"]]["snippet"]
                start, end = evidence["start"], evidence["end"]
                if (
                    type(start) is not int
                    or type(end) is not int
                    or not 0 <= start < end <= len(text)
                    or text[start:end] != evidence["quote"]
                ):
                    raise ValueError(
                        "oracle quote/Unicode offsets do not match the actual material"
                    )
    return raw, rows


def runtime_modules(module_root):
    source = (module_root / "src").resolve()
    sys.path.insert(0, str(source))
    names = ["config", "schemas", "native_rwkv", "rwkv_pipeline"]
    modules = {name: importlib.import_module(f"llamaindex_retrieval.{name}") for name in names}
    for name, module in tuple(sys.modules.items()):
        if name.startswith("llamaindex_retrieval") and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(source):
                raise ValueError("another checkout's llamaindex_retrieval was already imported")
    return modules


def source_inventory(module_root):
    base = module_root / "src"
    inventory = []
    for path in sorted((base / "llamaindex_retrieval").rglob("*.py")):
        raw = path.read_bytes()
        inventory.append(
            {"path": path.relative_to(base).as_posix(), "sha256": sha256(raw), "bytes": len(raw)}
        )
    if not inventory:
        raise ValueError("no production source files found")
    return inventory


class NoIndex:
    def __getattr__(self, name):
        raise AssertionError(f"fixed-material smoke must not access a database/index: {name}")


def recording_client_class(native_class):
    class RecordingClient(native_class):
        def __init__(self, output, **kwargs):
            super().__init__(**kwargs)
            self.output = output
            self.traces = {}
            self.call_counts = Counter()
            self.http_counts = Counter()

        async def complete(self, messages, **kwargs):
            case_id = CASE_ID.get()
            self.call_counts[case_id] += 1
            number = self.call_counts[case_id]
            trace = kwargs.pop("trace", {})
            self.traces.setdefault(case_id, []).append(trace)
            prefix = f"{case_id}.{number:03}"
            write_new(
                self.output / "model-started" / f"{prefix}.json",
                {
                    "id": case_id,
                    "model_call_number": number,
                    "started_at": utc_now(),
                    "messages": messages,
                    "arguments": kwargs,
                    "first_attempt": True,
                    "automatic_retry": False,
                },
            )
            try:
                return await super().complete(messages, trace=trace, **kwargs)
            finally:
                write_new(
                    self.output / "model-completed" / f"{prefix}.json",
                    {
                        "id": case_id,
                        "model_call_number": number,
                        "trace": trace,
                        "trace_sha256": sha256(encode(trace)),
                    },
                )

        async def _post(self, url, payload, stage, trace):
            case_id = CASE_ID.get()
            self.http_counts[case_id] += 1
            prefix = f"{case_id}.{self.http_counts[case_id]:03}.{stage}"
            # Build only, without sending. The actual native method uses the
            # same client/JSON serializer; verify its recorded body afterwards.
            request = self._client.build_request("POST", url, headers=self._headers, json=payload)
            request_sha = sha256(request.content)
            write_new(
                self.output / "http-started" / f"{prefix}.json",
                {
                    "id": case_id,
                    "stage": stage,
                    "url": url,
                    "started_at": utc_now(),
                    "payload": payload,
                    "request_body_base64": base64.b64encode(request.content).decode(),
                    "request_body_sha256": request_sha,
                    "automatic_retry": False,
                },
            )
            before = len(trace["http"])
            try:
                return await super()._post(url, payload, stage, trace)
            finally:
                event = trace["http"][before] if len(trace["http"]) > before else None
                verified = bool(event and event.get("request_body_sha256") == request_sha)
                write_new(
                    self.output / "http-completed" / f"{prefix}.json",
                    {
                        "id": case_id,
                        "event": event,
                        "pre_send_body_matches": verified,
                    },
                )
                if event is not None and not verified:
                    raise RuntimeError("the native client changed the frozen pre-send body")

    return RecordingClient


async def execute(args, *, transport=None):
    fixtures_raw, fixtures = load_fixtures(args.fixtures)
    inventory = source_inventory(args.module_root)
    modules = runtime_modules(args.module_root)
    settings = modules["config"].Settings(
        _env_file=None,
        rag_pipeline="rwkv",
        generation_output_mode="immutable",
        native_base_url=args.base_url,
        native_model=args.model,
        native_api_key=os.environ.get(args.api_key_env, ""),
        native_timeout_seconds=args.timeout_seconds,
        native_context_window_tokens=args.context_window,
        native_max_concurrency=args.concurrency,
        generation_max_tokens=args.max_output_tokens,
    )
    source_class = modules["schemas"].SourceItem
    history_class = modules["schemas"].ConversationMessage
    requests = [
        {
            "id": row["id"],
            "question": row["question"],
            "history": [
                history_class.model_validate(x).model_dump(mode="json") for x in row["history"]
            ],
            "materials": [
                source_class.model_validate(x).model_dump(mode="json") for x in row["materials"]
            ],
        }
        for row in fixtures
    ]
    client_type = recording_client_class(modules["native_rwkv"].NativeRWKVClient)
    # Validate URL/model before reserving the run. Construction never makes HTTP.
    client = client_type(
        args.output,
        base_url=args.base_url,
        model=args.model,
        api_key=settings.native_api_key,
        timeout_seconds=args.timeout_seconds,
        context_window_tokens=args.context_window,
        max_concurrency=args.concurrency,
        transport=transport,
    )
    try:
        args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    except BaseException:
        await client.aclose()
        raise
    for directory in [
        "started",
        "completed",
        "model-started",
        "model-completed",
        "http-started",
        "http-completed",
        "frozen",
    ]:
        (args.output / directory).mkdir(mode=0o700)
    write_new(args.output / "frozen" / "fixtures.jsonl", fixtures_raw)
    write_new(args.output / "frozen" / "run_smoke.py", Path(__file__).read_bytes())
    for item in inventory:
        path = args.output / "frozen" / "src" / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (args.module_root / "src" / item["path"]).read_bytes()
        if sha256(raw) != item["sha256"]:
            await client.aclose()
            raise ValueError("source changed during snapshot; no requests sent")
        write_new(path, raw)
    manifest = {
        "schema": "rwkvrag-native-material-smoke-v1",
        "created_at": utc_now(),
        "purpose": "exposed development smoke, not blind evaluation or retrieval quality",
        "entrypoint": "RWKVPipeline.ask_materials",
        "database_access_allowed": False,
        "planned_cases": len(fixtures),
        "case_ids": [row["id"] for row in fixtures],
        "expected_fact_count": sum(len(row["expected_facts"]) for row in fixtures),
        "automatic_retry": False,
        "resume_allowed": False,
        "semantic_scoring": "explicit review only",
        "fixture_sha256": sha256(fixtures_raw),
        "runner_sha256": sha256(Path(__file__).read_bytes()),
        "production_sources": inventory,
        "production_sources_sha256": sha256(encode(inventory)),
        "configuration": {
            "base_url": args.base_url,
            "model": args.model,
            "concurrency": args.concurrency,
            "native_max_concurrency": args.concurrency,
            "timeout_seconds": args.timeout_seconds,
            "context_window_tokens": args.context_window,
            "max_output_tokens": args.max_output_tokens,
            "generation_output_mode": "immutable",
            "api_key_present": bool(settings.native_api_key),
            "native_complete_parameter_defaults": {
                key: parameter.default
                for key, parameter in inspect.signature(
                    modules["native_rwkv"].NativeRWKVClient.complete
                ).parameters.items()
                if key
                in {
                    "assistant_prefill",
                    "temperature",
                    "top_p",
                    "top_k",
                    "presence_penalty",
                    "frequency_penalty",
                    "seed",
                }
            },
        },
        "oracle_is_model_input": False,
        "per_case_timeout_note": "native timeout bounds tokenization plus completion after acquiring its slot",
    }
    write_new(args.output / "manifest.json", manifest)
    write_new(args.output / "requests.jsonl", b"".join(encode(row) + b"\n" for row in requests))
    if inventory != source_inventory(args.module_root):
        await client.aclose()
        raise ValueError("source changed before dispatch; no requests sent")
    pipeline = modules["rwkv_pipeline"].RWKVPipeline(settings, NoIndex(), model=client)
    gate = asyncio.Semaphore(args.concurrency)
    records = {}
    start = time.perf_counter()

    async def one(request):
        case_id = request["id"]
        context = CASE_ID.set(case_id)
        record = {"id": case_id, "attempted": False, "response": None, "error_type": None}
        queued = time.perf_counter()
        active = None
        try:
            async with gate:
                active = time.perf_counter()
                record["queue_ms"] = (active - queued) * 1000
                record["request_sha256"] = sha256(encode(request))
                write_new(
                    args.output / "started" / f"{case_id}.json",
                    {
                        "id": case_id,
                        "started_at": utc_now(),
                        "first_attempt": True,
                        "request": request,
                        "request_sha256": record["request_sha256"],
                        "manifest_sha256": sha256(encode(manifest) + b"\n"),
                    },
                )
                record["attempted"] = True
                response = await pipeline.ask_materials(
                    question=request["question"],
                    materials=[source_class.model_validate(x) for x in request["materials"]],
                    history=[history_class.model_validate(x) for x in request["history"]],
                )
                record["response"] = response.model_dump(mode="json")
                record["status"] = record["response"]["generation"]["status"]
                record["response_sha256"] = sha256(encode(record["response"]))
        except asyncio.CancelledError:
            record.update(
                status="cancelled" if record["attempted"] else "not_started",
                error_type="CancelledError",
            )
        except Exception as error:
            record.update(status="runner_error", error_type=type(error).__name__)
        finally:
            record.update(
                completed_at=utc_now(),
                elapsed_ms=(time.perf_counter() - active) * 1000 if active is not None else None,
                model_traces=client.traces.get(case_id, []),
            )
            record["model_traces_sha256"] = sha256(encode(record["model_traces"]))
            write_new(args.output / "completed" / f"{case_id}.json", record)
            records[case_id] = record
            CASE_ID.reset(context)

    tasks = [asyncio.create_task(one(request)) for request in requests]
    interrupted = False
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        interrupted = True
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await pipeline.aclose()
    ordered = [records[request["id"]] for request in requests]
    unchanged = inventory == source_inventory(args.module_root)
    summary = {
        "schema": "rwkvrag-native-material-smoke-summary-v1",
        "planned_cases": len(fixtures),
        "recorded_cases": len(ordered),
        "attempted_cases": sum(row["attempted"] for row in ordered),
        "status_counts": dict(Counter(row["status"] for row in ordered)),
        "model_calls": sum(client.call_counts.values()),
        "http_requests": sum(client.http_counts.values()),
        "elapsed_ms": (time.perf_counter() - start) * 1000,
        "interrupted": interrupted,
        "source_unchanged": unchanged,
        "semantic_support_verified": False,
        "semantic_review_required": True,
        "complete_rag_quality_claim": False,
        "automatic_retries": 0,
        "results_sha256": sha256(encode(ordered) + b"\n"),
    }
    write_new(args.output / "results.json", ordered)
    write_new(args.output / "summary.json", summary)
    return summary


async def self_test(args):
    import httpx

    calls = []

    async def handler(request):
        case_id = CASE_ID.get()
        body = strict_json(request.content)
        calls.append((case_id, request.url.path, body))
        if request.url.path == "/tokenize":
            return httpx.Response(
                200, json={"count": 4, "tokens": [0, 1, 2, 3], "max_model_len": 16384}
            )
        if case_id == "smoke_002":
            return httpx.Response(503, json={"error": "offline fixture failure; must not retry"})
        text = "></think>离线模拟答复，保持原样。"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"text": text, "finish_reason": "length" if case_id == "smoke_006" else "stop"}
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 12, "total_tokens": 16},
            },
        )

    with tempfile.TemporaryDirectory(prefix="native-smoke-offline-") as temporary:
        mocked = argparse.Namespace(**vars(args))
        mocked.output = Path(temporary) / "run"
        mocked.base_url, mocked.model = "http://offline.invalid/v1", "offline-model"
        summary = await execute(mocked, transport=httpx.MockTransport(handler))
        assert summary["attempted_cases"] == summary["recorded_cases"] == 8
        assert summary["model_calls"] == 8 and len(calls) == summary["http_requests"] == 16
        assert summary["status_counts"] == {"completed": 6, "http_error": 1, "length": 1}
        assert summary["source_unchanged"] and not summary["semantic_support_verified"]
        _, fixtures = load_fixtures(args.fixtures)
        for row in fixtures:
            completion = next(
                body
                for cid, path, body in calls
                if cid == row["id"] and path.endswith("/completions")
            )
            prompt = completion["prompt"]
            assert "expected_facts" not in prompt and "expected_decision" not in prompt
            assert "oracle_notes" not in prompt and "supporting_quotes" not in prompt
            for material in row["materials"]:
                assert json.dumps(material["snippet"], ensure_ascii=False) in prompt
            for message in row["history"]:
                assert json.dumps(message["content"], ensure_ascii=False) in prompt
            saved = strict_json((mocked.output / "started" / f"{row['id']}.json").read_text())
            assert saved["request"]["history"] == row["history"]
            assert [m["snippet"] for m in saved["request"]["materials"]] == [
                m["snippet"] for m in row["materials"]
            ]
            complete = strict_json((mocked.output / "completed" / f"{row['id']}.json").read_text())
            for event in complete["model_traces"][0]["http"]:
                assert (
                    sha256(base64.b64decode(event["request_body_base64"]))
                    == event["request_body_sha256"]
                )
                assert (
                    sha256(base64.b64decode(event["response_body_base64"]))
                    == event["response_body_sha256"]
                )
        long_record = strict_json((mocked.output / "completed" / "smoke_006.json").read_text())
        assert long_record["response"]["answer"] == "></think>离线模拟答复，保持原样。"
        for receipt in (mocked.output / "http-completed").glob("*.json"):
            assert strict_json(receipt.read_text())["pre_send_body_matches"]
        old_manifest = (mocked.output / "manifest.json").read_bytes()
        try:
            await execute(mocked, transport=httpx.MockTransport(handler))
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing run was accepted")
        assert (mocked.output / "manifest.json").read_bytes() == old_manifest
        assert len(calls) == 16
        duplicated = [dict(row) for row in fixtures]
        duplicated[1]["id"] = duplicated[0]["id"]
        bad_fixture = Path(temporary) / "duplicate.jsonl"
        bad_fixture.write_bytes(b"".join(encode(row) + b"\n" for row in duplicated))
        try:
            load_fixtures(bad_fixture)
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate case IDs were accepted")

        completion_entered = asyncio.Event()
        cancelled_calls = []

        async def cancel_handler(request):
            cancelled_calls.append(request.url.path)
            if request.url.path == "/tokenize":
                return httpx.Response(
                    200, json={"count": 4, "tokens": [0, 1, 2, 3], "max_model_len": 16384}
                )
            completion_entered.set()
            await asyncio.Event().wait()
            raise AssertionError("cancelled mock completion unexpectedly resumed")

        cancelled = argparse.Namespace(**vars(mocked))
        cancelled.output = Path(temporary) / "cancelled"
        cancelled.concurrency = 1
        task = asyncio.create_task(
            execute(cancelled, transport=httpx.MockTransport(cancel_handler))
        )
        await asyncio.wait_for(completion_entered.wait(), timeout=10)
        task.cancel()
        cancelled_summary = await task
        assert cancelled_summary["interrupted"]
        assert cancelled_summary["recorded_cases"] == 8
        assert cancelled_summary["attempted_cases"] == 1
        assert cancelled_summary["status_counts"] == {"cancelled": 1, "not_started": 7}
        assert cancelled_calls == ["/tokenize", "/v1/completions"]
        assert len(list((cancelled.output / "started").glob("*.json"))) == 1
        assert len(list((cancelled.output / "completed").glob("*.json"))) == 8
        cancel_record = strict_json((cancelled.output / "completed" / "smoke_001.json").read_text())
        assert cancel_record["model_traces"][0]["status"] == "cancelled"
        assert cancel_record["model_traces"][0]["http"][-1]["error_type"] == "CancelledError"
    return {
        "status": "PASS",
        "network_requests": 0,
        "in_process_mock_http_requests": 18,
        "cases": 8,
        "checks": [
            "full material/history preservation",
            "oracle request boundary",
            "request/response raw body hashes",
            "pre-send durable receipt equivalence",
            "all first attempts and 503 retained without retry",
            "length raw output immutable",
            "existing run refusal without overwrite or dispatch",
            "duplicate case refusal",
            "index access prohibited",
            "no automatic semantic scoring",
            "cancellation keeps in-flight trace and all 8 denominator slots",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).with_name("fixtures.jsonl"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument(
        "--api-key-env",
        default="RWKVRAG_NATIVE_API_KEY",
        help="environment variable name; the key value is never saved",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--context-window", type=int, default=16384)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 256:
        parser.error("--concurrency must be between 1 and 256")
    if args.validate_only:
        raw, rows = load_fixtures(args.fixtures)
        result = {
            "status": "PASS",
            "cases": len(rows),
            "facts": sum(len(row["expected_facts"]) for row in rows),
            "fixture_sha256": sha256(raw),
            "network_requests": 0,
        }
    elif args.self_test:
        result = asyncio.run(self_test(args))
    else:
        if not args.output or not args.base_url or not args.model:
            parser.error("--output, --base-url and --model are required for an actual run")
        result = asyncio.run(execute(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("interrupted"):
        return 130
    if result.get("source_unchanged") is False:
        return 2
    if result.get("status_counts") and any(key != "completed" for key in result["status_counts"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
