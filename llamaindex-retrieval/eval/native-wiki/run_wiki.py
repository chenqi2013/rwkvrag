#!/usr/bin/env python3
"""Eight previously exposed Wiki questions: development E2E smoke, not a blind test.

Examples (each invocation requires a new output directory):
  python eval/native-wiki/run_wiki.py --dry-run --out /tmp/wiki-dry \
    --base-url http://127.0.0.1:28421/v1 --opensearch-url http://127.0.0.1:28438 \
    --index rwkvrag-bm250820-wiki-5000-20260908 --model MODEL
  # Add --expected-index-uuid UUID and choose --preflight or --run.

--dry-run performs no network calls. --preflight performs only index reads.
--run explicitly enables inference. Existing output directories are always refused;
there is no resume, retry, selective rerun, answer repair, or automatic semantic score.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from collections import Counter
import contextvars
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from urllib.parse import urlencode, urlsplit


FIXTURE_SHA256 = "469dec3deb84115dceaec16af72a9f98f0ee51db503d4cfeeeed73e27e46a258"
CASE_IDS = [f"held_{i:03}" for i in (4, 10, 12, 14, 16, 17, 19, 21)]
CORPUS_MANIFEST_SHA256 = "35563fa6e5d7c9205cea3d9e1cd5476b01774f11a292dafcfed82fe4d8256ced"
CASE = contextvars.ContextVar("wiki_case", default="preflight")


def now():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n").encode("utf-8")


def digest(data):
    return sha256(data).hexdigest()


def write_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save(path, value):
    write_bytes(path, encoded(value))


def load_fixtures(path):
    data = path.read_bytes()
    if digest(data) != FIXTURE_SHA256:
        raise ValueError("fixtures do not match the frozen eight exposed questions")
    rows = [json.loads(line) for line in data.splitlines()]
    if [row["id"] for row in rows] != CASE_IDS:
        raise ValueError("fixture IDs or order differ from the declared eight cases")
    for row in rows:
        if set(row) != {"id", "question", "history", "oracle", "provenance"}:
            raise ValueError("unexpected fixture fields")
        if (not isinstance(row["question"], str) or not row["question"].strip()
                or not isinstance(row["history"], list)):
            raise ValueError("invalid original question/history")
        for message in row["history"]:
            if (set(message) != {"role", "content"}
                    or message["role"] not in {"user", "assistant"}
                    or not isinstance(message["content"], str)):
                raise ValueError("unsupported history message")
        if row["provenance"]["corpus_manifest_sha256"] != CORPUS_MANIFEST_SHA256:
            raise ValueError("corpus identity mismatch")
    return rows, data


def request_payload(row, candidate_k, knowledge_base_id):
    # The only bridge into the pipeline. Oracle and provenance are never forwarded.
    return {"question": row["question"], "history": row["history"],
            "candidate_k": candidate_k, "knowledge_base_id": knowledge_base_id}


def source_files(module_root):
    files = {str(path.relative_to(module_root)): path.read_bytes()
             for path in sorted((module_root / "src").rglob("*.py"))}
    for name in ("pyproject.toml", "uv.lock", "ARCHITECTURE_RULES.md"):
        path = module_root / name
        if path.is_file():
            files[name] = path.read_bytes()
    required = ("native_rwkv.py", "rwkv_pipeline.py", "lexical_index.py", "config.py", "schemas.py")
    if any(f"src/llamaindex_retrieval/{name}" not in files for name in required):
        raise ValueError("incomplete pipeline source tree")
    return files


def file_manifest(files):
    return {name: {"bytes": len(data), "sha256": digest(data)} for name, data in files.items()}


def expected_index(args):
    if args.index_freeze:
        raw = args.index_freeze.read_bytes()
        if not args.index_freeze_sha256 or digest(raw) != args.index_freeze_sha256:
            raise ValueError("--index-freeze requires its exact --index-freeze-sha256")
        freeze = json.loads(raw)
        if freeze.get("index") != args.index or freeze.get("document_count") != 5000:
            raise ValueError("index freeze must bind index and document_count=5000")
        if freeze.get("corpus_manifest_sha256") != CORPUS_MANIFEST_SHA256:
            raise ValueError("index freeze must bind the original corpus manifest")
        uuid = freeze.get("index_uuid")
    else:
        uuid = args.expected_index_uuid
    if uuid is not None and (not isinstance(uuid, str) or not re.fullmatch(r"[\w-]+", uuid)):
        raise ValueError("invalid index UUID")
    if not args.dry_run and not uuid:
        raise ValueError("online preflight/run requires an expected index UUID or frozen index record")
    return uuid


def index_preflight(client, index, expected_uuid, knowledge_base_id=None):
    """Exact exhaustive document IDs, plus primary shard sequence numbers; no writes."""
    settings = client.indices.get_settings(index=index)
    if set(settings) != {index}:
        raise ValueError("index must resolve to exactly one concrete index")
    actual_uuid = settings[index]["settings"]["index"]["uuid"]
    if actual_uuid != expected_uuid:
        raise ValueError("index UUID differs from the declared frozen index")
    query = ({"term": {"knowledge_base_id": knowledge_base_id}}
             if knowledge_base_id is not None else {"match_all": {}})
    count = client.count(index=index, body={"query": query})["count"]
    ids, after, seen_after = [], None, set()
    while True:
        composite = {"size": 1000, "sources": [{"document": {"terms": {"field": "document_id"}}}]}
        if after is not None:
            composite["after"] = after
        response = client.search(index=index, body={"size": 0, "query": query,
            "aggs": {"documents": {"composite": composite}}})
        if response.get("timed_out") or response.get("_shards", {}).get("failed", 0):
            raise ValueError("incomplete index document enumeration")
        aggregation = response["aggregations"]["documents"]
        batch = [bucket["key"]["document"] for bucket in aggregation["buckets"]]
        if not all(isinstance(item, str) and item for item in batch):
            raise ValueError("missing/invalid document identity")
        ids.extend(batch)
        next_after = aggregation.get("after_key")
        if not batch or next_after is None:
            break
        key = json.dumps(next_after, sort_keys=True)
        if key in seen_after:
            raise ValueError("index pagination repeated its cursor")
        seen_after.add(key)
        after = next_after
    if len(ids) != 5000 or len(set(ids)) != 5000:
        raise ValueError(f"expected exactly 5000 unique documents; observed {len(ids)}")
    missing = client.count(index=index, body={"query": {"bool": {
        "filter": [query], "must_not": [{"exists": {"field": "document_id"}}]}}})["count"]
    if missing:
        raise ValueError("indexed chunks without a document identity")
    stats = client.indices.stats(index=index, level="shards")
    primaries = []
    for shard, copies in stats["indices"][index]["shards"].items():
        for copy in copies:
            if copy["routing"]["primary"]:
                sequence = copy.get("seq_no", {})
                if type(sequence.get("max_seq_no")) is not int:
                    raise ValueError("primary shard sequence number unavailable")
                primaries.append({"shard": shard, "max_seq_no": sequence["max_seq_no"]})
    if not primaries:
        raise ValueError("index has no primary shard")
    return {"status": "PASS", "index": index, "index_uuid": actual_uuid,
            "document_count": len(ids), "chunk_count": count,
            "document_ids_sha256": digest(encoded(sorted(ids))),
            "primary_sequences": sorted(primaries, key=lambda item: item["shard"]),
            "knowledge_base_id": knowledge_base_id}


def receipt_value(value):
    """Preserve byte-valued client parameters without changing transport arguments."""
    if isinstance(value, bytes):
        return {"type": "bytes", "base64": base64.b64encode(value).decode("ascii"),
                "sha256": digest(value)}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("receipt parameter keys must be strings")
        return {key: receipt_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [receipt_value(item) for item in value]
    return value


def recording_connection(base_class, out, run_binding=None):
    counter, lock = itertools.count(1), threading.Lock()

    class RecordedConnection(base_class):
        def perform_request(self, method, url, params=None, body=None, **kwargs):
            # Enforce a read-only API allowlist even if production constructor code changes.
            if method not in {"GET", "HEAD", "POST"} or (method == "POST" and not url.endswith(("/_search", "/_count"))):
                raise ValueError(f"runner forbids index mutation: {method} {url}")
            with lock:
                ordinal = next(counter)
            stem = f"{ordinal:06}"
            raw = body if isinstance(body, bytes) else (body.encode("utf-8") if body else b"")
            started = {"run_binding": run_binding, "ordinal": ordinal, "case_id": CASE.get(), "started_at": now(),
                       "method": method, "url": url, "params": receipt_value(params),
                       "query_string": urlencode(params) if params else "",
                       "request_body_base64": base64.b64encode(raw).decode(),
                       "request_body_sha256": digest(raw)}
            save(out / "index-http" / (stem + ".started.json"), started)
            tick = time.perf_counter()
            completed = {"run_binding": run_binding, "ordinal": ordinal, "case_id": CASE.get()}
            try:
                result = super().perform_request(method, url, params=params, body=body, **kwargs)
                status, _, content = result
                raw_response = content if isinstance(content, bytes) else content.encode("utf-8")
                completed.update(http_status=status, response_body_base64=base64.b64encode(raw_response).decode(),
                                 response_body_sha256=digest(raw_response))
                return result
            except BaseException as error:
                completed.update(error_type=type(error).__name__, error=str(error))
                raise
            finally:
                completed.update(completed_at=now(), elapsed_ms=(time.perf_counter() - tick) * 1000)
                save(out / "index-http" / (stem + ".completed.json"), completed)
    return RecordedConnection


def recording_model(base_class, out, run_binding=None):
    counter = itertools.count(1)

    class RecordedModel(base_class):
        async def complete(self, messages, **kwargs):
            ordinal, case = next(counter), CASE.get()
            stem = f"{ordinal:06}"
            trace = kwargs.pop("trace", None)
            trace = trace if trace is not None else {}
            start = {"run_binding": run_binding, "ordinal": ordinal, "case_id": case, "started_at": now(),
                     "messages": messages, "parameters": kwargs}
            save(out / "model-calls" / (stem + ".started.json"), start)
            completed = {"run_binding": run_binding, "ordinal": ordinal, "case_id": case}
            try:
                result = await super().complete(messages, trace=trace, **kwargs)
                completed.update(status=result.status, raw_text=result.raw_text,
                                 finish_reason=result.finish_reason)
                return result
            except BaseException as error:
                completed.update(status="exception", error_type=type(error).__name__, error=str(error))
                raise
            finally:
                completed.update(completed_at=now(), trace=trace)
                save(out / "model-calls" / (stem + ".completed.json"), completed)
    return RecordedModel


async def run_cases(pipeline, requests, request_class, out, workers, run_binding=None):
    semaphore = asyncio.Semaphore(workers)

    async def one(item):
        async with semaphore:
            token = CASE.set(item["id"])
            started = time.perf_counter()
            row = {"run_binding": run_binding, "id": item["id"], "started_at": now(), "request": item["payload"]}
            save(out / "started" / (item["id"] + ".json"), row)
            completed = {"run_binding": run_binding, "id": item["id"],
                         "request_sha256": digest(encoded(item["payload"]))}
            try:
                request = request_class(**item["payload"])
                if request.question != item["payload"]["question"]:
                    raise ValueError("request schema changed the original question")
                history = [message.model_dump() for message in request.history]
                if history != item["payload"]["history"]:
                    raise ValueError("request schema changed the original history")
                response = await pipeline.ask(request)
                completed.update(outcome="response", response=response.model_dump(mode="json"))
            except BaseException as error:
                completed.update(outcome="exception", error_type=type(error).__name__, error=str(error))
                if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                    raise
            finally:
                completed.update(completed_at=now(), elapsed_ms=(time.perf_counter() - started) * 1000)
                save(out / "completed" / (item["id"] + ".json"), completed)
                CASE.reset(token)
            return completed
    return await asyncio.gather(*(one(item) for item in requests))


def summarize(out, ids):
    completed = [json.loads(path.read_text()) for path in sorted((out / "completed").glob("*.json"))]
    calls = [json.loads(path.read_text()) for path in sorted((out / "model-calls").glob("*.completed.json"))]
    status = Counter(row.get("response", {}).get("generation", {}).get("status", row.get("outcome"))
                     for row in completed)
    traces = [row["trace"] for row in calls]
    started_calls = [json.loads(path.read_text())
                     for path in sorted((out / "model-calls").glob("*.started.json"))]
    return {"kind": "previously_exposed_development_e2e_smoke", "original_case_count": len(ids),
            "completed_case_slots": len(completed), "case_statuses": dict(status),
            "not_completed_case_ids": [id for id in ids if id not in {row["id"] for row in completed}],
            "model_call_records": len(calls), "model_calls_started": len(started_calls),
            "model_calls_without_completed_receipts": [row["ordinal"] for row in started_calls
                if row["ordinal"] not in {call["ordinal"] for call in calls}],
            "generation_attempts": sum(bool(trace.get("completion_attempted")) for trace in traces),
            "model_http_requests": sum(len(trace.get("http", [])) for trace in traces),
            "model_finish_reasons": dict(Counter(str(trace.get("finish_reason")) for trace in traces)),
            "semantic_quality_scored": False, "oracle_sent_to_pipeline": False,
            "automatic_retries": 0, "all_original_failures_retained": True}


async def execute(args, out, requests, expected_uuid):
    # Import only the snapshot written before this process makes any inference call.
    sys.path.insert(0, str(out / "source" / "src"))
    if any(name.startswith("llamaindex_retrieval") for name in sys.modules):
        raise RuntimeError("pipeline modules were loaded before the frozen source snapshot")
    from opensearchpy import OpenSearch, Urllib3HttpConnection
    from llamaindex_retrieval.config import Settings
    from llamaindex_retrieval.lexical_index import LexicalIndex
    from llamaindex_retrieval.native_rwkv import NativeRWKVClient
    from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline
    from llamaindex_retrieval.schemas import SearchRequest

    class ExplicitSettings(Settings):
        @classmethod
        def settings_customise_sources(cls, settings_cls, init_settings, env_settings,
                                       dotenv_settings, file_secret_settings):
            return (init_settings,)

    settings = ExplicitSettings(
        rag_pipeline="rwkv", native_base_url=args.base_url, native_model=args.model,
        native_api_key=os.environ.get(args.api_key_env, ""),
        native_timeout_seconds=args.timeout, native_context_window_tokens=16384,
        native_max_concurrency=args.native_concurrency, native_resolver_sources=args.resolver_sources,
        native_planner_max_tokens=1024, native_resolver_max_tokens=1024,
        native_planner_prefill=args.planner_prefill,
        native_plan_protocol=args.plan_protocol,
        native_resolver_prefill=args.resolver_prefill,
        native_resolver_protocol=args.resolver_protocol,
        generation_max_tokens=args.max_tokens, generation_output_mode="immutable",
        opensearch_url=args.opensearch_url, opensearch_index=args.index,
        opensearch_username=None, opensearch_password=None, opensearch_verify_certs=True,
        candidate_k=args.candidate_k,
    )
    public_settings = settings.model_dump(mode="json")
    for name in list(public_settings):
        if any(word in name.lower() for word in ("password", "api_key", "secret")):
            public_settings[name] = {"configured": bool(public_settings[name]), "value_archived": False}
    save(out / "settings.json", public_settings)
    save(out / "settings-freeze.json", {
        "created_at": now(), "settings_sha256": digest(encoded(public_settings)),
        "manifest_sha256": digest((out / "manifest.json").read_bytes()),
        "all_environment_configuration_ignored_except_explicit_api_key": True,
    })
    run_binding = {"manifest_sha256": digest((out / "manifest.json").read_bytes()),
                   "settings_sha256": digest(encoded(public_settings))}
    client = OpenSearch(hosts=[args.opensearch_url], max_retries=0, retry_on_timeout=False,
                        retry_on_status=(), sniff_on_start=False, sniff_on_connection_fail=False,
                        http_compress=False, verify_certs=True, timeout=30,
                        connection_class=recording_connection(Urllib3HttpConnection, out, run_binding))
    model = pipeline = None
    try:
        before = await asyncio.to_thread(index_preflight, client, args.index, expected_uuid,
                                         args.knowledge_base_id)
        save(out / "index-before.json", before)
        if args.preflight:
            return {"status": "PREFLIGHT_PASS", "index": before, "model_calls": 0}

        class ReadOnlyIndex(LexicalIndex):
            def ensure_index(self):
                # The complete read-only identity preflight above replaces mutating auto-create/mapping.
                return None

        index = ReadOnlyIndex(settings, client=client)
        model = recording_model(NativeRWKVClient, out, run_binding)(
            base_url=settings.native_base_url, model=settings.native_model,
            api_key=settings.native_api_key, timeout_seconds=settings.native_timeout_seconds,
            context_window_tokens=settings.native_context_window_tokens,
            max_concurrency=settings.native_max_concurrency,
        )
        pipeline = RWKVPipeline(settings, index, model=model)
        save(out / "run-started.json", {"run_binding": run_binding, "started_at": now(),
            "case_count": len(requests), "question_concurrency": args.question_concurrency})
        await run_cases(pipeline, requests, SearchRequest, out, args.question_concurrency, run_binding)
        after = await asyncio.to_thread(index_preflight, client, args.index, expected_uuid,
                                        args.knowledge_base_id)
        save(out / "index-after.json", after)
        if after != before:
            raise ValueError("index identity, document set, count or primary sequences drifted")
        return {"status": "COMPLETED", "index_unchanged": True}
    finally:
        if model is not None:
            await model.aclose()
        client.close()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--opensearch-url", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--model", required=True)
    identity = parser.add_mutually_exclusive_group()
    identity.add_argument("--expected-index-uuid")
    identity.add_argument("--index-freeze", type=Path)
    parser.add_argument("--index-freeze-sha256")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).with_name("fixtures.jsonl"))
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--api-key-env", default="RWKVRAG_NATIVE_API_KEY")
    parser.add_argument("--question-concurrency", type=int, default=4)
    parser.add_argument("--native-concurrency", type=int, default=32)
    parser.add_argument("--resolver-sources", type=int, default=24)
    parser.add_argument("--planner-prefill", choices=("<think", "<think></think"), default="<think")
    parser.add_argument("--plan-protocol", choices=("queries_fields", "shared_tasks"), default="queries_fields")
    parser.add_argument("--resolver-prefill", choices=("<think", "<think></think"), default="<think")
    parser.add_argument("--resolver-protocol", choices=("fields", "task_units"), default="fields")
    parser.add_argument("--candidate-k", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def main():
    args = arguments()
    for endpoint in (args.base_url, args.opensearch_url):
        parsed = urlsplit(endpoint)
        if (parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("endpoints must be credential-free HTTP(S) URLs")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.index):
        raise ValueError("index must be a concrete non-wildcard name")
    if not 1 <= args.question_concurrency <= 8:
        raise ValueError("question concurrency must be 1..8")
    rows, fixtures = load_fixtures(args.fixtures)
    expected_uuid = expected_index(args)
    files = source_files(args.source_root.resolve())
    source_manifest = file_manifest(files)
    source_sha = digest(encoded(source_manifest))
    if args.expected_source_sha256 and source_sha != args.expected_source_sha256:
        raise ValueError("source differs from the explicitly frozen source hash")
    out = args.out.resolve()
    # Exclusive reservation happens before any imports with network-capable initialization.
    out.mkdir(parents=True, exist_ok=False)
    try:
        for name, data in files.items():
            write_bytes(out / "source" / name, data)
        save(out / "source-manifest.json", source_manifest)
        write_bytes(out / "frozen" / "fixtures.jsonl", fixtures)
        runner = Path(__file__).read_bytes()
        write_bytes(out / "frozen" / "run_wiki.py", runner)
        if args.index_freeze:
            write_bytes(out / "frozen" / "index-freeze.json", args.index_freeze.read_bytes())
        requests = [{"id": row["id"], "payload": request_payload(row, args.candidate_k,
                    args.knowledge_base_id)} for row in rows]
        save(out / "requests.json", requests)
        packages = {}
        for name in ("httpx", "opensearch-py", "pydantic", "pydantic-settings", "jieba",
                     "opencc-python-reimplemented", "llama-index-core", "pyarrow"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        manifest = {"schema": "bm250820-native-wiki-smoke-v1", "created_at": now(),
            "kind": "previously_exposed_development_e2e_smoke", "case_ids": CASE_IDS,
            "original_case_count": 8, "oracle_required_fact_count": 29,
            "fixtures_sha256": digest(fixtures), "runner_sha256": digest(runner),
            "source_manifest_sha256": source_sha, "requests_sha256": digest(encoded(requests)),
            "expected_index_uuid": expected_uuid, "corpus_manifest_sha256": CORPUS_MANIFEST_SHA256,
            "arguments": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
            "python": sys.version, "packages": packages,
            "oracle_in_pipeline_requests": False, "automatic_retries": 0}
        save(out / "manifest.json", manifest)
        if args.dry_run:
            result = {"status": "DRY_RUN_PASS", "network_calls": 0,
                      "source_manifest_sha256": source_sha, "case_count": len(requests)}
        else:
            result = asyncio.run(execute(args, out, requests, expected_uuid))
        if file_manifest(source_files(args.source_root.resolve())) != source_manifest:
            raise ValueError("working source changed during the run; frozen execution bytes retained")
        result.update(summary=summarize(out, CASE_IDS), completed_at=now())
        save(out / "result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except BaseException as error:
        save(out / "run-error.json", {"error_type": type(error).__name__, "error": str(error),
                                      "at": now(), "summary": summarize(out, CASE_IDS)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
