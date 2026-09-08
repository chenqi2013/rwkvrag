import asyncio
from copy import deepcopy
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import httpx
import pytest

from llamaindex_retrieval.batch_receipts import summarize_batch_http, summarize_token_http


ROOT = Path(__file__).resolve().parents[1]


def receipt(directory, identity, event, attempted, **extra):
    directory.mkdir(parents=True, exist_ok=True)
    row = {"batch_id": identity, "http_attempted": attempted, **extra}
    (directory / f"{identity}.{event}.json").write_text(json.dumps(row))
    return row


def test_deduplicate_shared_batch_and_keep_each_dispatch_class(tmp_path):
    receipt(tmp_path, "sent", "batch_started", False)
    sent = receipt(tmp_path, "sent", "batch_completed", True, ended_at="finished")
    receipt(tmp_path, "unsent", "batch_started", False)
    receipt(tmp_path, "unsent", "batch_completed", False, ended_at="finished")
    receipt(tmp_path, "interrupted", "batch_started", False)
    traces = [{"batch_id": "sent", "http": [sent]} for _ in range(8)]
    original = deepcopy(traces)

    summary = summarize_batch_http(tmp_path, traces)

    assert summary["unique_batches"] == 3
    assert summary["confirmed_attempted"] == summary["never_attempted"] == summary["unknown"] == 1
    assert summary["http_requests"] is None
    assert summary["count_exact"] is False
    assert summary["confirmed_attempted_lower_bound"] == 1
    assert summary["possible_attempted_upper_bound"] == 2
    assert traces == original


@pytest.mark.parametrize("completed,attempted,expected", [
    (False, False, "unknown"),
    (True, False, "never_attempted"),
    (True, True, "confirmed_attempted"),
    (True, None, "unknown"),
])
def test_receipt_evidence_without_any_model_completed_files(tmp_path, completed, attempted, expected):
    receipt(tmp_path, "only", "batch_completed" if completed else "batch_started", attempted)
    summary = summarize_batch_http(tmp_path)
    assert summary[expected] == 1
    assert summary["http_requests"] == (None if expected == "unknown" else int(attempted))


def test_completed_model_trace_recovers_failed_batch_recorder_without_double_count(tmp_path):
    receipt(tmp_path, "batch", "batch_started", False)
    trace = {"batch_id": "batch", "http": [
        {"batch_id": "batch", "http_attempted": True, "ended_at": "finished"}]}
    assert summarize_batch_http(tmp_path, [trace] * 8)["http_requests"] == 1
    trace["http"][0]["http_attempted"] = False
    assert summarize_batch_http(tmp_path, [trace])["never_attempted"] == 1
    del trace["http"][0]["ended_at"]
    assert summarize_batch_http(tmp_path, [trace])["unknown"] == 1


def test_corrupt_receipt_remains_unknown_and_conflicting_terminal_records_are_flagged(tmp_path):
    (tmp_path / "broken.batch_started.json").write_text('{"batch_id":')
    receipt(tmp_path, "conflict", "batch_completed", False)
    summary = summarize_batch_http(tmp_path, [{"http": [
        {"batch_id": "conflict", "http_attempted": True, "ended_at": "finished"}]}])
    assert summary["unknown"] == 2
    assert summary["unreadable_receipts"] == ["broken.batch_started.json"]
    assert next(row for row in summary["batches"] if row["batch_id"] == "conflict")[
        "conflicting_terminal_evidence"] is True


def test_token_receipts_and_generation_batches_have_separate_id_spaces(tmp_path):
    receipt(tmp_path, "same-id", "batch_completed", True)
    count = {"count_id": "same-id", "call_id": "call", "http_attempted": True,
             "ended_at": "finished"}
    (tmp_path / "same-id.token_count_completed.json").write_text(json.dumps(count))
    traces = [{"token_count_id": "same-id", "batch_id": "same-id", "http": [count,
               {"batch_id": "same-id", "http_attempted": True, "ended_at": "finished"}]}] * 8
    assert summarize_batch_http(tmp_path, traces)["unique_batches"] == 1
    assert summarize_token_http(tmp_path, traces)["unique_counts"] == 1
    assert summarize_token_http(tmp_path, traces)["http_requests"] == 1
    incomplete = summarize_token_http(tmp_path / "absent", [{"token_count_id": "pending"}])
    assert incomplete["unknown"] == 1
    assert incomplete["http_requests"] is None


def test_wiki_summary_counts_batch_receipts_before_any_model_completed_receipt(tmp_path):
    summarize = runpy.run_path(str(ROOT / "eval/native-wiki/run_wiki.py"))["summarize"]
    receipt(tmp_path / "batch-http", "shared", "batch_started", False)
    incomplete = summarize(tmp_path, ["case1", "case2"])
    assert incomplete["model_http_requests"] is None
    assert incomplete["batch_http"]["unknown"] == 1
    assert incomplete["not_completed_case_ids"] == ["case1", "case2"]

    receipt(tmp_path / "batch-http", "shared", "batch_completed", True)
    complete_http = summarize(tmp_path, ["case1", "case2"])
    assert complete_http["model_http_requests"] == 1
    assert complete_http["model_call_records"] == 0
    assert complete_http["not_completed_case_ids"] == ["case1", "case2"]


def test_wiki_native_http_accounting_stays_unchanged(tmp_path):
    folder = tmp_path / "model-calls"
    folder.mkdir()
    (folder / "001.completed.json").write_text(json.dumps({
        "ordinal": 1, "trace": {"http": [{"stage": "tokenize"}, {"stage": "completion"}]}}))
    summarize = runpy.run_path(str(ROOT / "eval/native-wiki/run_wiki.py"))["summarize"]
    result = summarize(tmp_path, ["case1"])
    assert result["model_http_requests"] == 2
    assert result["batch_http"] is None


def test_wiki_token_only_interruption_is_not_mistaken_for_zero_model_http(tmp_path):
    folder = tmp_path / "token-http"
    folder.mkdir()
    row = {"count_id": "counter", "call_id": "call", "http_attempted": False}
    (folder / "counter.token_count_started.json").write_text(json.dumps(row))
    summarize = runpy.run_path(str(ROOT / "eval/native-wiki/run_wiki.py"))["summarize"]
    result = summarize(tmp_path, ["case1"])
    assert result["model_http_requests"] is None
    assert result["generation_batch_http_requests"] == 0
    assert result["input_token_count_http_requests"] is None
    row.update(http_attempted=True, ended_at="finished")
    (folder / "counter.token_count_completed.json").write_text(json.dumps(row))
    result = summarize(tmp_path, ["case1"])
    assert result["model_http_requests"] == result["input_token_count_http_requests"] == 1
    assert result["generation_batch_http_requests"] == 0
    assert result["model_call_records"] == 0


@pytest.mark.parametrize("transport_kind,fail_after_start,count_tokens", [
    ("rwkvos_batch", False, False), ("rwkvos_batch", True, False),
    ("rwkvos_batch", False, True), ("native", False, False),
])
def test_material_runner_real_receipt_lifecycle_with_offline_transport(
    tmp_path, monkeypatch, transport_kind, fail_after_start, count_tokens,
):
    execute = runpy.run_path(str(ROOT / "eval/native-smoke/run_smoke.py"))["execute"]
    if fail_after_start:
        write = execute.__globals__["write_new"]

        def failing_write(path, value):
            write(path, value)
            if path.name.endswith(".batch_started.json"):
                raise OSError("simulated directory fsync failure after file creation")

        monkeypatch.setitem(execute.__globals__, "write_new", failing_write)

    requests = []

    async def handler(request):
        requests.append(request.url.path)
        body = json.loads(request.content)
        if request.url.path == "/v1/tokens/count":
            assert isinstance(body["text"], str) and body["text"]
            return httpx.Response(200, json={"tokens": 123})
        if transport_kind == "rwkvos_batch":
            return httpx.Response(200, json={"object": "chat.completion", "model": "offline-model",
                "choices": [{"index": i, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": "离线模拟正文。"}}
                    for i in range(len(body["contents"]))]})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 2, "tokens": [0, 1], "max_model_len": 16384})
        return httpx.Response(200, json={"choices": [{"text": ">离线模拟正文。", "finish_reason": "stop"}],
                                        "usage": {"prompt_tokens": 2}})

    args = SimpleNamespace(fixtures=ROOT / "eval/native-smoke/fixtures.jsonl", module_root=ROOT,
        output=tmp_path / "run", base_url="http://offline.invalid/v1", model="offline-model",
        api_key_env="UNSET_AUDIT_API_KEY", cf_id_env="UNSET_AUDIT_CF_ID", cf_secret_env="UNSET_AUDIT_CF_SECRET",
        transport_kind=transport_kind, writer_prefill="<think></think", prefill_mode="complete",
        count_input_tokens=count_tokens, input_token_limit=None,
        batch_size=8, batch_wait_ms=5, timeout_seconds=10, context_window=16384,
        concurrency=8, max_output_tokens=2048)
    result = asyncio.run(execute(args, transport=httpx.MockTransport(handler)))

    assert result["recorded_cases"] == 8
    assert result["model_calls"] == 8
    assert result["http_requests"] == len(requests)
    if transport_kind == "native":
        assert len(requests) == 16
        assert result["batch_http"] is None
    else:
        assert len(requests) == int(not fail_after_start) + 8 * int(count_tokens)
        assert result["batch_http"]["unique_batches"] == 1
        assert result["batch_http"]["never_attempted"] == int(fail_after_start)
        assert result["batch_http"]["unknown"] == 0
        assert result["input_token_count_http_requests"] == 8 * int(count_tokens)
        assert result["generation_batch_http_requests"] == int(not fail_after_start)
        assert len(list((args.output / "token-http").glob("*.token_count_started.json"))) == 8 * int(count_tokens)
        assert len(list((args.output / "token-http").glob("*.token_count_completed.json"))) == 8 * int(count_tokens)
        if fail_after_start:
            assert result["status_counts"] == {"transport_error": 8}
