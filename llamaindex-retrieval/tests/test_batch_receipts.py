from copy import deepcopy
import json

import pytest

from llamaindex_retrieval.batch_receipts import summarize_batch_http, summarize_token_http




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
