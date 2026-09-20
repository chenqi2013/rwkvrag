"""A successful HTTP response cannot by itself pass a semantic quality gate."""

import argparse
import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from llamaindex_retrieval.quality_gate import (
    audit, automatic_checks, digest, encode, load_run, read_json, semantic_checks,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("quality_smoke_runner",
                                              ROOT / "eval/native-smoke/run_smoke.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def run(tmp_path, request):
    canonical = getattr(request, "param", None) == "canonical"
    async def handler(request):
        if canonical:
            payload = json.loads(request.content)
            assert all(text.endswith("Assistant: <think></think>\n")
                       for text in payload["contents"])
            assert payload["state_id"] == "fixture-writer"
            return httpx.Response(200, json={
                "object": "chat.completion", "model": "offline-model",
                "choices": [{"index": i, "message": {"role": "assistant", "content": "无依据"},
                             "finish_reason": "stop"}
                            for i in range(len(payload["contents"]))]})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 4, "tokens": [0, 1, 2, 3],
                                            "max_model_len": 16384})
        return httpx.Response(200, json={
            "choices": [{"text": "></think>模拟😀答案，必须独立复核。", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 10, "total_tokens": 14}})

    args = argparse.Namespace(
        fixtures=ROOT / ("eval/writer-p0-20260916/holdout.jsonl"
                         if getattr(request, "param", None) == "holdout"
                         else "eval/native-smoke/fixtures.jsonl"), module_root=ROOT,
        output=tmp_path / "run", base_url="http://offline.invalid/v1", model="offline-model",
        api_key_env="QUALITY_TEST_NO_API_KEY", timeout_seconds=5,
        context_window=16384, concurrency=2, max_output_tokens=64,
    )
    if canonical:
        args.transport_kind = "rwkvos_batch"
        args.writer_prefill = "<think></think"
        args.writer_prompt_protocol = "evidence_first"
        args.writer_transport_protocol = "rwkv_g1j_no_think_v1"
        args.writer_state_id = "fixture-writer"
    asyncio.run(runner.execute(args, transport=httpx.MockTransport(handler)))
    return args.output


def filled_review(run, tmp_path):
    _, template = audit(run)
    template.update(reviewer="synthetic test judgments, not a model quality result",
                    reviewed_at="2026-09-16")
    for row in template["cases"]:
        row.update(decision_correct=True, no_unsupported_claims=True,
                   no_uncontrolled_repetition=True, notes="Synthetic gate test only.")
        for fact in row["facts"]:
            fact.update(correct=True, citation_supported=True)
    path = tmp_path / "review.json"
    path.write_bytes(encode(template))
    return path


def test_completed_calls_are_pending_until_every_fact_is_reviewed(run, tmp_path):
    report, template = audit(run)
    assert report["planned_cases"] == report["pending_cases"] == 8
    assert report["decision"] == "pending_review" and report["passed_cases"] == 0
    assert report["metrics"]["automatic"]["execution_completed"] == {
        "passed": 8, "assessed": 8, "total": 8}
    assert report["metrics"]["semantic"] == {}
    assert report["model_calls"] == 8 and report["full_rag_verified"] is False
    assert all(row["decision_correct"] is None for row in template["cases"])
    review = filled_review(run, tmp_path)
    reviewed = audit(run, review)[0]
    assert reviewed["decision"] == "pass"
    assert reviewed["metrics"]["semantic"]["facts_correct"] == {
        "passed": 7, "assessed": 7, "total": 8}
    content = read_json(review)
    content["cases"][0]["facts"][0]["citation_supported"] = False
    review.write_bytes(encode(content))
    report, _ = audit(run, review)
    assert report["decision"] == "fail" and report["failed_cases"] == 1


@pytest.mark.parametrize("change", ["stale", "omit_case", "omit_fact", "duplicate_fact",
                                   "no_author", "no_notes", "string_boolean"])
def test_reviews_cannot_change_denominators_or_review_different_outputs(run, tmp_path, change):
    path = filled_review(run, tmp_path)
    review = read_json(path)
    first = review["cases"][0]
    if change == "stale":
        first["response_sha256"] = "wrong"
    elif change == "omit_case":
        review["cases"].pop()
    elif change == "omit_fact":
        first["facts"].pop()
    elif change == "duplicate_fact":
        first["facts"].append(first["facts"][0])
    elif change == "no_author":
        review["reviewer"] = ""
    elif change == "no_notes":
        first["notes"] = ""
    else:
        first["decision_correct"] = "true"
    path.write_bytes(encode(review))
    with pytest.raises(ValueError):
        audit(run, path)


def test_partial_semantic_review_stays_pending(run, tmp_path):
    path = filled_review(run, tmp_path)
    review = read_json(path)
    review["cases"][0]["facts"][0]["correct"] = None
    path.write_bytes(encode(review))
    report, _ = audit(run, path)
    assert report["reviewed_cases"] == 7 and report["pending_cases"] == 1
    assert report["decision"] == "pending_review"


@pytest.mark.parametrize("relative", ["frozen/fixtures.jsonl", "frozen/run_smoke.py",
                                     "results.json", "completed/smoke_001.json"])
def test_changed_artifacts_are_rejected(run, relative):
    path = run / relative
    path.write_bytes(b"{}\n")
    with pytest.raises((ValueError, KeyError)):
        audit(run)


def test_automatic_checks_do_not_trust_reported_citation_audit_or_completion(run):
    _, _, records, _ = load_run(run)
    record = deepcopy(records[0])
    before = deepcopy(record)
    assert all(automatic_checks(record, "offline-model").values())
    assert record == before
    gen = record["response"]["generation"]
    gen["status"] = "planner_partial_failure"
    assert not automatic_checks(record, "offline-model")["execution_completed"]
    gen["status"] = "completed"
    gen["planner_fallback"] = "original_question"
    assert not automatic_checks(record, "offline-model")["execution_completed"]
    for text in ("错误引用[资料 0]", "错误引用[资料 999]", "模板[资料 N]"):
        record["response"]["answer"] = gen["raw_model_answer"] = text
        gen["answer_span"] = [0, len(text)]
        gen["citation_audit"] = {"unknown_label_ids": [], "semantic_support_verified": True}
        assert not automatic_checks(record, "offline-model")["citation_labels_valid"]
    gen["answer_span"] = [True, 4]
    assert not automatic_checks(record, "offline-model")["answer_contract_valid"]


def test_model_failure_keeps_its_denominator_and_cannot_be_approved_semantically(run, tmp_path):
    manifest, fixtures, records, _ = load_run(run)
    record = deepcopy(records[0])
    record.update(status="timeout", response=None)
    checks = automatic_checks(record, manifest["configuration"]["model"])
    assert not any(checks.values())
    review = read_json(filled_review(run, tmp_path))["cases"][0]
    assert all(semantic_checks(review, fixtures[0], records[0]).values())
    assert digest(records[0]["response"]) == records[0]["response_sha256"]


def test_duplicate_review_keys_are_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"decision_correct":false,"decision_correct":true}')
    with pytest.raises(ValueError, match="duplicate"):
        read_json(path)


@pytest.mark.parametrize("run", ["holdout"], indirect=True)
def test_holdout_is_labeled_separately_and_cannot_be_pooled_with_development(run, tmp_path):
    report, _ = audit(run)
    assert report["scope"] == "fixed_material_holdout_only"
    assert report["evaluation_split"] == "frozen_holdout"
    assert report["full_rag_verified"] is False
    rows = [json.loads(line) for line in
            (run / "frozen/fixtures.jsonl").read_text().splitlines()]
    rows[0]["phase"] = "development_smoke_not_blind"
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_bytes(b"".join(encode(row) + b"\n" for row in rows))
    with pytest.raises(ValueError, match="cannot be pooled"):
        runner.load_fixtures(mixed)


@pytest.mark.parametrize("run", ["canonical"], indirect=True)
def test_runner_preserves_canonical_writer_template_and_stage_state(run):
    manifest, _, records, _ = load_run(run)
    assert manifest["configuration"]["writer_transport_protocol"] == "rwkv_g1j_no_think_v1"
    assert manifest["configuration"]["writer_state_id"] == "fixture-writer"
    assert all(row["status"] == "completed" for row in records)
    for row in records:
        call = row["model_traces"][0]
        assert call["prompt"].endswith("Assistant: <think></think>\n")
        assert call["state_id"] == "fixture-writer"
        assert call["writer_prompt_protocol"] == "rwkv_g1j_no_think_v1"
