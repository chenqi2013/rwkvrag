import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("defect_audit", Path(__file__).with_name("audit.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_literal_value_cannot_become_semantic_gold():
    call = {"messages": [{"content": '说明\n{"object":"甲","source_title":"说明书","field":{"name":"价格"},"evidence":{"E1":"甲重18kg。"}}'}],
            "raw_text": '{"quote":"甲重18kg。","value":"18kg","source_scope":"说明书"}'}
    result = audit.scope_observations(call)
    assert result["quote_is_literal"] and result["value_is_literal_in_quote"]
    assert result["scope_equals_title"]
    assert result["semantic_correctness"] == "not_inferred"


def test_repeated_failures_are_one_job_but_not_deleted():
    base = {"case": 0, "error": "scope", "material_fingerprint": "source-hash", "input": {},
            "prompt_sha256": "p1", "call": {"call_id": "a", "raw_text": "bad"}}
    repeated = dict(base, call={"call_id": "b", "raw_text": "another bad output"})
    other_field = dict(base, prompt_sha256="p2", call={"call_id": "c", "raw_text": "bad field"})
    job, = audit.group_jobs([base, repeated, other_field])
    assert len(job["occurrences"]) == 3
    assert len(job["distinct_prompts"]) == 2
    assert job["gold_target"] is None and job["diagnostic_only"]
    assert all(p["gold_target"] is None for p in job["distinct_prompts"])


def test_distinct_materials_never_merge_on_error_label():
    base = {"case": 0, "error": "quote", "material_fingerprint": "a", "input": {},
            "prompt_sha256": "p1", "call": {"call_id": "a", "raw_text": "bad"}}
    assert len(audit.group_jobs([base, dict(base, material_fingerprint="b")])) == 2


def test_scheduler_failure_without_model_trace_is_not_training_sample():
    assert audit.group_jobs([{"call": None}]) == []
    assert audit.family("") == "unclassified_execution"


def test_bad_payload_is_retained_as_unparsed():
    assert audit.payload({"messages": [{"content": "not json"}]}) == {}
    assert audit.scope_observations({"raw_text": "not json", "messages": []}) == {"inspection_error": True}


def test_missing_questions_cannot_be_published_as_full_audit(tmp_path):
    with pytest.raises(ValueError, match="36 frozen questions"):
        audit.audit(tmp_path)
