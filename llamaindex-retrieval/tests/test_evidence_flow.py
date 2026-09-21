import copy
from time import monotonic
from llamaindex_retrieval.evidence_flow import evidence_flow
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline
from llamaindex_retrieval.schemas import SourceItem


def test_no_materials_is_not_a_claim_of_no_answer():
    flow = evidence_flow({"input_source_ids": [], "writer_source_ids": []})
    assert flow["state"] == "no_input_materials"
    assert flow["source_absence_established"] is False


def test_successful_empty_selection_does_not_imply_source_omission():
    flow = evidence_flow({"input_source_ids": ["s"], "writer_source_ids": [], "failures": []})
    assert flow["state"] == "no_selected_evidence"
    assert flow["failed_node_count"] == 0
    assert flow["source_absence_established"] is False


def test_failure_and_budget_omissions_coexist_with_valid_evidence_without_mutation():
    graph = {"input_source_ids": ["s", "t"], "writer_source_ids": ["s"],
        "facts": [{"id": "a", "verification": {"verdict": "supported"}},
                  {"id": "b", "verification": {"verdict": "supported"}},
                  {"id": "c", "verification": {"verdict": "mismatch"}}],
        "cells": [{"fact_ids": ["a"], "execution_status": "completed"}, {"execution_status": "failed"}],
        "failures": [{"purpose": "funnel_fact:s:O1:F1", "status": "length"}],
        "call_budget": {"unexamined_jobs": [{"source_index": 1}]}}
    before = copy.deepcopy(graph)
    flow = evidence_flow(graph)
    assert flow["state"] == "selected_evidence_present"
    assert flow["verified_unselected_fact_ids"] == ["b"]
    assert flow["unexamined_job_count"] == flow["failed_node_count"] == flow["failed_cell_count"] == 1
    assert flow["semantic_support_verified"] is False
    assert graph == before


def test_missing_legacy_trace_and_foreign_source_are_never_reported_as_no_materials():
    assert evidence_flow({})["state"] == "unrecorded"
    assert evidence_flow({"input_source_ids": ["s"], "writer_source_ids": ["x"]})["state"] == "inconsistent_trace"


def test_api_retains_input_provenance_after_final_evidence_filter_and_keeps_raw_answer():
    source = SourceItem(id="s", document_id="d", source="test", title="record", snippet="功率18W", score=1, metadata={})
    graph = {"protocol": "typed-evidence-funnel-v10", "input_source_ids": ["s"], "writer_source_ids": [],
        "failures": [{"purpose": "funnel_fact:s:O1:F1", "status": "invalid_response"}]}
    raw = "原文没有记载功率。"
    event = {"stage": "writer", "status": "completed", "raw_text": raw, "funnel": graph}
    original = copy.deepcopy(event)
    response = RWKVPipeline(Settings(), None, model=object())._response(raw, [source], {}, [event], "completed", monotonic())
    assert response.sources == []
    assert response.retrieval["evidence_flow"]["input_source_count"] == 1
    assert response.generation["evidence_flow"]["state"] == "no_selected_evidence"
    assert response.answer == response.generation["raw_model_answer"] == raw
    assert response.generation["status"] == "funnel_partial_failure"
    assert event == original
