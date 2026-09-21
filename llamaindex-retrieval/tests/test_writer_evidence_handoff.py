import copy
import pytest
from llamaindex_retrieval.writer_evidence_handoff import with_evidence_handoff
from llamaindex_retrieval.evidence_flow import evidence_flow


def test_processing_failure_is_passed_without_editing_the_original_prompt_or_graph():
    graph = {"input_source_ids":["one","two"], "writer_source_ids":[],
        "failures":[{"purpose":"funnel_fact:one","status":"length"}]}
    before = copy.deepcopy(graph)
    prompt = "原始问题与上下文"
    revised = with_evidence_handoff(prompt, evidence_flow(graph))
    assert revised.startswith(prompt)
    assert '"input_source_count": 2' in revised and '"failed_node_count": 1' in revised
    assert "不证明原文没有答案" in revised
    assert graph == before


@pytest.mark.parametrize("graph", [{"input_source_ids":[],"writer_source_ids":[]},
    {"input_source_ids":["s"],"writer_source_ids":["s"]}, {}])
def test_real_empty_input_and_evidence_bearing_or_unknown_inputs_are_byte_identical(graph):
    assert with_evidence_handoff("原始输入😀", evidence_flow(graph)) == "原始输入😀"


def test_inconsistent_metadata_cannot_be_forwarded_as_a_confirmed_processing_loss():
    with pytest.raises(ValueError):
        with_evidence_handoff("q", {"protocol":"evidence-flow-v1","state":"no_selected_evidence", "input_source_count":0,"writer_source_count":0})
