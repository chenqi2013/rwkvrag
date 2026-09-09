from llamaindex_retrieval.lexical_index import LexicalResult
from llamaindex_retrieval.retrieval_candidates import fuse_document_rrf


def _result(node_id: str, document_id: str, score: float) -> LexicalResult:
    return LexicalResult(
        node_id=node_id,
        document_id=document_id,
        text=node_id,
        metadata={"document_id": document_id},
        score=score,
    )


def test_document_rrf_counts_each_document_once_per_query() -> None:
    fused, trace = fuse_document_rrf(
        [
            [_result("a-1", "a", 1.0), _result("a-2", "a", 0.9), _result("b", "b", 0.8)],
            [_result("b", "b", 1.0), _result("a-1", "a", 0.8)],
        ],
        queries=["q1", "q2"],
    )

    assert {item.document_id for item in fused} == {"a", "b"}
    decisions = {item["document_id"]: item for item in trace["decisions"]}
    assert len(decisions["a"]["occurrences"]) == 3
    assert sum(
        item["action"] == "duplicate_document_in_query"
        for item in decisions["a"]["occurrences"]
    ) == 1
    assert trace["raw_candidate_count"] == 5
    assert trace["document_candidate_count"] == 2


def test_document_rrf_limit_is_only_a_selection_decision() -> None:
    fused, trace = fuse_document_rrf(
        [[_result("a", "a", 1.0), _result("b", "b", 0.9)]],
        limit=1,
    )

    assert len(fused) == 2
    assert [item["action"] for item in trace["decisions"]] == ["selected", "candidate_limit"]
