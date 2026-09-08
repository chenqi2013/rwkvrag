from dataclasses import replace

import pytest

from llamaindex_retrieval.lexical_index import LexicalResult
from llamaindex_retrieval.rwkv_pipeline import fuse_chunks


def hit(identity, document="same-document"):
    return LexicalResult(node_id=identity, document_id=document, text=f"Original {identity}",
                         metadata={"title": document}, score=1)


def test_query_rotation_serves_each_independent_query_before_shared_lower_matches():
    a, b, common1, common2 = [hit(name) for name in ("a", "b", "common1", "common2")]
    groups = [[a, common1, common2], [b, common1, common2]]
    rrf = fuse_chunks(groups)
    rotated = fuse_chunks(groups, order="query_round_robin")
    assert [row.id for row in rrf[:2]] == ["common1", "common2"]
    assert [row.id for row in rotated] == ["a", "b", "common1", "common2"]
    # Scheduling changes order only. Evidence text, identities and scores survive.
    assert {row.id: row.model_dump() for row in rrf} == {row.id: row.model_dump() for row in rotated}


def test_rotation_has_no_document_limit_and_handles_empty_or_duplicate_queues():
    rows = [hit(str(i)) for i in range(6)]
    result = fuse_chunks([[], [rows[0], rows[0], *rows[1:]], rows], order="query_round_robin")
    assert len(result) == 6
    assert {row.document_id for row in result} == {"same-document"}
    assert {row.id for row in result} == {row.node_id for row in rows}
    assert fuse_chunks([], order="query_round_robin") == []


def test_rotation_still_rejects_a_conflicting_chunk_before_scheduling():
    row = hit("same")
    with pytest.raises(ValueError, match="inconsistent"):
        fuse_chunks([[row], [replace(row, text="different")]], order="query_round_robin")


def test_unknown_order_cannot_silently_change_retrieval():
    with pytest.raises(ValueError):
        fuse_chunks([[hit("a")]], order="unregistered")
