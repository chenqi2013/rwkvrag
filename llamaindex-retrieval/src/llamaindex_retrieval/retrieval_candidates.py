"""Deterministic candidate fusion with complete provenance trace.

This module only organizes retrieval output. It never decides whether a
candidate answers the question; that decision remains with later evidence
extraction and generation stages.
"""

from dataclasses import replace
from typing import Any

from .lexical_index import LexicalResult


def fuse_document_rrf(
    result_groups: tuple[list[LexicalResult], ...] | list[list[LexicalResult]],
    *,
    queries: tuple[str, ...] | list[str] | None = None,
    limit: int | None = None,
    rrf_k: int = 60,
) -> tuple[list[LexicalResult], dict[str, Any]]:
    """Fuse query branches once per document and return an auditable trace."""

    scores: dict[str, float] = {}
    best: dict[str, LexicalResult] = {}
    first_seen: dict[str, int] = {}
    occurrences: dict[str, list[dict[str, Any]]] = {}
    sequence = 0
    for query_index, group in enumerate(result_groups):
        seen_documents: set[str] = set()
        for rank, result in enumerate(group, start=1):
            document_id = result.document_id or result.node_id
            occurrence = {
                "query_index": query_index,
                "query": queries[query_index] if queries and query_index < len(queries) else None,
                "rank": rank,
                "node_id": result.node_id,
                "document_id": document_id,
                "bm25_score": (
                    result.raw_score if result.raw_score is not None else result.score
                ),
                "normalized_branch_score": result.score,
                "action": "counted",
            }
            if document_id in seen_documents:
                occurrence["action"] = "duplicate_document_in_query"
                occurrences.setdefault(document_id, []).append(occurrence)
                continue
            seen_documents.add(document_id)
            occurrences.setdefault(document_id, []).append(occurrence)
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (rrf_k + rank)
            if document_id not in first_seen:
                first_seen[document_id] = sequence
                sequence += 1
            current = best.get(document_id)
            result_strength = result.raw_score if result.raw_score is not None else result.score
            current_strength = current.raw_score if current and current.raw_score is not None else (
                current.score if current else None
            )
            if current is None or result_strength > current_strength:
                best[document_id] = result

    ranked_ids = sorted(
        scores,
        key=lambda document_id: (scores[document_id], -first_seen[document_id]),
        reverse=True,
    )
    top_score = max((scores[document_id] for document_id in ranked_ids), default=1.0)
    fused = [
        replace(
            best[document_id],
            score=scores[document_id] / top_score if top_score else 0.0,
        )
        for document_id in ranked_ids
    ]
    selected_ids = {
        item.document_id or item.node_id
        for item in (fused[:limit] if limit is not None else fused)
    }
    decisions = []
    raw_candidates = []
    for document_occurrences in occurrences.values():
        raw_candidates.extend(document_occurrences)
    raw_candidates.sort(key=lambda item: (item["query_index"], item["rank"]))
    for rank, document_id in enumerate(ranked_ids, start=1):
        decisions.append({
            "document_id": document_id,
            "node_id": best[document_id].node_id,
            "rrf_score": scores[document_id],
            "best_branch_score": best[document_id].score,
            "normalized_rrf_score": scores[document_id] / top_score if top_score else 0.0,
            "rank": rank,
            "action": "selected" if document_id in selected_ids else "candidate_limit",
            "occurrences": occurrences.get(document_id, []),
        })
    return fused, {
        "algorithm": "document_rrf",
        "k": rrf_k,
        "query_count": len(result_groups),
        "queries": list(queries or ()),
        "raw_candidate_count": sum(len(group) for group in result_groups),
        "document_candidate_count": len(ranked_ids),
        "candidate_limit": limit,
        "raw_candidates": raw_candidates,
        "decisions": decisions,
    }
