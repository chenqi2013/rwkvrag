import asyncio

from .config import Settings
from .lexical_index import (
    LexicalIndex,
    LexicalResult,
    intent_content_types,
    lexical_tokens,
    query_tokens,
)
from .schemas import SearchRequest, SearchResponse, SourceItem

_MULTI_EVIDENCE_MARKERS = (
    "哪些",
    "有哪些",
    "列表",
    "全部",
    "所有",
    "分别",
    "几个",
    "多少",
)


class SearchService:
    def __init__(self, settings: Settings, index: LexicalIndex) -> None:
        self.settings = settings
        self.index = index

    async def search(self, request: SearchRequest) -> SearchResponse:
        top_k = min(request.top_k or self.settings.default_top_k, self.settings.max_top_k)
        candidate_k = max(request.candidate_k or self.settings.candidate_k, top_k)
        results = await asyncio.to_thread(
            self.index.search,
            request.question,
            candidate_k=candidate_k,
            knowledge_base_id=request.knowledge_base_id,
        )
        results, structure_expanded = await self._expand_structured_results(
            request.question,
            results,
            knowledge_base_id=request.knowledge_base_id,
            top_k=top_k,
        )
        min_score = (
            request.min_score
            if request.min_score is not None
            else self.settings.min_relevance_score
        )
        max_chunks_per_document = self._max_chunks_per_document(request.question)
        filtered = self._select_results(
            results,
            top_k,
            min_score,
            max_chunks_per_document=max_chunks_per_document,
        )
        return SearchResponse(
            results=[self._source_item(result) for result in filtered],
            retrieval={
                "algorithm": "OpenSearch BM25",
                "mode": "bm25+keyword",
                "keyword_fields": ["body", "title", "tags", "section", "structure"],
                "candidate_k": candidate_k,
                "top_k": top_k,
                "returned": len(filtered),
                "max_chunks_per_document": max_chunks_per_document,
                "multi_evidence": max_chunks_per_document > self.settings.max_chunks_per_document,
                "structure_expanded": structure_expanded,
                "knowledge_base_id": request.knowledge_base_id,
            },
        )

    async def _expand_structured_results(
        self,
        question: str,
        results: list[LexicalResult],
        *,
        knowledge_base_id: str | None,
        top_k: int,
    ) -> tuple[list[LexicalResult], bool]:
        content_types = set(intent_content_types(question))
        if not content_types or not results:
            return results, False
        top_document = results[0].document_id
        candidates = [
            result
            for result in results
            if result.metadata.get("parent_id")
            and result.metadata.get("content_type") in content_types
        ]
        same_document = [result for result in candidates if result.document_id == top_document]
        anchor_pool = same_document or candidates
        anchor = max(
            anchor_pool,
            key=lambda result: self._structure_relevance(question, result),
            default=None,
        )
        if anchor is None:
            return results, False
        expanded = await asyncio.to_thread(
            self.index.structure_chunks,
            str(anchor.metadata["parent_id"]),
            knowledge_base_id=knowledge_base_id,
            limit=max(top_k, self.settings.list_query_max_chunks_per_document),
            score=anchor.score,
        )
        if not expanded:
            return results, False
        seen = {result.node_id for result in expanded}
        merged = [*expanded, *(result for result in results if result.node_id not in seen)]
        return merged, True

    @staticmethod
    def _structure_relevance(question: str, result: LexicalResult) -> tuple[float, float]:
        title_tokens = set(lexical_tokens(str(result.metadata.get("title") or "")))
        question_tokens = set(query_tokens(question)) - title_tokens
        section_tokens = set(lexical_tokens(str(result.metadata.get("section") or "")))
        section_specific = section_tokens - title_tokens
        overlap = len(question_tokens & section_specific)
        coverage = overlap / max(1, len(section_specific))
        return coverage, result.score

    def _select_results(
        self,
        results: list[LexicalResult],
        top_k: int,
        min_score: float,
        max_chunks_per_document: int | None = None,
    ) -> list[LexicalResult]:
        if not results:
            return []
        top_score = max(float(result.score or 0) for result in results)
        relative_floor = max(0, top_score) * self.settings.relative_score_threshold
        score_floor = max(min_score, relative_floor)
        document_limit = max_chunks_per_document or self.settings.max_chunks_per_document
        document_counts: dict[str, int] = {}
        selected: list[LexicalResult] = []
        for result in results:
            if float(result.score or 0) < score_floor:
                continue
            document_id = result.document_id or result.node_id
            if document_counts.get(document_id, 0) >= document_limit:
                continue
            document_counts[document_id] = document_counts.get(document_id, 0) + 1
            selected.append(result)
            if len(selected) >= top_k:
                break
        return selected

    def _max_chunks_per_document(self, question: str) -> int:
        if any(marker in question for marker in _MULTI_EVIDENCE_MARKERS):
            return max(
                self.settings.max_chunks_per_document,
                self.settings.list_query_max_chunks_per_document,
            )
        return self.settings.max_chunks_per_document

    @staticmethod
    def _source_item(result: LexicalResult) -> SourceItem:
        metadata = dict(result.metadata)
        full_answer = str(metadata.pop("full_answer", "")).strip()
        title = str(metadata.pop("title", ""))
        source = str(metadata.pop("source", ""))
        uri = metadata.pop("uri", None)
        content = result.text.strip()
        snippet = full_answer or (content if len(content) <= 900 else content[:900].rstrip() + "...")
        return SourceItem(
            id=result.node_id,
            document_id=result.document_id,
            source=source,
            title=title,
            uri=str(uri) if uri else None,
            score=float(result.score),
            snippet=snippet,
            metadata=metadata,
        )
