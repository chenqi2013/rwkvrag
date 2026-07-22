import asyncio

from .config import Settings
from .lexical_index import LexicalIndex, LexicalResult
from .schemas import SearchRequest, SearchResponse, SourceItem


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
        min_score = (
            request.min_score
            if request.min_score is not None
            else self.settings.min_relevance_score
        )
        filtered = self._select_results(results, top_k, min_score)
        return SearchResponse(
            results=[self._source_item(result) for result in filtered],
            retrieval={
                "algorithm": "OpenSearch BM25",
                "mode": "bm25+keyword",
                "keyword_fields": ["body", "title", "tags"],
                "candidate_k": candidate_k,
                "top_k": top_k,
                "returned": len(filtered),
                "max_chunks_per_document": self.settings.max_chunks_per_document,
                "knowledge_base_id": request.knowledge_base_id,
            },
        )

    def _select_results(
        self,
        results: list[LexicalResult],
        top_k: int,
        min_score: float,
    ) -> list[LexicalResult]:
        if not results:
            return []
        top_score = max(float(result.score or 0) for result in results)
        relative_floor = max(0, top_score) * self.settings.relative_score_threshold
        score_floor = max(min_score, relative_floor)
        document_counts: dict[str, int] = {}
        selected: list[LexicalResult] = []
        for result in results:
            if float(result.score or 0) < score_floor:
                continue
            document_id = result.document_id or result.node_id
            if document_counts.get(document_id, 0) >= self.settings.max_chunks_per_document:
                continue
            document_counts[document_id] = document_counts.get(document_id, 0) + 1
            selected.append(result)
            if len(selected) >= top_k:
                break
        return selected

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
