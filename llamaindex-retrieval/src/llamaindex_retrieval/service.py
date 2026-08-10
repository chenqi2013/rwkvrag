import asyncio
from collections import OrderedDict
from time import monotonic

from .config import Settings
from .evidence_utils import clean_evidence_text, direct_evidence_answer
from .generation import EvidenceAnswerGenerator
from .lexical_index import (
    LexicalIndex,
    LexicalResult,
    intent_content_types,
    lexical_tokens,
    normalize_query_text,
    query_tokens,
)
from .schemas import AskResponse, SearchRequest, SearchResponse, SourceItem

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
_STRUCTURE_QUESTION_WORDS = {
    "什么",
    "什么时候",
    "时候",
    "何时",
    "哪里",
    "哪儿",
    "哪个",
    "哪些",
    "怎么",
    "如何",
    "多少",
    "几个",
}
_STRUCTURE_TYPE_BONUS = {
    "table_summary": 2.4,
    "table": 1.8,
    "list": 1.4,
    "key_value": 0.8,
    "timeline": 0.4,
    "prose": 0.0,
}
_LIST_ANSWER_HINTS = (
    "车站列表",
    "站点列表",
    "站名列表",
    "站名/",
    "列表",
    "全部车站",
    "全部站点",
)
_LIST_TOPIC_HINTS = ("车站", "站点", "站名", "站")
_EXPLANATORY_SECTION_HINTS = ("问题", "問題", "歷史", "历史", "命名", "更名", "工程", "續建", "续建")
_TRANSFER_HINTS = ("转乘", "轉乘", "换乘", "換乘", "线网转乘", "線網轉乘")
_ANSWER_CACHE_TTL_SECONDS = 300.0
_ANSWER_CACHE_MAX_ENTRIES = 128


def _is_station_list_question(question: str) -> bool:
    return any(marker in question for marker in ("有哪些站", "哪些站", "车站", "站点", "站名"))


class SearchService:
    def __init__(
        self,
        settings: Settings,
        index: LexicalIndex,
        generator: EvidenceAnswerGenerator | None = None,
    ) -> None:
        self.settings = settings
        self.index = index
        self.generator = generator or EvidenceAnswerGenerator(settings)
        self._answer_cache: OrderedDict[tuple[object, ...], tuple[float, str]] = OrderedDict()

    async def search(self, request: SearchRequest) -> SearchResponse:
        top_k = min(request.top_k or self.settings.default_top_k, self.settings.max_top_k)
        candidate_k = max(request.candidate_k or self.settings.candidate_k, top_k)
        normalized_question = normalize_query_text(request.question)
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
                "normalized_question": normalized_question,
                "query_normalized": normalized_question != request.question,
            },
        )

    async def ask(self, request: SearchRequest) -> AskResponse:
        response = await self.search(request)
        question = str(response.retrieval.get("normalized_question") or request.question)
        assessment = self.generator.assess_evidence(question, response.results)
        cache_key = self._answer_cache_key(question, response)
        answer = self._get_cached_answer(cache_key)
        cache_hit = answer is not None
        answer_strategy = "cache" if cache_hit else "model"
        if answer is None:
            answer = direct_evidence_answer(question, response.results) if assessment.grounded else None
            if answer is not None:
                answer_strategy = "direct_extract"
            else:
                answer = await self.generator.generate(question, response.results)
            if assessment.grounded and answer not in {
                "未检索到可用于回答该问题的资料。",
                "根据检索到的资料，无法确定。",
            }:
                self._store_cached_answer(cache_key, answer)
        model_name = await self.generator.current_model()
        return AskResponse(
            answer=answer,
            sources=response.results,
            retrieval=response.retrieval,
            generation={
                "model": model_name,
                "endpoint": self.settings.generation_base_url,
                "evidence_count": len(response.results),
                "evidence_grounded": assessment.grounded,
                "question_terms": sorted(assessment.question_terms),
                "matched_evidence_terms": sorted(assessment.matched_terms),
                "matched_specific_terms": sorted(assessment.matched_specific_terms),
                "evidence_anchors": sorted(assessment.anchors),
                "matched_evidence_anchors": sorted(assessment.matched_anchors),
                "citation_required": True,
                "cache_hit": cache_hit,
                "answer_strategy": answer_strategy,
                "blocked_reason": "insufficient_evidence" if not assessment.grounded else None,
            },
        )

    @staticmethod
    def _answer_cache_key(question: str, response: SearchResponse) -> tuple[object, ...]:
        evidence = tuple(
            (source.id, round(source.score, 6), source.snippet[:256])
            for source in response.results
        )
        return question, response.retrieval.get("knowledge_base_id"), evidence

    def _get_cached_answer(self, key: tuple[object, ...]) -> str | None:
        cached = self._answer_cache.get(key)
        if cached is None:
            return None
        created_at, answer = cached
        if monotonic() - created_at > _ANSWER_CACHE_TTL_SECONDS:
            self._answer_cache.pop(key, None)
            return None
        self._answer_cache.move_to_end(key)
        return answer

    def _store_cached_answer(self, key: tuple[object, ...], answer: str) -> None:
        self._answer_cache[key] = (monotonic(), answer)
        self._answer_cache.move_to_end(key)
        while len(self._answer_cache) > _ANSWER_CACHE_MAX_ENTRIES:
            self._answer_cache.popitem(last=False)

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
        document_candidates = await self._document_structure_candidates(
            question,
            document_id=top_document,
            content_types=content_types,
            knowledge_base_id=knowledge_base_id,
        )
        seen_candidate_ids = {result.node_id for result in candidates}
        if document_candidates:
            same_document = [
                *same_document,
                *(
                    result
                    for result in document_candidates
                    if result.node_id not in seen_candidate_ids
                    and result.metadata.get("parent_id")
                    and result.metadata.get("content_type") in content_types
                ),
            ]
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

    async def _document_structure_candidates(
        self,
        question: str,
        *,
        document_id: str,
        content_types: set[str],
        knowledge_base_id: str | None,
    ) -> list[LexicalResult]:
        lookup = getattr(self.index, "document_structure_candidates", None)
        if lookup is None or not document_id:
            return []
        return await asyncio.to_thread(
            lookup,
            question,
            document_id=document_id,
            content_types=content_types,
            knowledge_base_id=knowledge_base_id,
            limit=max(self.settings.candidate_k, self.settings.list_query_max_chunks_per_document),
        )

    @staticmethod
    def _structure_relevance(question: str, result: LexicalResult) -> tuple[float, float]:
        metadata = result.metadata
        title_tokens = set(lexical_tokens(str(metadata.get("title") or "")))
        question_tokens = (
            set(query_tokens(question)) - title_tokens - _STRUCTURE_QUESTION_WORDS
        )
        context = " ".join(
            str(value or "")
            for value in (
                metadata.get("section"),
                metadata.get("content_type"),
                " ".join(str(item) for item in metadata.get("keywords") or []),
                result.text[:800],
            )
        )
        context_tokens = set(lexical_tokens(context)) - title_tokens
        overlap = len(question_tokens & context_tokens)
        coverage = overlap / max(1, len(question_tokens))
        content_type = str(metadata.get("content_type") or "prose")
        type_bonus = _STRUCTURE_TYPE_BONUS.get(content_type, 0.0)
        normalized_context = context.replace(" ", "")
        list_bonus = 0.0
        if any(marker in question for marker in _MULTI_EVIDENCE_MARKERS):
            if any(hint in normalized_context for hint in _LIST_ANSWER_HINTS):
                list_bonus += 2.4
            elif any(hint in normalized_context for hint in _LIST_TOPIC_HINTS):
                list_bonus += 0.8
            if _is_station_list_question(question):
                if "站名" in normalized_context and "列表" in normalized_context:
                    list_bonus += 3.0
                elif "站名/" in normalized_context:
                    list_bonus += 2.2
                if any(hint in normalized_context for hint in _TRANSFER_HINTS) and not any(
                    hint in question for hint in _TRANSFER_HINTS
                ):
                    list_bonus -= 2.6
            if any(hint in normalized_context for hint in _EXPLANATORY_SECTION_HINTS):
                list_bonus -= 1.0
        score = coverage * 3.0 + type_bonus + list_bonus
        return score, result.score

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
        content = clean_evidence_text(result.text)
        if full_answer:
            snippet = clean_evidence_text(full_answer)
        else:
            snippet = content if len(content) <= 900 else content[:900].rstrip() + "..."
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
