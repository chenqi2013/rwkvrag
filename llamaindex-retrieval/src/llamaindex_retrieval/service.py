import asyncio
import re
from collections import OrderedDict
from dataclasses import replace
from hashlib import sha256
from time import monotonic

from .active_retrieval import ActiveRetrievalAgent
from .config import Settings
from .document_reranking import (
    DocumentRerankResult,
    LanguageModelDocumentReranker,
)
from .evidence_utils import (
    agent_evidence_answer,
    birthplace_evidence_answer,
    cause_evidence_answer,
    clean_evidence_text,
    coordinated_time_evidence_answer,
    definition_evidence_answer,
    direct_evidence_answer,
    location_evidence_answer,
    list_evidence_answer,
    ordinal_evidence_answer,
    structured_list_answer,
    time_evidence_answer,
)
from .generation import AnswerGenerationError, EvidenceAnswerGenerator, GenerationResult
from .evidence_extraction import (
    EvidenceSpan,
    EvidenceExtractionResult,
    LanguageModelEvidenceExtractor,
)
from .evidence_quality import is_repetitive_garbage
from .failure_diagnosis import diagnose_failure
from .evidence_gate import (
    document_aliases,
    evaluate_answer_support,
    evaluate_evidence_gate,
    repair_answer_citations,
    title_matches_subject,
    title_matches_subject_event,
    title_matches_subject_topic,
)
from .lexical_index import (
    LexicalIndex,
    LexicalResult,
    intent_content_types,
    lexical_tokens,
    normalize_search_text,
    query_tokens,
)
from .schemas import AskResponse, SearchRequest, SearchResponse, SourceItem
from .semantic_query_planning import LanguageModelQueryPlanner, QueryPlanningResult
from .qa_analysis import (
    ambiguity_candidates,
    analyze_question,
    remove_unsupported_number_sentences,
    validate_grounding,
    validate_list_answer,
)
from .query_planning import QueryPlan, TaskField, build_query_plan

_MULTI_EVIDENCE_MARKERS: tuple[str, ...] = ()
_STRUCTURE_QUESTION_WORDS: set[str] = set()
_SCOPE_TOKENS: set[str] = set()
_STRUCTURE_TYPE_BONUS: dict[str, float] = {}
_LIST_ANSWER_HINTS: tuple[str, ...] = ()
_LIST_TOPIC_HINTS: tuple[str, ...] = ()
_EXPLANATORY_SECTION_HINTS: tuple[str, ...] = ()
_TRANSFER_HINTS: tuple[str, ...] = ()
_STATION_STRUCTURE_NOISE: tuple[str, ...] = ()
_ANSWER_CACHE_TTL_SECONDS = 300.0
_ANSWER_CACHE_MAX_ENTRIES = 128
_ASK_MIN_EVIDENCE_TOP_K = 5
_CAUSE_CONTEXT_MAX_CHUNKS = 6
_CITATION_INDEX_PATTERN = re.compile(r"\[资料\s*([1-9]\d*)\]")
_REFUSAL_ANSWERS = {
    "未检索到可用于回答该问题的资料。",
    "根据检索到的资料，无法确定。",
}
_EMPTY_ANSWER_SHELL = re.compile(
    r"(?:原因|因素|项目|条目|内容)(?:包括|有|是)?[：:]?\s*"
    r"(?:\[资料\s*\d+\])?[。.]?$"
)
_LEAD_FACT_MARKERS: tuple[str, ...] = ()
_CHRONOLOGICAL_LIST_MARKERS: tuple[str, ...] = ()
_EXHAUSTIVE_LIST_MARKERS: tuple[str, ...] = ()
_ROUTE_ENDPOINT_PATTERN = None


def _is_station_list_question(question: str) -> bool:
    return False


def _core_subject_tokens(subject: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in lexical_tokens(subject)
        if token not in _SCOPE_TOKENS and len(token) >= 2
    )


def _answer_covers_subject_topics(answer: str, subject: str) -> bool:
    normalized_answer = normalize_search_text(answer).replace(" ", "")
    core_tokens = _core_subject_tokens(subject)
    if len(core_tokens) < 2:
        return True
    return all(
        any(
            normalize_search_text(variant).replace(" ", "") in normalized_answer
            for variant in query_tokens(token)
            if len(variant) >= 2
        )
        for token in core_tokens
    )


class SearchService:
    def __init__(
        self,
        settings: Settings,
        index: LexicalIndex,
        generator: EvidenceAnswerGenerator | None = None,
        query_planner: LanguageModelQueryPlanner | None = None,
        retrieval_agent: ActiveRetrievalAgent | None = None,
        evidence_extractor: LanguageModelEvidenceExtractor | None = None,
        document_reranker: LanguageModelDocumentReranker | None = None,
    ) -> None:
        self.settings = settings
        self.index = index
        self.generator = generator or EvidenceAnswerGenerator(settings)
        self.query_planner = query_planner
        self.retrieval_agent = retrieval_agent
        self.evidence_extractor = evidence_extractor
        self.document_reranker = document_reranker
        self._answer_cache: OrderedDict[tuple[object, ...], tuple[float, str]] = OrderedDict()

    async def search(
        self,
        request: SearchRequest,
        *,
        use_model_planner: bool = True,
        query_override: tuple[str, ...] | None = None,
    ) -> SearchResponse:
        search_started = monotonic()
        top_k = min(request.top_k or self.settings.default_top_k, self.settings.max_top_k)
        candidate_k = max(request.candidate_k or self.settings.candidate_k, top_k)
        fallback_plan = build_query_plan(request.question)
        planning_started = monotonic()
        if query_override:
            planning = QueryPlanningResult(
                replace(fallback_plan, queries=query_override),
                "deterministic_fallback",
                model_queries=query_override,
                error="active_tool_query",
            )
        elif use_model_planner:
            planning = await self._plan_queries(request.question, fallback_plan)
        else:
            planning = QueryPlanningResult(
                fallback_plan,
                "deterministic_fallback",
                error="model_planner_disabled_for_request",
            )
        planning_ms = self._elapsed_ms(planning_started)
        plan = planning.plan
        analysis = plan.analysis
        bm25_started = monotonic()
        results = await self._execute_query_plan(
            plan,
            candidate_k=candidate_k,
            knowledge_base_id=request.knowledge_base_id,
        )
        results = self._prioritize_topic_document(plan, results)
        bm25_ms = self._elapsed_ms(bm25_started)
        context_started = monotonic()
        results, structure_expanded = await self._expand_structured_results(
            plan.normalized_question,
            results,
            knowledge_base_id=request.knowledge_base_id,
            top_k=top_k,
        )
        relation_context_expanded = False
        if not structure_expanded:
            relation_plan = plan if plan.relations else fallback_plan
            results, relation_context_expanded = await self._expand_document_relation_context(
                relation_plan,
                results,
                knowledge_base_id=request.knowledge_base_id,
                top_k=top_k,
            )
        # Context expansion can add a more specific topic document after the
        # initial fusion. Reapply the same whole-document ordering before the
        # top-k cut so supplemental chunks cannot put a broad page back first.
        results = self._prioritize_topic_document(plan, results)
        min_score = (
            request.min_score
            if request.min_score is not None
            else self.settings.min_relevance_score
        )
        max_chunks_per_document = self._max_chunks_per_document(
            plan.normalized_question,
            top_k=top_k,
        )
        filtered = self._select_results(
            results,
            top_k,
            min_score,
            max_chunks_per_document=max_chunks_per_document,
        )
        section_context_expanded = False
        if plan.context_policy == "section":
            filtered, section_context_expanded = await self._expand_section_context(
                filtered,
                knowledge_base_id=request.knowledge_base_id,
                top_k=top_k,
            )
        filtered = await self._replace_with_lead_chunks(
            plan.normalized_question,
            filtered,
            knowledge_base_id=request.knowledge_base_id,
            intent=analysis.intent,
        )
        context_ms = self._elapsed_ms(context_started)
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
                "multi_evidence": (
                    max_chunks_per_document > self.settings.max_chunks_per_document
                    or section_context_expanded
                ),
                "structure_expanded": structure_expanded,
                "section_context_expanded": section_context_expanded,
                "document_relation_expanded": relation_context_expanded,
                "cause_context_expanded": (
                    section_context_expanded and analysis.intent == "cause"
                ),
                "knowledge_base_id": request.knowledge_base_id,
                "normalized_question": plan.normalized_question,
                "query_normalized": plan.normalized_question != request.question,
                "intent": analysis.intent,
                "entity_type": analysis.entity_type,
                "query_decomposition": list(analysis.subjects),
                "query_plan": {
                    "queries": list(plan.queries),
                    "subject": plan.subject,
                    "relations": list(plan.relations),
                    "intent": plan.analysis.intent,
                    "answer_shape": plan.answer_shape,
                    "set_semantics": plan.set_semantics,
                    "fields": [
                        {
                            "field_id": field.field_id,
                            "question": field.question,
                            "relations": list(field.relations),
                        }
                        for field in plan.fields
                    ],
                    "merge_strategy": plan.merge_strategy,
                    "fusion": "weighted_rrf",
                    "context_policy": plan.context_policy,
                    "planner": planning.strategy,
                    "model_queries": list(planning.model_queries),
                    "fallback_reason": planning.error,
                },
                "timings_ms": {
                    "query_planning": planning_ms,
                    "bm25": bm25_ms,
                    "context_expansion": context_ms,
                    "total": self._elapsed_ms(search_started),
                },
            },
        )

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return max(0, round((monotonic() - started_at) * 1000))

    @staticmethod
    def _remaining_budget(deadline: float, *, reserve: float = 0) -> float:
        return max(0.0, deadline - monotonic() - reserve)

    async def _rerank_documents(
        self,
        question: str,
        plan: QueryPlan,
        sources: list[SourceItem],
        *,
        deadline: float,
    ) -> DocumentRerankResult:
        if (
            not self.settings.semantic_pipeline_enabled
            or self.document_reranker is None
            or not sources
        ):
            return DocumentRerankResult(
                tuple(sources),
                (),
                strategy="disabled",
            )
        remaining = self._remaining_budget(
            deadline,
            reserve=self.settings.ask_generation_reserve,
        )
        if remaining <= 0:
            return DocumentRerankResult(
                tuple(sources),
                (),
                errors=("request_budget_exhausted",),
                strategy="budget_fallback",
            )
        try:
            return await asyncio.wait_for(
                self.document_reranker.rerank(question, plan, sources),
                timeout=remaining,
            )
        except Exception as error:
            return DocumentRerankResult(
                tuple(sources),
                (),
                errors=(f"{type(error).__name__}: {error}",),
                strategy="model_error_fallback",
            )

    async def _plan_queries(
        self,
        question: str,
        fallback_plan: QueryPlan,
    ) -> QueryPlanningResult:
        if self.query_planner is None:
            return QueryPlanningResult(
                fallback_plan,
                "deterministic_fallback",
                error="planner_not_configured",
            )
        return await self.query_planner.plan(question, fallback_plan)

    async def _execute_query_plan(
        self,
        plan: QueryPlan,
        *,
        candidate_k: int,
        knowledge_base_id: str | None,
    ) -> list[LexicalResult]:
        result_groups = await asyncio.gather(*(
            asyncio.to_thread(
                self.index.search,
                query,
                candidate_k=candidate_k,
                knowledge_base_id=knowledge_base_id,
            )
            for query in plan.queries
        ))
        if plan.merge_strategy == "document_interleave":
            return self._merge_comparison_results(result_groups)
        return self._merge_rank_fusion(result_groups, plan=plan)

    @staticmethod
    def _merge_rank_fusion(
        result_groups: tuple[list[LexicalResult], ...],
        *,
        plan: QueryPlan,
    ) -> list[LexicalResult]:
        if len(result_groups) <= 1:
            return list(result_groups[0]) if result_groups else []
        document_scores: dict[str, float] = {}
        chunks: dict[str, dict[str, LexicalResult]] = {}
        chunk_scores: dict[str, float] = {}
        chunk_rank_scores: dict[str, float] = {}
        first_seen: dict[str, int] = {}
        sequence = 0
        for group_index, group in enumerate(result_groups):
            route_weight = max(0.85, 1.15 - group_index * 0.05)
            seen_documents: set[str] = set()
            for rank, result in enumerate(group, start=1):
                document_id = result.document_id or result.node_id
                if document_id not in seen_documents:
                    document_scores[document_id] = (
                        document_scores.get(document_id, 0.0)
                        + route_weight / (20 + rank)
                    )
                    seen_documents.add(document_id)
                chunks.setdefault(document_id, {})[result.node_id] = result
                chunk_scores[result.node_id] = max(
                    chunk_scores.get(result.node_id, 0.0),
                    float(result.score or 0.0),
                )
                chunk_rank_scores[result.node_id] = (
                    chunk_rank_scores.get(result.node_id, 0.0)
                    + route_weight / (20 + rank)
                )
                if document_id not in first_seen:
                    first_seen[document_id] = sequence
                    sequence += 1
        normalized_subject = normalize_search_text(plan.subject).replace(" ", "")
        endpoint_match = (
            _ROUTE_ENDPOINT_PATTERN.search(plan.normalized_question)
            if _ROUTE_ENDPOINT_PATTERN is not None else None
        )
        if endpoint_match:
            start = normalize_search_text(endpoint_match.group("start")).replace(" ", "")
            end = normalize_search_text(endpoint_match.group("end")).replace(" ", "")
            for document_id, document_chunks in chunks.items():
                title = normalize_search_text(
                    next(
                        (
                            str(result.metadata.get("title") or "")
                            for result in document_chunks.values()
                            if result.metadata.get("title")
                        ),
                        "",
                    )
                ).replace(" ", "")
                supports_route = any(
                    start in (text := normalize_search_text(result.text).replace(" ", ""))
                    and end in text
                    and bool(re.search(r"(?:线路|路线|由.{0,16}至|起点|终点)", text))
                    for result in document_chunks.values()
                )
                if supports_route:
                    document_scores[document_id] += 0.18
                    if re.search(r"\d+号线(?:[（(].+?[）)])?$", title):
                        document_scores[document_id] += 0.12
        if normalized_subject:
            narrative_relation = any(
                len(relation.replace(" ", "")) >= 4
                for relation in plan.relations
            )
            list_topic_tokens: set[str] = set()
            if plan.analysis.intent in {"list", "fact"} and endpoint_match is None:
                subject_tokens = set(query_tokens(plan.subject))
                list_topic_tokens = {
                    token
                    for token in query_tokens(plan.normalized_question)
                    if token not in subject_tokens
                    and token not in {
                        "哪些", "哪几个", "全部", "所有", "总共", "一共",
                        "从古至今", "至今",
                    }
                    and len(token) >= 2
                }
            for document_id, document_chunks in chunks.items():
                title = next(
                    (
                        str(result.metadata.get("title") or "")
                        for result in document_chunks.values()
                        if result.metadata.get("title")
                    ),
                    "",
                )
                normalized_title = normalize_search_text(title).replace(" ", "")
                title_tokens = set(lexical_tokens(title))
                subject_tokens = set(lexical_tokens(plan.subject))
                subject_overlap = len(subject_tokens & title_tokens)
                core_tokens = _core_subject_tokens(plan.subject)
                if plan.answer_shape == "list" and subject_tokens and endpoint_match is None:
                    if subject_overlap:
                        document_scores[document_id] += min(
                            9.0,
                            subject_overlap * 3.0,
                        )
                        if core_tokens and core_tokens[0] in title_tokens:
                            document_scores[document_id] += 6.0
                    elif endpoint_match is None:
                        document_scores[document_id] -= 6.0
                metadata = next(iter(document_chunks.values())).metadata
                aliases = document_aliases(title, metadata)
                has_relation_context = bool(plan.relations and any(
                    normalized_subject in normalize_search_text(result.text).replace(" ", "")
                    and any(
                        normalize_search_text(relation).replace(" ", "")
                        in normalize_search_text(result.text).replace(" ", "")
                        for relation in plan.relations
                    )
                    for result in document_chunks.values()
                ))
                if title_matches_subject(title, normalized_subject):
                    document_scores[document_id] += (
                        1.5
                        if plan.answer_shape in {"summary", "narrative"}
                        else 0.3 if not narrative_relation or has_relation_context else 0.02
                    )
                elif (
                    list_topic_tokens
                    and normalized_title.startswith(normalized_subject)
                    and normalized_title != normalized_subject
                ):
                    base_title = re.split(r"[（(]", normalized_title, maxsplit=1)[0]
                    suffix_tokens = set(query_tokens(base_title[len(normalized_subject):]))
                    if list_topic_tokens & suffix_tokens:
                        document_scores[document_id] += (
                            8.0 if plan.analysis.intent == "list" else 5.0
                        )
                        if any(marker in base_title for marker in ("列表", "清单", "目录")):
                            document_scores[document_id] += 0.8
                elif normalized_subject in aliases:
                    document_scores[document_id] += (
                        0.2 if not narrative_relation or has_relation_context else 0.02
                    )
                elif (
                    title_matches_subject_event(
                        title,
                        normalized_subject,
                        plan.normalized_question,
                    )
                ):
                    document_scores[document_id] += 0.24
                elif normalized_title.startswith(
                    (f"{normalized_subject}(", f"{normalized_subject}（")
                ):
                    document_scores[document_id] -= 0.08
                if has_relation_context:
                    document_scores[document_id] += 0.6 if narrative_relation else 0.06
                if plan.analysis.intent == "cause":
                    event_terms = tuple(
                        normalize_search_text(relation).replace(" ", "")
                        for relation in plan.relations
                        if len(normalize_search_text(relation).replace(" ", "")) >= 4
                        and relation not in {"原因", "因素", "导致", "因由", "缘由"}
                    )
                    if event_terms and any(
                        term in normalize_search_text(
                            f"{title}\n" + "\n".join(result.text for result in document_chunks.values())
                        ).replace(" ", "")
                        for term in event_terms
                    ):
                        document_scores[document_id] += 2.0
        ranked_documents = sorted(
            document_scores,
            key=lambda key: (document_scores[key], -first_seen[key]),
            reverse=True,
        )
        top_score = max(
            (document_scores[key] for key in ranked_documents),
            default=1.0,
        ) or 1.0
        merged: list[LexicalResult] = []
        relation_tokens = set(query_tokens(" ".join(plan.relations)))
        for document_id in ranked_documents:
            ranked_chunks = sorted(
                chunks[document_id].values(),
                key=lambda result: (
                    len(
                        relation_tokens
                        & set(lexical_tokens(
                            f"{result.metadata.get('section') or ''}\n{result.text}"
                        ))
                    ),
                    chunk_rank_scores[result.node_id],
                    chunk_scores[result.node_id],
                ),
                reverse=True,
            )
            merged.extend(
                LexicalResult(
                    node_id=result.node_id,
                    document_id=result.document_id,
                    text=result.text,
                    metadata=result.metadata,
                    score=min(1.0, document_scores[document_id] / top_score),
                )
                for result in ranked_chunks
            )
        return merged

    @staticmethod
    def _merge_comparison_results(result_groups: tuple[list[LexicalResult], ...]) -> list[LexicalResult]:
        merged: list[LexicalResult] = []
        seen: set[str] = set()
        depth = max((len(group) for group in result_groups), default=0)
        for index in range(depth):
            for group in result_groups:
                if index >= len(group):
                    continue
                result = group[index]
                if result.document_id in seen:
                    continue
                seen.add(result.document_id)
                merged.append(result)
        return merged

    @staticmethod
    def _merge_supplemental_results(result_groups: tuple[list[LexicalResult], ...]) -> list[LexicalResult]:
        merged: list[LexicalResult] = []
        seen: set[str] = set()
        depth = max((len(group) for group in result_groups), default=0)
        for index in range(depth):
            for group in result_groups:
                if index >= len(group):
                    continue
                result = group[index]
                if result.node_id in seen:
                    continue
                seen.add(result.node_id)
                merged.append(result)
        return merged

    async def _replace_with_lead_chunks(
        self,
        question: str,
        results: list[LexicalResult],
        *,
        knowledge_base_id: str | None,
        intent: str,
    ) -> list[LexicalResult]:
        if intent not in {
            "definition", "comparison", "fact", "time", "agent", "location", "birthplace", "list",
        } or not results:
            return results
        lookup = getattr(self.index, "document_lead_chunk", None)
        if lookup is None:
            return results
        if intent in {"fact", "time", "agent", "location", "birthplace", "list"}:
            augmented: list[LexicalResult] = []
            looked_up_documents: set[str] = set()
            for result in results:
                augmented.append(result)
                if len(looked_up_documents) >= 3 or result.document_id in looked_up_documents:
                    continue
                looked_up_documents.add(result.document_id)
                lead = await asyncio.to_thread(
                    lookup,
                    result.document_id,
                    knowledge_base_id=knowledge_base_id,
                    score=result.score,
                )
                if lead is not None and lead.node_id != result.node_id:
                    augmented.append(lead)
            return augmented
        replaced: list[LexicalResult] = []
        for result in results:
            lead = await asyncio.to_thread(
                lookup,
                result.document_id,
                knowledge_base_id=knowledge_base_id,
                score=result.score,
            )
            replaced.append(lead or result)
        return replaced

    async def _expand_document_relation_context(
        self,
        plan: QueryPlan,
        results: list[LexicalResult],
        *,
        knowledge_base_id: str | None,
        top_k: int,
    ) -> tuple[list[LexicalResult], bool]:
        if not results or not plan.subject or not plan.relations:
            return results, False
        if plan.analysis.intent not in {
            "agent",
            "cause",
            "procedure",
            "time",
            "location",
            "birthplace",
            "ordinal",
            "list",
            "fact",
        }:
            return results, False
        lookup = getattr(self.index, "document_relation_candidates", None)
        if lookup is None:
            return results, False
        normalized_subject = normalize_search_text(plan.subject).replace(" ", "")
        relation_terms = self._document_relation_search_terms(plan)
        topic_tokens = self._topic_tokens(plan.normalized_question, plan.subject)
        first_title = normalize_search_text(
            str(results[0].metadata.get("title") or "")
        ).replace(" ", "")
        topic_document_selected = bool(
            topic_tokens
            and any(token in first_title for token in topic_tokens)
        )
        document_ids: list[str] = (
            [results[0].document_id]
            if topic_document_selected
            else []
        )
        if not topic_document_selected:
            for result in results:
                title = str(result.metadata.get("title") or "")
                if (
                    result.document_id not in document_ids
                    and (
                        title_matches_subject(title, normalized_subject)
                        or normalized_subject in document_aliases(title, result.metadata)
                        or (
                            plan.analysis.intent == "list"
                            and (
                                title_matches_subject_topic(title, normalized_subject)
                                or self._title_contains_core_topic(title, plan.subject)
                            )
                        )
                        or (
                            plan.analysis.intent in {"cause", "time", "list"}
                            and title_matches_subject_event(
                                title,
                                normalized_subject,
                                plan.normalized_question,
                            )
                        )
                    )
                ):
                    document_ids.append(result.document_id)
                if len(document_ids) >= 2:
                    break
        if plan.analysis.intent == "list" and document_ids:
            document_ids = [
                self._primary_topic_document_id(
                    results,
                    set(document_ids),
                    plan.subject,
                )
            ]
        if not document_ids:
            return results, False
        candidate_groups = await asyncio.gather(*(
            asyncio.to_thread(
                lookup,
                plan.normalized_question,
                document_id=document_id,
                relations=relation_terms,
                knowledge_base_id=knowledge_base_id,
                limit=max(top_k, 6),
            )
            for document_id in document_ids
        ))
        relation_candidates = [
            candidate
            for group in candidate_groups
            for candidate in group
        ]
        if plan.answer_shape in {"summary", "narrative"}:
            terminal_markers = (
                "结局", "結局", "结尾", "結尾", "终结", "終結", "结束", "結束",
                "归一", "歸一", "一统", "一統", "统一天下", "統一天下",
                "最终结果", "最終結果", "最后结果", "最後結果", "灭亡", "滅亡",
            )
            relation_candidates.sort(
                key=lambda candidate: (
                    sum(
                        5.0
                        for marker in terminal_markers
                        if marker in normalize_search_text(candidate.text).replace(" ", "")
                    ),
                    sum(
                        normalize_search_text(relation).replace(" ", "")
                        in normalize_search_text(candidate.text).replace(" ", "")
                        for relation in relation_terms
                    ),
                    -int(candidate.metadata.get("chunk_order") or 0),
                ),
                reverse=True,
            )
        if not relation_candidates:
            return results, False
        seen: set[str] = set()
        merged: list[LexicalResult] = []
        for result in (*relation_candidates, *results):
            if result.node_id in seen:
                continue
            seen.add(result.node_id)
            merged.append(result)
        return merged, True

    @staticmethod
    def _document_relation_search_terms(plan: QueryPlan) -> tuple[str, ...]:
        subject_tokens = set(lexical_tokens(plan.subject))
        ignored = subject_tokens | {
            "什么", "哪些", "哪个", "哪几个", "怎么", "如何", "多少",
            "主要", "历史", "著名", "伟大", "列表", "全部", "所有",
        }
        query_relations = [
            token
            for token in query_tokens(" ".join(plan.queries))
            if len(token.strip()) >= 2 and token not in ignored
        ]
        summary_terms = (
            "结局", "结尾", "终结", "结束", "最终", "最后", "结果",
            "归一", "一统", "统一", "灭亡", "完成",
        ) if plan.answer_shape in {"summary", "narrative"} else ()
        return tuple(dict.fromkeys((*plan.relations, *query_relations, *summary_terms)))

    @staticmethod
    def _topic_tokens(question: str, subject: str) -> set[str]:
        subject_tokens = set(query_tokens(subject))
        return {
            token
            for token in query_tokens(question)
            if token not in subject_tokens
            and token not in _STRUCTURE_QUESTION_WORDS
            and token not in _SCOPE_TOKENS
            and token not in {
                "个", "总共", "一共", "哪些", "哪几个", "全部", "所有",
                "列表", "主要", "著名", "伟大", "分别", "从古至今", "至今",
            }
            and len(token) >= 2
        }

    async def _expand_section_context(
        self,
        results: list[LexicalResult],
        *,
        knowledge_base_id: str | None,
        top_k: int,
    ) -> tuple[list[LexicalResult], bool]:
        if not results or top_k <= 1:
            return results, False
        anchor = results[0]
        parent_id = str(anchor.metadata.get("parent_id") or "")
        if not parent_id or int(anchor.metadata.get("structure_size") or 1) <= 1:
            return results, False
        lookup = getattr(self.index, "structure_chunks", None)
        if lookup is None:
            return results, False
        siblings = await asyncio.to_thread(
            lookup,
            parent_id,
            knowledge_base_id=knowledge_base_id,
            limit=min(top_k, _CAUSE_CONTEXT_MAX_CHUNKS),
            score=anchor.score,
        )
        if len(siblings) <= 1:
            return results, False
        sibling_ids = {result.node_id for result in siblings}
        merged = [
            *siblings,
            *(result for result in results if result.node_id not in sibling_ids),
        ]
        return merged[:top_k], True

    async def _run_active_retrieval(
        self,
        request: SearchRequest,
        response: SearchResponse,
        *,
        evidence_top_k: int,
        deadline: float,
    ) -> tuple[SearchResponse, dict[str, object], EvidenceExtractionResult | None]:
        active_started = monotonic()
        timings: dict[str, int] = {
            "initial_extraction": 0,
            "planning": 0,
            "search": 0,
            "final_extraction": 0,
        }
        trace: dict[str, object] = {
            "enabled": bool(
                self.retrieval_agent is not None
                and self.settings.active_retrieval_enabled
            ),
            "rounds": [],
            "tool_calls": 0,
            "stop_reason": "agent_not_configured",
            "timings_ms": timings,
        }
        question = str(response.retrieval.get("normalized_question") or request.question)
        preliminary_plan = self._answer_plan(question, response.retrieval)
        analysis = preliminary_plan.analysis
        document_rerank = await self._rerank_documents(
            question,
            preliminary_plan,
            response.results,
            deadline=deadline,
        )
        trace["document_reranking"] = {
            "strategy": document_rerank.strategy,
            "selected_document_ids": list(dict.fromkeys(
                source.document_id for source in document_rerank.sources
            )),
            "decisions": [
                {
                    "document_id": decision.document_id,
                    "relevant": decision.relevant,
                    "score": decision.score,
                    "reason": decision.reason,
                }
                for decision in document_rerank.decisions
            ],
            "errors": list(document_rerank.errors),
        }
        if self.settings.semantic_pipeline_enabled:
            preliminary_sources = list(document_rerank.sources)
            response = response.model_copy(update={"results": preliminary_sources})
        else:
            preliminary_sources = self._focus_sources_on_subject(
                preliminary_plan,
                response.results,
            )
        lexical_gate = evaluate_evidence_gate(
            question,
            analysis,
            preliminary_sources,
            subject=preliminary_plan.subject,
            relations=preliminary_plan.relations,
            field_evidence_available=False,
            field_candidate_count=0,
        )
        direct_answer = self._deterministic_answer(
            question,
            preliminary_plan,
            preliminary_sources,
        )
        if (
            not self.settings.semantic_pipeline_enabled
            and lexical_gate.passed
            and direct_answer is not None
        ):
            trace["stop_reason"] = "initial_evidence_sufficient"
            trace["deterministic_shortcut"] = True
            timings["total"] = self._elapsed_ms(active_started)
            return response, trace, None

        extraction: EvidenceExtractionResult | None = None
        if lexical_gate.passed or (
            self.settings.semantic_pipeline_enabled and preliminary_sources
        ):
            extraction_started = monotonic()
            extraction = await self._extract_field_evidence(
                question,
                preliminary_plan,
                preliminary_sources,
                timeout=self._remaining_budget(
                    deadline,
                    reserve=self.settings.ask_generation_reserve,
                ),
            )
            timings["initial_extraction"] = self._elapsed_ms(extraction_started)
        preliminary_gate = evaluate_evidence_gate(
            question,
            analysis,
            preliminary_sources,
            subject=preliminary_plan.subject,
            relations=preliminary_plan.relations,
            field_evidence_available=bool(extraction and extraction.available),
            field_candidate_count=len(extraction.candidates) if extraction else 0,
        )
        missing_relation_context = bool(
            analysis.intent in {"cause", "procedure"}
            and not (extraction and extraction.has_candidates)
            and not (
                response.retrieval.get("section_context_expanded")
                or response.retrieval.get("document_relation_expanded")
            )
        )
        triggers: list[str] = []
        if self.settings.semantic_pipeline_enabled and not (
            extraction and extraction.has_candidates
        ):
            triggers.append(
                "field_evidence_extraction_failed"
                if self._field_extraction_failed(extraction)
                else "field_evidence_missing"
            )
        if not preliminary_gate.passed:
            trigger = (
                "field_evidence_missing"
                if extraction and extraction.available
                else "evidence_gate_failed"
            )
            if trigger not in triggers:
                triggers.append(trigger)
        if missing_relation_context:
            triggers.append("relation_context_missing")
        trace["trigger"] = triggers
        if self.retrieval_agent is None:
            timings["total"] = self._elapsed_ms(active_started)
            return response, trace, extraction
        if not self.settings.active_retrieval_enabled:
            trace["stop_reason"] = "disabled"
            timings["total"] = self._elapsed_ms(active_started)
            return response, trace, extraction
        if not triggers:
            trace["stop_reason"] = "initial_evidence_sufficient"
            timings["total"] = self._elapsed_ms(active_started)
            return response, trace, extraction

        initial_queries = response.retrieval.get("query_plan", {}).get("queries", [])
        used_queries = [
            str(query).strip()
            for query in initial_queries
            if str(query).strip()
        ]
        used_normalized = {
            normalize_search_text(query).replace(" ", "")
            for query in used_queries
        }
        current = response
        rounds = trace["rounds"]
        assert isinstance(rounds, list)
        for round_number in range(1, self.settings.active_retrieval_max_rounds + 1):
            remaining = self._remaining_budget(
                deadline,
                reserve=self.settings.ask_generation_reserve,
            )
            if remaining <= 0:
                trace["stop_reason"] = "request_budget_exhausted"
                break
            planning_started = monotonic()
            try:
                result = await asyncio.wait_for(
                    self.retrieval_agent.decide(
                        request.question,
                        [] if not preliminary_gate.passed else current.results,
                        used_queries=tuple(used_queries),
                        round_number=round_number,
                    ),
                    timeout=remaining,
                )
            except TimeoutError:
                trace["stop_reason"] = "request_budget_exhausted"
                break
            finally:
                timings["planning"] += self._elapsed_ms(planning_started)
            if result.decision is None:
                rounds.append({"round": round_number, "error": result.error})
                trace["stop_reason"] = result.error or "agent_error"
                break
            decision = result.decision
            round_trace: dict[str, object] = {
                "round": round_number,
                "action": decision.action,
                "queries": list(decision.queries),
                "reason": decision.reason,
            }
            rounds.append(round_trace)
            if decision.action == "finish":
                trace["stop_reason"] = "model_finish"
                break

            queries: list[str] = []
            for query in decision.queries:
                normalized = normalize_search_text(query).replace(" ", "")
                if not normalized or normalized in used_normalized:
                    continue
                used_normalized.add(normalized)
                used_queries.append(query)
                queries.append(query)
            if not queries:
                trace["stop_reason"] = "no_new_queries"
                break

            search_started = monotonic()
            search_results = await asyncio.gather(*(
                self.search(
                    SearchRequest(
                        question=request.question,
                        top_k=evidence_top_k,
                        candidate_k=request.candidate_k,
                        min_score=request.min_score,
                        knowledge_base_id=request.knowledge_base_id,
                    ),
                    use_model_planner=False,
                    query_override=(query,),
                )
                for query in queries
            ), return_exceptions=True)
            timings["search"] += self._elapsed_ms(search_started)
            supplemental: list[SearchResponse] = []
            tool_results: list[dict[str, object]] = []
            for query, search_result in zip(queries, search_results, strict=True):
                if isinstance(search_result, BaseException):
                    tool_results.append({"query": query, "error": str(search_result)})
                    continue
                supplemental.append(search_result)
                tool_results.append({"query": query, "returned": len(search_result.results)})
            round_trace["tool_results"] = tool_results
            trace["tool_calls"] = int(trace["tool_calls"]) + len(queries)
            if not supplemental or not any(item.results for item in supplemental):
                trace["stop_reason"] = "no_results"
                break
            merged = self._merge_active_sources(
                [item.results for item in supplemental],
                current.results,
                limit=self.settings.active_retrieval_max_results,
            )
            current = current.model_copy(update={"results": merged})
            round_trace["evidence_count"] = len(merged)
        else:
            trace["stop_reason"] = "max_rounds"
        if current is response:
            timings["total"] = self._elapsed_ms(active_started)
            return current, trace, extraction
        if self.settings.semantic_pipeline_enabled:
            final_rerank = await self._rerank_documents(
                question,
                preliminary_plan,
                current.results,
                deadline=deadline,
            )
            final_sources = list(final_rerank.sources)
            trace["final_document_reranking"] = {
                "strategy": final_rerank.strategy,
                "selected_document_ids": list(dict.fromkeys(
                    source.document_id for source in final_rerank.sources
                )),
                "errors": list(final_rerank.errors),
            }
        else:
            final_sources = self._focus_sources_on_subject(
                preliminary_plan,
                current.results,
            )
        final_extraction_started = monotonic()
        final_extraction = await self._extract_field_evidence(
            question,
            preliminary_plan,
            final_sources,
            timeout=self._remaining_budget(
                deadline,
                reserve=self.settings.ask_generation_reserve,
            ),
        )
        timings["final_extraction"] = self._elapsed_ms(final_extraction_started)
        timings["total"] = self._elapsed_ms(active_started)
        return current.model_copy(update={"results": final_sources}), trace, final_extraction

    @staticmethod
    def _merge_active_sources(
        supplemental_groups: list[list[SourceItem]],
        original: list[SourceItem],
        *,
        limit: int,
    ) -> list[SourceItem]:
        groups = [*supplemental_groups, original]
        depth = max((len(group) for group in groups), default=0)
        merged: list[SourceItem] = []
        seen: set[str] = set()
        for index in range(depth):
            for group in groups:
                if index >= len(group):
                    continue
                source = group[index]
                if source.id in seen:
                    continue
                seen.add(source.id)
                merged.append(source)
                if len(merged) >= limit:
                    return merged
        return merged

    async def ask(self, request: SearchRequest) -> AskResponse:
        if self.settings.generation_output_mode == "immutable":
            return await self._ask_immutable(request)
        ask_started = monotonic()
        deadline = ask_started + self.settings.ask_total_timeout
        timings: dict[str, object] = {}
        display_top_k = min(request.top_k or self.settings.default_top_k, self.settings.max_top_k)
        evidence_top_k, evidence_policy = self._adaptive_evidence_top_k(
            request.question,
            display_top_k=display_top_k,
        )
        evidence_request = request.model_copy(
            update={"top_k": evidence_top_k}
        )
        initial_search_started = monotonic()
        try:
            evidence_response = await self.search(evidence_request)
        except Exception as error:
            elapsed = self._elapsed_ms(initial_search_started)
            retrieval = {
                "algorithm": "OpenSearch BM25",
                "mode": "bm25+keyword",
                "top_k": display_top_k,
                "returned": 0,
                "retrieval_error": str(error),
                "retrieval_error_type": type(error).__name__,
                "timings_ms": {
                    "initial_search": elapsed,
                    "total": self._elapsed_ms(ask_started),
                },
            }
            generation = {
                "evidence_count": 0,
                "displayed_evidence_count": 0,
                "evidence_gate_passed": False,
                "answer_strategy": "retrieval_error",
            }
            diagnosis = diagnose_failure(
                answer="根据检索到的资料，无法确定。",
                sources=[],
                retrieval=retrieval,
                generation=generation,
            )
            generation.update({
                "failure_category": diagnosis.category,
                "failure_reason": diagnosis.reason,
                "failure_stage": diagnosis.stage,
            })
            return AskResponse(
                answer="根据检索到的资料，无法确定。",
                sources=[],
                retrieval=retrieval,
                generation=generation,
            )
        timings["initial_search"] = self._elapsed_ms(initial_search_started)
        timings["initial_search_detail"] = evidence_response.retrieval.get("timings_ms", {})
        active_started = monotonic()
        evidence_response, active_retrieval, field_extraction = await self._run_active_retrieval(
            request,
            evidence_response,
            evidence_top_k=evidence_top_k,
            deadline=deadline,
        )
        timings["evidence_and_active_retrieval"] = self._elapsed_ms(active_started)
        retrieval = {
            **evidence_response.retrieval,
            "top_k": display_top_k,
            "returned": min(len(evidence_response.results), display_top_k),
            "answer_evidence_top_k": evidence_response.retrieval.get("top_k"),
            "answer_evidence_count": len(evidence_response.results),
            "evidence_top_k_policy": evidence_policy,
            "active_retrieval": active_retrieval,
        }
        question = str(retrieval.get("normalized_question") or request.question)
        answer_plan = self._answer_plan(question, retrieval)
        question_analysis = answer_plan.analysis
        retrieval["evidence_subject"] = answer_plan.subject
        retrieval["evidence_relations"] = list(answer_plan.relations)
        if (
            not self.settings.semantic_pipeline_enabled
            and question_analysis.intent == "cause"
        ):
            evidence_response = evidence_response.model_copy(
                update={
                    "results": self._trim_post_event_evidence(
                        evidence_response.results,
                        subject=answer_plan.subject,
                        relations=answer_plan.relations,
                    )
                }
            )
        if not self.settings.semantic_pipeline_enabled:
            evidence_response = evidence_response.model_copy(
                update={
                    "results": self._focus_sources_on_subject(
                        answer_plan,
                        evidence_response.results,
                    )
                }
            )
        if (
            field_extraction is not None
            and not field_extraction.matches_sources(evidence_response.results)
        ):
            remapped_extraction = field_extraction.remap_sources(
                evidence_response.results
            )
            if remapped_extraction is not None:
                field_extraction = remapped_extraction
                timings["post_filter_extraction"] = 0
                timings["field_evidence_remapped"] = True
            else:
                extraction_started = monotonic()
                field_extraction = await self._extract_field_evidence(
                    question,
                    answer_plan,
                    evidence_response.results,
                    timeout=self._remaining_budget(
                        deadline,
                        reserve=self.settings.ask_generation_reserve,
                    ),
                )
                timings["post_filter_extraction"] = self._elapsed_ms(extraction_started)
        answer_sources = (
            field_extraction.answer_sources(evidence_response.results)
            if field_extraction is not None and field_extraction.has_candidates
            else evidence_response.results
        )
        if (
            self.settings.semantic_pipeline_enabled
            and field_extraction is not None
            and field_extraction.has_candidates
            and len({source.document_id for source in answer_sources}) > 1
        ):
            evidence_document_rerank = await self._rerank_documents(
                question,
                answer_plan,
                answer_sources,
                deadline=deadline,
            )
            answer_sources = list(evidence_document_rerank.sources)
            active_retrieval["evidence_document_reranking"] = {
                "strategy": evidence_document_rerank.strategy,
                "selected_document_ids": list(dict.fromkeys(
                    source.document_id for source in evidence_document_rerank.sources
                )),
                "decisions": [
                    {
                        "document_id": decision.document_id,
                        "relevant": decision.relevant,
                        "score": decision.score,
                        "reason": decision.reason,
                    }
                    for decision in evidence_document_rerank.decisions
                ],
                "errors": list(evidence_document_rerank.errors),
            }
        if (
            self.settings.semantic_pipeline_enabled
            and answer_plan.answer_shape == "list"
            and answer_plan.set_semantics == "all"
        ):
            focused_answer_sources = self._focus_complete_list_evidence(answer_sources)
            if len(focused_answer_sources) < len(answer_sources):
                active_retrieval["evidence_density_focus"] = {
                    "before": len(answer_sources),
                    "after": len(focused_answer_sources),
                    "selected_document_ids": list(dict.fromkeys(
                        source.document_id for source in focused_answer_sources
                    )),
                }
                answer_sources = focused_answer_sources
        verified_structured_answer = None
        if self.settings.semantic_pipeline_enabled:
            verified_structured_answer = self._verified_structured_render(
                question,
                answer_plan,
                answer_sources,
            )
            if verified_structured_answer is not None:
                active_retrieval["verified_structured_render"] = True
        if (
            answer_plan.analysis.intent == "agent"
        ):
            answer_sources = self._sort_relation_sources(
                answer_sources,
                subject=answer_plan.subject,
                relations=answer_plan.relations,
            )
        gate = evaluate_evidence_gate(
            question,
            question_analysis,
            answer_sources,
            subject=answer_plan.subject,
            relations=answer_plan.relations,
            field_evidence_available=bool(field_extraction and field_extraction.available),
            field_candidate_count=len(field_extraction.candidates) if field_extraction else 0,
        )
        field_extraction_failed = self._field_extraction_failed(field_extraction)
        fallback_sources = (
            evidence_response.results
            if (
                answer_plan.answer_shape in {"summary", "narrative"}
                or field_extraction is not None and field_extraction.errors
            )
            else answer_sources
        )
        if (
            self.settings.semantic_pipeline_enabled
            and not (field_extraction and field_extraction.has_candidates)
            and not field_extraction_failed
        ):
            gate = replace(
                gate,
                passed=False,
                issues=tuple(dict.fromkeys((*gate.issues, "field_evidence_missing"))),
            )
        assessment = gate.assessment
        answer_response = evidence_response.model_copy(update={"results": answer_sources})
        cache_key = self._answer_cache_key(question, answer_response, answer_plan)
        cache_allowed = not (
            self.settings.semantic_pipeline_enabled
            and answer_plan.answer_shape in {"list", "summary", "narrative"}
        )
        answer = self._get_cached_answer(cache_key) if cache_allowed else None
        cache_hit = answer is not None
        answer_strategy = "cache" if cache_hit else "model"
        if answer is None and verified_structured_answer is not None:
            answer = verified_structured_answer
            if answer is not None:
                answer_strategy = "verified_structured_render"
        ambiguity = ambiguity_candidates(question, answer_sources)
        raw_model_answer: str | None = None
        retry_model_answer: str | None = None
        verification_model_answer: str | None = None
        verification_queries: list[str] = []
        blocked_answer: str | None = None
        answer_block_reason: str | None = None
        if answer is None:
            if (
                not self.settings.semantic_pipeline_enabled
                and ambiguity
                and gate.passed
            ):
                labels = "、".join(
                    f"{title}[资料 {next(index for index, source in enumerate(answer_sources, start=1) if source.title == title)}]"
                    for title in ambiguity
                )
                answer = f"这个名称可能指：{labels}。请补充你想查询的具体对象。"
                answer_strategy = "clarification"
            elif gate.passed and not self.settings.semantic_pipeline_enabled:
                answer = self._deterministic_answer(
                    question,
                    answer_plan,
                    answer_sources,
                )
            if answer is not None:
                answer_strategy = "direct_extract"
            elif gate.passed:
                generation_started = monotonic()
                try:
                    answer = await self._generate_answer(
                        question,
                        answer_sources,
                        plan=answer_plan,
                        subject=answer_plan.subject,
                        relations=answer_plan.relations,
                        trusted_evidence=bool(field_extraction and field_extraction.has_candidates),
                        timeout=self._remaining_budget(deadline),
                    )
                    raw_model_answer = self._model_raw_output(answer)
                    if (
                        self.settings.semantic_pipeline_enabled
                        and self._answer_contract_failed(question, answer)
                        and self._remaining_budget(deadline) > 0
                    ):
                        retry_model_answer = await self._generate_answer(
                            question,
                            answer_sources,
                            plan=answer_plan,
                            subject=answer_plan.subject,
                            relations=answer_plan.relations,
                            trusted_evidence=True,
                            timeout=self._remaining_budget(deadline),
                        )
                        answer = retry_model_answer
                        answer_strategy = "model_retry"
                except (AnswerGenerationError, TimeoutError) as error:
                    fallback = None
                    if not self.settings.semantic_pipeline_enabled:
                        fallback = (
                            cause_evidence_answer(answer_sources)
                            if question_analysis.intent == "cause"
                            else None
                        )
                        if (
                            fallback is None
                            and field_extraction
                            and field_extraction.has_candidates
                        ):
                            fallback = self._field_evidence_quote_answer(
                                answer_sources,
                                answer_plan.relations,
                            )
                    if fallback is None and field_extraction_failed:
                        fallback = self._evidence_fallback_answer(
                            question,
                            answer_plan,
                            fallback_sources,
                        )
                        if fallback is not None:
                            answer_sources = list(fallback_sources)
                    answer = fallback or "根据检索到的资料，无法确定。"
                    answer_strategy = (
                        "generation_timeout_fallback"
                        if isinstance(error, TimeoutError)
                        else "generation_error_fallback"
                    )
                    answer_block_reason = type(error).__name__
                finally:
                    timings["answer_generation"] = self._elapsed_ms(generation_started)
                if (
                    not self.settings.semantic_pipeline_enabled
                    and
                    any(
                        marker in answer
                        for marker in ("无法确定", "无法从资料", "资料不足", "不能确定")
                    )
                    and question_analysis.intent == "cause"
                ):
                    fallback = cause_evidence_answer(answer_sources)
                    if fallback is not None:
                        answer = fallback
                        answer_strategy = "evidence_fallback"
                if (
                    self.settings.semantic_pipeline_enabled
                    and (
                        field_extraction_failed
                        or bool(field_extraction and field_extraction.has_candidates)
                    )
                    and answer in _REFUSAL_ANSWERS
                ):
                    fallback = self._evidence_fallback_answer(
                        question,
                        answer_plan,
                        fallback_sources,
                    )
                    if fallback is not None:
                        answer_sources = list(fallback_sources)
                        answer = fallback
                        answer_strategy = "evidence_fallback"
                if (
                    not self.settings.semantic_pipeline_enabled
                    and
                    any(
                        marker in answer
                        for marker in ("无法确定", "无法从资料", "资料不足", "不能确定")
                    )
                    and field_extraction
                    and field_extraction.has_candidates
                ):
                    fallback = self._field_evidence_quote_answer(
                        answer_sources,
                        answer_plan.relations,
                    )
                    if fallback is not None:
                        answer = fallback
                        answer_strategy = "evidence_fallback"
            else:
                answer = "根据检索到的资料，无法确定。"
                if self.settings.semantic_pipeline_enabled:
                    answer_strategy = "evidence_blocked"
                answer_block_reason = "field_evidence_missing" if (
                    field_extraction and field_extraction.available
                ) else "insufficient_evidence"
        if (
            not self.settings.semantic_pipeline_enabled
            and question_analysis.intent == "cause"
            and self._is_empty_answer_shell(answer)
        ):
            fallback = cause_evidence_answer(answer_sources)
            if fallback is not None:
                answer = fallback
                answer_strategy = "evidence_fallback"
        validation = validate_grounding(answer, answer_sources)
        if (
            not self.settings.semantic_pipeline_enabled
            and "unsupported_number" in validation.issues
            and answer_strategy in {"model", "cache"}
        ):
            cleaned_answer = remove_unsupported_number_sentences(answer, validation.unsupported_numbers)
            answer = cleaned_answer or "根据检索到的资料，无法确定。"
            validation = validate_grounding(answer, answer_sources)
        list_validation = validate_list_answer(question, answer, answer_sources)
        if (
            not self.settings.semantic_pipeline_enabled
            and list_validation.complete is False
        ):
            fallback = list_evidence_answer(question, answer_sources)
            blocked_answer = answer
            answer = fallback or "根据检索到的资料，无法确定完整列表。[资料 1]"
            answer_strategy = "evidence_fallback" if fallback else "incomplete_list_blocked"
            answer_block_reason = "incomplete_list"
            validation = validate_grounding(answer, answer_sources)
        if answer_strategy == "model" and not self.settings.semantic_pipeline_enabled:
            answer = repair_answer_citations(answer, answer_sources)
        answer_support = evaluate_answer_support(answer, answer_sources, question=question)
        if (
            self.settings.semantic_pipeline_enabled
            and answer_strategy in {"model", "model_retry"}
            and any(
                issue in answer_support.issues
                for issue in {"weak_answer_evidence_overlap", "unsupported_entity_term"}
            )
            or (
                self.settings.semantic_pipeline_enabled
                and answer_strategy in {"model", "model_retry"}
                and self._answer_contract_failed(question, answer)
            )
            or (
                self.settings.semantic_pipeline_enabled
                and answer_strategy in {"model", "model_retry"}
                and answer_plan.answer_shape in {"summary", "narrative"}
                and answer_support.coverage < 0.8
            )
        ):
            fallback = self._evidence_fallback_answer(
                question,
                answer_plan,
                fallback_sources,
            )
            if fallback is not None:
                blocked_answer = answer
                answer_sources = list(fallback_sources)
                answer = fallback
                answer_strategy = "evidence_fallback"
                validation = validate_grounding(answer, answer_sources)
                answer_support = evaluate_answer_support(
                    answer,
                    answer_sources,
                    question=question,
                )
        if (
            self.settings.semantic_pipeline_enabled
            and field_extraction_failed
            and answer_strategy in {"model", "model_retry"}
            and answer_support.unsupported_terms
        ):
            fallback = self._evidence_fallback_answer(
                question,
                answer_plan,
                fallback_sources,
            )
            if fallback is not None:
                blocked_answer = answer
                answer_sources = list(fallback_sources)
                answer = fallback
                answer_strategy = "evidence_fallback"
                validation = validate_grounding(answer, answer_sources)
                answer_support = evaluate_answer_support(
                    answer,
                    answer_sources,
                    question=question,
                )
        if (
            self.settings.semantic_pipeline_enabled
            and answer_strategy in {"model", "model_retry"}
            and answer_plan.answer_shape in {"summary", "narrative"}
            and field_extraction
            and field_extraction.has_candidates
        ):
            fallback = self._evidence_fallback_answer(
                question,
                answer_plan,
                fallback_sources,
            )
            if (
                fallback is not None
                and not self._extraction_covers_fallback(field_extraction, fallback)
            ):
                blocked_answer = answer
                answer_sources = list(fallback_sources)
                answer = fallback
                answer_strategy = "evidence_fallback"
                validation = validate_grounding(answer, answer_sources)
                answer_support = evaluate_answer_support(
                    answer,
                    answer_sources,
                    question=question,
                )
        if (
            not self.settings.semantic_pipeline_enabled
            and answer_plan.answer_shape in {"summary", "narrative"}
            and answer_plan.relations
            and any(
                marker in normalize_search_text(" ".join(answer_plan.relations))
                for marker in (
                    "结局", "結局", "结尾", "結尾", "终结", "終結", "结束", "結束",
                    "最终", "最終", "最后", "最後", "结果", "結果", "归一", "歸一",
                    "一统", "一統", "统一", "統一", "灭亡", "滅亡",
                )
            )
            and not any(
                marker in normalize_search_text(answer)
                for marker in (
                    "结局", "結局", "结尾", "結尾", "终结", "終結", "结束", "結束",
                    "最终", "最終", "最后", "最後", "结果", "結果", "归一", "歸一",
                    "一统", "一統", "统一", "統一", "灭亡", "滅亡",
                )
            )
        ):
            answer_support = replace(
                answer_support,
                passed=False,
                issues=(*answer_support.issues, "summary_relation_mismatch"),
            )
        if (
            not self.settings.semantic_pipeline_enabled
            and not answer_support.passed
            and answer_strategy == "model"
            and raw_model_answer
            and answer_support.unsupported_terms
            and self._remaining_budget(deadline)
            >= self.settings.answer_verification_min_budget
        ):
            verification_started = monotonic()
            (
                verified_sources,
                verified_extraction,
                verification_queries,
            ) = await self._answer_guided_verification(
                request,
                answer_plan,
                answer_support.unsupported_terms,
                answer_sources,
                evidence_top_k=evidence_top_k,
                deadline=deadline,
            )
            if verified_sources and verified_extraction and verified_extraction.has_candidates:
                candidate_answer = repair_answer_citations(
                    raw_model_answer,
                    verified_sources,
                )
                candidate_validation = validate_grounding(candidate_answer, verified_sources)
                candidate_support = evaluate_answer_support(
                    candidate_answer,
                    verified_sources,
                    question=question,
                )
                if not (candidate_validation.valid and candidate_support.passed):
                    try:
                        candidate_answer = await self._generate_answer(
                            question,
                            verified_sources,
                            plan=answer_plan,
                            subject=answer_plan.subject,
                            relations=answer_plan.relations,
                            trusted_evidence=True,
                            timeout=self._remaining_budget(deadline),
                        )
                    except Exception:
                        candidate_answer = ""
                    verification_model_answer = candidate_answer
                    candidate_answer = repair_answer_citations(
                        candidate_answer,
                        verified_sources,
                    )
                    candidate_validation = validate_grounding(
                        candidate_answer,
                        verified_sources,
                    )
                    candidate_support = evaluate_answer_support(
                        candidate_answer,
                        verified_sources,
                        question=question,
                    )
                if candidate_validation.valid and candidate_support.passed:
                    answer = candidate_answer
                    answer_sources = verified_sources
                    field_extraction = verified_extraction
                    answer_strategy = "model_verified_retrieval"
                    validation = candidate_validation
                    answer_support = candidate_support
                    gate = evaluate_evidence_gate(
                        question,
                        question_analysis,
                        answer_sources,
                        subject=answer_plan.subject,
                        relations=answer_plan.relations,
                        field_evidence_available=True,
                        field_candidate_count=len(field_extraction.candidates),
                    )
                    assessment = gate.assessment
                    answer_response = evidence_response.model_copy(update={"results": answer_sources})
                    cache_key = self._answer_cache_key(
                        question,
                        answer_response,
                        answer_plan,
                    )
            timings["answer_verification"] = self._elapsed_ms(verification_started)
        if (
            not self.settings.semantic_pipeline_enabled
            and not answer_support.passed
            and answer_strategy in {"model", "cache"}
        ):
            blocked_answer = answer
            fallback = (
                cause_evidence_answer(answer_sources)
                if question_analysis.intent == "cause"
                else None
            )
            if fallback is None and field_extraction and field_extraction.has_candidates:
                fallback = self._field_evidence_quote_answer(
                    answer_sources,
                    answer_plan.relations,
                )
            answer = fallback or "根据检索到的资料，无法确定。"
            answer_strategy = "evidence_fallback" if fallback else "answer_grounding_blocked"
            answer_block_reason = None if fallback else "answer_support_failed"
            validation = validate_grounding(answer, answer_sources)
            answer_support = evaluate_answer_support(answer, answer_sources, question=question)
        answer_response = evidence_response.model_copy(update={"results": answer_sources})
        cache_key = self._answer_cache_key(question, answer_response, answer_plan)
        if (
            cache_allowed
            and gate.passed
            and assessment.grounded
            and validation.valid
            and answer_support.passed
            and answer not in _REFUSAL_ANSWERS
            and not self._is_empty_answer_shell(answer)
        ):
            self._store_cached_answer(cache_key, answer)
        elif cache_hit:
            self._answer_cache.pop(cache_key, None)
        response_results = self._display_sources(
            answer_sources,
            answer,
            display_top_k=display_top_k,
        )
        retrieval["returned"] = len(response_results)
        model_lookup_started = monotonic()
        try:
            model_name = await asyncio.wait_for(
                self.generator.current_model(),
                timeout=1.0,
            )
        except TimeoutError:
            model_name = None
        timings["model_lookup"] = self._elapsed_ms(model_lookup_started)
        timings["total"] = self._elapsed_ms(ask_started)
        timings["budget_ms"] = self.settings.ask_total_timeout * 1000
        retrieval["timings_ms"] = timings
        retrieval["request_budget_exhausted"] = monotonic() >= deadline
        generation = {
            "model": model_name,
            "endpoint": self.settings.generation_base_url,
            "evidence_count": len(answer_sources),
            "displayed_evidence_count": len(response_results),
            "evidence_grounded": assessment.grounded,
            "question_terms": sorted(assessment.question_terms),
            "matched_evidence_terms": sorted(assessment.matched_terms),
            "matched_specific_terms": sorted(assessment.matched_specific_terms),
            "evidence_anchors": sorted(assessment.anchors),
            "matched_evidence_anchors": sorted(assessment.matched_anchors),
            "evidence_gate_passed": gate.passed,
            "evidence_gate_issues": list(gate.issues),
            "relation_terms": list(gate.relation_terms),
            "matched_relation_terms": list(gate.matched_relation_terms),
            "citation_required": not self.settings.semantic_pipeline_enabled,
            "cache_hit": cache_hit,
            "answer_strategy": answer_strategy,
            "answer_shape": answer_plan.answer_shape,
            "intent": question_analysis.intent,
            "entity_type": question_analysis.entity_type,
            "ambiguity_candidates": ambiguity,
            "grounding_valid": validation.valid,
            "grounding_issues": list(validation.issues),
            "answer_support_passed": answer_support.passed,
            "answer_support_coverage": round(answer_support.coverage, 4),
            "answer_support_issues": list(answer_support.issues),
            "unsupported_answer_terms": list(answer_support.unsupported_terms),
            "unsupported_numbers": list(validation.unsupported_numbers),
            "list_complete": list_validation.complete,
            "list_expected_count": list_validation.expected_count,
            "list_answer_count": list_validation.answer_count,
            "list_issues": list(list_validation.issues),
            "blocked_reason": "insufficient_evidence" if not gate.passed else None,
            "raw_model_answer": raw_model_answer,
            "retry_model_answer": retry_model_answer,
            "verification_model_answer": verification_model_answer,
            "verification_queries": verification_queries,
            "blocked_answer": blocked_answer,
            "answer_block_reason": answer_block_reason,
            "field_evidence_available": bool(field_extraction and field_extraction.available),
            "field_evidence": [
                {
                    "field_id": candidate.field_id,
                    "source_index": candidate.source_index + 1,
                    "span": candidate.span,
                    "sha256": candidate.content_hash,
                }
                for candidate in (field_extraction.candidates if field_extraction else ())
            ],
            "field_evidence_errors": list(field_extraction.errors) if field_extraction else [],
            "field_evidence_strategy": field_extraction.strategy if field_extraction else None,
            "field_evidence_fallback": (
                "raw_retrieval" if field_extraction_failed else None
            ),
            "trace_version": "1",
            "trace_stages": self._trace_stages(
                request=request,
                evidence_response=evidence_response,
                answer_sources=answer_sources,
                field_extraction=field_extraction,
                active_retrieval=active_retrieval,
            ),
        }
        if isinstance(self.generator, EvidenceAnswerGenerator):
            generation["writer_trace"] = {
                "output_modified": True,
                "note": "legacy pipeline output is retained for compatibility only",
            }
        diagnosis = diagnose_failure(
            answer=answer,
            sources=response_results,
            retrieval=retrieval,
            generation=generation,
        )
        generation.update({
            "failure_category": diagnosis.category,
            "failure_reason": diagnosis.reason,
            "failure_stage": diagnosis.stage,
        })
        return AskResponse(
            answer=answer,
            sources=response_results,
            retrieval=retrieval,
            generation=generation,
        )

    async def ask_materials(
        self,
        question: str,
        materials: list[SourceItem],
    ) -> AskResponse:
        started = monotonic()
        plan = build_query_plan(question)
        result = await self._generate_with_trace(
            question,
            materials,
            plan=plan,
            trusted_evidence=True,
            timeout=self.settings.generation_total_timeout,
        )
        return AskResponse(
            answer=result.answer,
            sources=materials,
            retrieval={
                "mode": "materials",
                "returned": len(materials),
                "timings_ms": {"total": self._elapsed_ms(started)},
            },
            generation={
                "answer_strategy": "single_writer_call",
                "output_mode": "immutable",
                "raw_model_answer": result.raw_output,
                "writer_trace": self._writer_trace(result, materials),
                "trace_version": "1",
                "trace_stages": [
                    {"stage": "request", "question": question},
                    {
                        "stage": "materials",
                        "source_ids": [source.id for source in materials],
                        "count": len(materials),
                    },
                    {
                        "stage": "writer",
                        "source_ids": [source.id for source in materials],
                    },
                ],
            },
        )

    async def _ask_immutable(self, request: SearchRequest) -> AskResponse:
        started = monotonic()
        display_top_k = min(
            request.top_k or self.settings.default_top_k,
            self.settings.max_top_k,
        )
        evidence_top_k, evidence_policy = self._adaptive_evidence_top_k(
            request.question,
            display_top_k=display_top_k,
        )
        retrieval_started = monotonic()
        planning = await self._immutable_plan(request.question)
        plan = planning.plan
        if plan.analysis.intent == "ordinal":
            fallback_plan = build_query_plan(request.question)
            ordinal_subject = plan.subject
            if fallback_plan.analysis.subjects:
                ordinal_subject = re.sub(
                    r"\s+(?:第一个|第一位|首位|最早)\s+",
                    " ",
                    fallback_plan.analysis.subjects[0],
                ).strip() or ordinal_subject
            ordinal_relations = fallback_plan.relations or (
                "第一个", "第一位", "首位", "最早"
            )
            plan = replace(
                plan,
                subject=ordinal_subject,
                relations=ordinal_relations,
                fields=tuple(
                    replace(field, relations=ordinal_relations)
                    for field in plan.fields
                ),
            )
        if self.settings.answer_point_fanout_enabled and plan.fields:
            evidence_response, extraction, fanout_trace = (
                await self._retrieve_answer_point_branches(
                    request,
                    plan,
                    evidence_top_k=evidence_top_k,
                    started=started,
                )
            )
            passage_expansion = fanout_trace.get("passage_expansion", {})
            extraction_ms = int(fanout_trace.get("extraction_ms", 0))
            fanout_retrieval_ms = int(fanout_trace.get("retrieval_ms", 0))
        else:
            search_method = self.index.search
            result_groups = await asyncio.gather(*(
                asyncio.to_thread(
                    search_method,
                    query,
                    candidate_k=max(
                        request.candidate_k or self.settings.candidate_k,
                        evidence_top_k,
                    ),
                    knowledge_base_id=request.knowledge_base_id,
                )
                for query in plan.queries
            ))
            fused = self._merge_plain_rrf(result_groups)
            if plan.answer_shape == "list" or plan.analysis.intent == "ordinal":
                fused = self._prioritize_topic_document(plan, fused)
            selected = fused[:evidence_top_k]
            selected, passage_expansion = await self._expand_immutable_passages(
                plan,
                selected,
                knowledge_base_id=request.knowledge_base_id,
            )
            evidence_response = SearchResponse(
                results=[self._source_item(result) for result in selected],
                retrieval={
                    "algorithm": "OpenSearch BM25",
                    "mode": "model-query+bm25+document-rrf",
                    "index": self.settings.opensearch_index,
                    "top_k": evidence_top_k,
                    "returned": len(selected),
                    "query_plan": {
                        "queries": list(plan.queries),
                        "subject": plan.subject,
                        "relations": list(plan.relations),
                        "intent": plan.analysis.intent,
                        "answer_shape": plan.answer_shape,
                        "set_semantics": plan.set_semantics,
                        "planner": planning.strategy,
                        "fallback_reason": planning.error,
                    },
                    "document_passage_expansion": passage_expansion,
                    "planner_trace": {
                        "prompt": planning.prompt,
                        "raw_output": planning.raw_output,
                        "prompt_sha256": sha256(planning.prompt.encode("utf-8")).hexdigest(),
                        "raw_output_sha256": sha256(
                            planning.raw_output.encode("utf-8")
                        ).hexdigest(),
                    },
                },
            )
            extraction_started = monotonic()
            extraction = await self._extract_field_evidence(
                request.question,
                plan,
                evidence_response.results,
                timeout=self._remaining_budget(
                    started + self.settings.ask_total_timeout,
                    reserve=self.settings.ask_generation_reserve,
                ),
            )
            fanout_trace = {"enabled": False}
            extraction_ms = self._elapsed_ms(extraction_started)
            fanout_retrieval_ms = 0
        retrieval_ms = self._elapsed_ms(retrieval_started)
        if fanout_retrieval_ms:
            retrieval_ms = fanout_retrieval_ms
        question = request.question
        writer_sources = (
            extraction.answer_sources(evidence_response.results)
            if extraction is not None and extraction.has_candidates
            else []
        )
        writer_sources = self._dedupe_evidence_sources(writer_sources)
        evidence_gate = evaluate_evidence_gate(
            question,
            plan.analysis,
            writer_sources,
            subject=plan.subject,
            relations=plan.relations,
            field_evidence_available=bool(extraction and extraction.available),
            field_candidate_count=len(extraction.candidates) if extraction else 0,
        )
        writer_result: GenerationResult | None = None
        generation_error: str | None = None
        generation_started = monotonic()
        writer_attempted = bool(writer_sources and evidence_gate.passed)
        if writer_attempted:
            try:
                writer_result = await self._generate_with_trace(
                    question,
                    writer_sources,
                    plan=plan,
                    trusted_evidence=True,
                    timeout=self._remaining_budget(started + self.settings.ask_total_timeout),
                )
            except Exception as error:
                generation_error = f"{type(error).__name__}: {error}"
        generation_ms = self._elapsed_ms(generation_started)
        answer = (
            writer_result.answer
            if writer_result is not None
            else "根据检索到的资料，无法确定。"
        )
        validation = validate_grounding(answer, writer_sources)
        answer_support = evaluate_answer_support(
            answer,
            writer_sources,
            question=question,
        )
        display_sources = writer_sources[:display_top_k]
        retrieval = {
            **evidence_response.retrieval,
            "top_k": display_top_k,
            "returned": len(display_sources),
            "answer_evidence_top_k": evidence_top_k,
            "answer_evidence_count": len(writer_sources),
            "evidence_top_k_policy": evidence_policy,
            "timings_ms": {
                "retrieval": retrieval_ms,
                "resolver": extraction_ms,
                "writer": generation_ms,
                "total": self._elapsed_ms(started),
            },
        }
        generation = {
            "answer_strategy": (
                "single_writer_call"
                if writer_result
                else "generation_failed"
                if writer_attempted
                else "evidence_blocked"
                if writer_sources and not evidence_gate.passed
                else "writer_not_called"
            ),
            "output_mode": "immutable",
            "evidence_count": len(writer_sources),
            "displayed_evidence_count": len(display_sources),
            "raw_model_answer": writer_result.raw_output if writer_result else None,
            "generation_error": generation_error,
            "field_evidence_available": bool(extraction and extraction.available),
            "field_evidence": [
                {
                    "field_id": candidate.field_id,
                    "source_index": candidate.source_index + 1,
                    "span": candidate.span,
                    "sha256": candidate.content_hash,
                }
                for candidate in (extraction.candidates if extraction else ())
            ],
            "field_evidence_errors": list(extraction.errors) if extraction else [],
            "field_evidence_strategy": extraction.strategy if extraction else None,
            "evidence_gate_passed": evidence_gate.passed,
            "evidence_gate_issues": list(evidence_gate.issues),
            "evidence_anchors": sorted(evidence_gate.assessment.anchors),
            "matched_evidence_anchors": sorted(evidence_gate.assessment.matched_anchors),
            "matched_evidence_terms": sorted(evidence_gate.assessment.matched_terms),
            "relation_terms": list(evidence_gate.relation_terms),
            "matched_relation_terms": list(evidence_gate.matched_relation_terms),
            "grounding_valid": validation.valid,
            "grounding_issues": list(validation.issues),
            "answer_support_passed": answer_support.passed,
            "answer_support_issues": list(answer_support.issues),
            "writer_trace": self._writer_trace(writer_result, writer_sources)
            if writer_result else None,
            "trace_version": "1",
            "trace_stages": [
                {"stage": "request", "question": request.question},
                {
                    "stage": "retrieval",
                    "source_ids": [source.id for source in evidence_response.results],
                    "count": len(evidence_response.results),
                    "index": self.settings.opensearch_index,
                },
                {
                    "stage": "resolver",
                    "source_ids": [source.id for source in writer_sources],
                    "candidate_count": len(extraction.candidates) if extraction else 0,
                    "errors": list(extraction.errors) if extraction else [],
                    "model_calls": list(extraction.trace_events) if extraction else [],
                },
                {
                    "stage": "evidence_gate",
                    "passed": evidence_gate.passed,
                    "issues": list(evidence_gate.issues),
                    "source_ids": [source.id for source in writer_sources],
                },
                {
                    "stage": "writer",
                    "called": writer_attempted,
                    "source_ids": [source.id for source in writer_sources],
                    "error": generation_error,
                },
                {
                    "stage": "audit",
                    "grounding_valid": validation.valid,
                    "grounding_issues": list(validation.issues),
                    "answer_support_passed": answer_support.passed,
                    "answer_support_issues": list(answer_support.issues),
                    "answer_modified": False,
                },
            ],
        }
        diagnosis = diagnose_failure(
            answer=answer,
            sources=display_sources,
            retrieval=retrieval,
            generation=generation,
        )
        generation.update({
            "failure_category": diagnosis.category,
            "failure_reason": diagnosis.reason,
            "failure_stage": diagnosis.stage,
        })
        return AskResponse(
            answer=answer,
            sources=display_sources,
            retrieval=retrieval,
            generation=generation,
        )

    async def _expand_immutable_passages(
        self,
        plan: QueryPlan,
        selected: list[LexicalResult],
        *,
        knowledge_base_id: str | None,
    ) -> tuple[list[LexicalResult], dict[str, object]]:
        """Expand each RRF-selected document back into relevant passages.

        RRF operates on documents to prevent long documents from voting more
        than once.  The writer, however, needs the passage containing the
        answer.  This second, document-scoped lexical pass restores that
        context while keeping document-level ranking unchanged.
        """

        lookup = getattr(self.index, "document_passage_candidates", None)
        if lookup is None or not selected:
            return selected, {"enabled": False, "documents": []}
        query = " ".join(dict.fromkeys((
            plan.original_question,
            *plan.queries,
            plan.subject,
            *plan.relations,
        )))
        per_document_limit = max(
            4,
            self.settings.list_query_max_chunks_per_document
            if plan.answer_shape == "list"
            else 4,
        )
        document_ids = list(dict.fromkeys(
            result.document_id or result.node_id for result in selected
        ))

        async def expand(document_id: str) -> tuple[str, list[LexicalResult]]:
            passages = await asyncio.to_thread(
                lookup,
                query,
                document_id=document_id,
                knowledge_base_id=knowledge_base_id,
                limit=per_document_limit,
            )
            return document_id, passages

        expanded_groups = await asyncio.gather(*(expand(document_id) for document_id in document_ids))
        by_document = dict(expanded_groups)
        output: list[LexicalResult] = []
        trace_documents: list[dict[str, object]] = []
        seen: set[str] = set()
        for original in selected:
            document_id = original.document_id or original.node_id
            passages = list(by_document.get(document_id) or ())
            if not passages:
                passages = [original]
            elif (
                original.node_id not in {passage.node_id for passage in passages}
                and original.metadata.get("content_type") != "key_value"
            ):
                passages.insert(0, original)

            # Structured chunks often represent one logical table/list row.
            # Reopen a parent only when the index exposes that structure; no
            # content-specific question rule is involved here.
            structure_lookup = getattr(self.index, "structure_chunks", None)
            if structure_lookup is not None:
                parent_ids = list(dict.fromkeys(
                    str(passage.metadata.get("parent_id") or "")
                    for passage in passages
                    if passage.metadata.get("parent_id")
                    and int(passage.metadata.get("structure_size") or 1) > 1
                ))
                for parent_id in parent_ids[:2]:
                    siblings = await asyncio.to_thread(
                        structure_lookup,
                        parent_id,
                        knowledge_base_id=knowledge_base_id,
                        limit=per_document_limit,
                        score=max((passage.score for passage in passages), default=original.score),
                    )
                    by_id = {passage.node_id: passage for passage in passages}
                    for sibling in siblings:
                        by_id.setdefault(sibling.node_id, sibling)
                    passages = list(by_id.values())[:per_document_limit]

            before_ids = [passage.node_id for passage in passages]
            for passage in passages:
                if passage.node_id in seen:
                    continue
                seen.add(passage.node_id)
                output.append(passage)
            trace_documents.append({
                "document_id": document_id,
                "before_node_ids": [original.node_id],
                "after_node_ids": before_ids,
                "expanded_count": len(before_ids),
            })
        return output, {
            "enabled": True,
            "query": query,
            "per_document_limit": per_document_limit,
            "documents": trace_documents,
        }

    async def _retrieve_answer_point_branches(
        self,
        request: SearchRequest,
        plan: QueryPlan,
        *,
        evidence_top_k: int,
        started: float,
    ) -> tuple[SearchResponse, EvidenceExtractionResult | None, dict[str, object]]:
        """Retrieve and extract evidence independently for each answer field."""

        branch_started = monotonic()
        search_method = self.index.search
        fields = plan.fields or (TaskField("f1", request.question, plan.relations),)
        semaphore = asyncio.Semaphore(self.settings.answer_point_fanout_concurrency)

        async def run(field: TaskField) -> dict[str, object]:
            async with semaphore:
                field_question = field.question.strip() or plan.original_question
                field_planning = QueryPlanningResult(
                    plan=replace(plan, fields=(field,)),
                    strategy="parent_plan",
                )
                field_queries = tuple(dict.fromkeys((
                    *plan.queries,
                    field_question,
                    f"{plan.subject} {field_question}".strip(),
                    plan.original_question,
                )))[: self.settings.model_query_planning_max_queries]
                retrieval_started = monotonic()
                groups = await asyncio.gather(*(
                    asyncio.to_thread(
                        search_method,
                        query,
                        candidate_k=max(
                            request.candidate_k or self.settings.candidate_k,
                            evidence_top_k,
                        ),
                        knowledge_base_id=request.knowledge_base_id,
                    )
                    for query in field_queries
                ))
                branch_retrieval_ms = self._elapsed_ms(retrieval_started)
                fused = self._merge_plain_rrf(groups)
                field_plan = replace(
                    plan,
                    fields=(field,),
                    relations=field.relations or plan.relations,
                )
                if field_plan.answer_shape == "list" or field_plan.analysis.intent == "ordinal":
                    fused = self._prioritize_topic_document(field_plan, fused)
                selected = fused[:evidence_top_k]
                selected, expansion = await self._expand_immutable_passages(
                    field_plan,
                    selected,
                    knowledge_base_id=request.knowledge_base_id,
                )
                sources = [self._source_item(result) for result in selected]
                extraction_started = monotonic()
                extraction = await self._extract_field_evidence(
                    request.question,
                    field_plan,
                    sources,
                    timeout=self._remaining_budget(
                        started + self.settings.ask_total_timeout,
                        reserve=self.settings.ask_generation_reserve,
                    ),
                )
                branch_extraction_ms = self._elapsed_ms(extraction_started)
                answer_sources = (
                    extraction.answer_sources(sources)
                    if extraction is not None and extraction.has_candidates
                    else []
                )
                return {
                    "field": field.field_id,
                    "queries": list(field_queries),
                    "planning": {
                        "strategy": field_planning.strategy,
                        "error": field_planning.error,
                        "raw_output": field_planning.raw_output,
                    },
                    "sources": sources,
                    "answer_sources": answer_sources,
                    "extraction": extraction,
                    "retrieval_ms": branch_retrieval_ms,
                    "extraction_ms": branch_extraction_ms,
                    "passage_expansion": expansion,
                }

        results = await asyncio.gather(*(run(field) for field in fields), return_exceptions=True)
        branches: list[dict[str, object]] = []
        merged_sources: list[SourceItem] = []
        extraction_errors: list[str] = []
        extraction_completed = 0
        seen_sources: set[tuple[str, str]] = set()
        for field, result in zip(fields, results, strict=True):
            if isinstance(result, BaseException):
                branches.append({"field": field.field_id, "error": f"{type(result).__name__}: {result}"})
                continue
            branches.append(result)
            for source in result["answer_sources"]:
                key = (source.document_id, source.snippet)
                if key in seen_sources:
                    continue
                seen_sources.add(key)
                merged_sources.append(source)
            extraction = result["extraction"]
            if extraction is not None:
                extraction_completed += extraction.completed_sources
                extraction_errors.extend(extraction.errors)
        extraction_candidates = [
            EvidenceSpan(
                field_id=str((source.metadata.get("evidence_field_ids") or ["f1"])[0]),
                source_index=index,
                span=source.snippet,
                content_hash=sha256(source.snippet.encode("utf-8")).hexdigest(),
            )
            for index, source in enumerate(merged_sources)
        ]
        combined = EvidenceExtractionResult(
            tuple(extraction_candidates),
            attempted_sources=len(merged_sources),
            completed_sources=extraction_completed,
            errors=tuple(extraction_errors),
            strategy="answer_point_fanout",
            trace_events=tuple(
                event
                for branch in branches
                for event in (
                    branch.get("extraction").trace_events
                    if isinstance(branch.get("extraction"), EvidenceExtractionResult)
                    else ()
                )
            ),
        )
        response = SearchResponse(
            results=merged_sources,
            retrieval={
                "algorithm": "OpenSearch BM25",
                "mode": "answer-point-fanout+bm25+document-rrf",
                "index": self.settings.opensearch_index,
                "top_k": evidence_top_k,
                "returned": len(merged_sources),
                "answer_point_fanout": True,
                "branches": [
                    {
                        "field": branch.get("field"),
                        "queries": branch.get("queries", []),
                        "source_count": len(branch.get("sources", [])),
                        "answer_source_count": len(branch.get("answer_sources", [])),
                        "error": branch.get("error"),
                    }
                    for branch in branches
                ],
                "query_plan": {
                    "queries": list(plan.queries),
                    "subject": plan.subject,
                    "relations": list(plan.relations),
                    "intent": plan.analysis.intent,
                    "answer_shape": plan.answer_shape,
                    "set_semantics": plan.set_semantics,
                    "planner": "answer_point_fanout",
                },
            },
        )
        return response, combined, {
            "enabled": True,
            "branches": [
                {
                    "field": branch.get("field"),
                    "queries": branch.get("queries", []),
                    "planning": branch.get("planning", {}),
                    "source_count": len(branch.get("sources", [])),
                    "answer_source_count": len(branch.get("answer_sources", [])),
                    "error": branch.get("error"),
                }
                for branch in branches
            ],
            "passage_expansion": {
                "enabled": True,
                "branches": [branch.get("passage_expansion", {}) for branch in branches],
            },
            "retrieval_ms": max(
                (int(branch.get("retrieval_ms", 0)) for branch in branches),
                default=self._elapsed_ms(branch_started),
            ),
            "extraction_ms": max(
                (int(branch.get("extraction_ms", 0)) for branch in branches),
                default=0,
            ),
        }

    @staticmethod
    def _dedupe_evidence_sources(sources: list[SourceItem]) -> list[SourceItem]:
        """Remove identical resolver evidence while preserving first rank.

        A span can be returned more than once when overlapping chunks or
        multiple query routes point at the same passage.  Passing those copies
        to the writer does not add evidence and can cause repetitive answers.
        The evidence text itself is never edited; only exact duplicate source
        entries are omitted.
        """

        output: list[SourceItem] = []
        seen: set[str] = set()
        for source in sources:
            key = normalize_search_text(source.snippet).strip()
            if key in seen:
                continue
            seen.add(key)
            output.append(source)
        return output

    async def _immutable_plan(self, question: str) -> QueryPlanningResult:
        if self.query_planner is not None and hasattr(
            self.query_planner,
            "plan_immutable",
        ):
            return await self.query_planner.plan_immutable(question)
        plan = build_query_plan(question)
        return QueryPlanningResult(
            plan=replace(
                plan,
                queries=(question,),
                context_policy="none",
                merge_strategy="rank_fusion",
            ),
            strategy="deterministic_fallback",
            error="immutable_model_planner_not_configured",
        )

    @staticmethod
    def _merge_plain_rrf(
        result_groups: tuple[list[LexicalResult], ...],
    ) -> list[LexicalResult]:
        scores: dict[str, float] = {}
        best: dict[str, LexicalResult] = {}
        first_seen: dict[str, int] = {}
        sequence = 0
        for group in result_groups:
            seen_documents: set[str] = set()
            for rank, result in enumerate(group, start=1):
                document_id = result.document_id or result.node_id
                if document_id in seen_documents:
                    continue
                seen_documents.add(document_id)
                scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (60 + rank)
                if document_id not in first_seen:
                    first_seen[document_id] = sequence
                    sequence += 1
                current = best.get(document_id)
                if current is None or result.score > current.score:
                    best[document_id] = result
        ranked_ids = sorted(
            scores,
            key=lambda document_id: (scores[document_id], -first_seen[document_id]),
            reverse=True,
        )
        top_score = max((scores[document_id] for document_id in ranked_ids), default=1.0)
        return [
            replace(
                best[document_id],
                score=scores[document_id] / top_score if top_score else 0.0,
            )
            for document_id in ranked_ids
        ]

    @staticmethod
    def _focus_complete_list_evidence(
        sources: list[SourceItem],
    ) -> list[SourceItem]:
        document_counts: dict[str, int] = {}
        for source in sources:
            hashes = source.metadata.get("evidence_span_hashes")
            evidence_count = len(hashes) if isinstance(hashes, list) else 0
            document_counts[source.document_id] = (
                document_counts.get(source.document_id, 0) + evidence_count
            )
        maximum = max(document_counts.values(), default=0)
        if maximum < 2:
            return sources
        minimum = maximum // 2 + 1
        selected_ids = {
            document_id
            for document_id, count in document_counts.items()
            if count >= minimum
        }
        focused = [
            source for source in sources
            if source.document_id in selected_ids
        ]
        return focused or sources

    @staticmethod
    def _verified_structured_render(
        question: str,
        plan: QueryPlan,
        sources: list[SourceItem],
    ) -> str | None:
        if plan.answer_shape != "list":
            return None
        answer = structured_list_answer(question, sources)
        if answer is None:
            return None
        list_validation = validate_list_answer(question, answer, sources)
        grounding = validate_grounding(answer, sources)
        if list_validation.complete is False or not grounding.valid:
            return None
        return answer

    def _adaptive_evidence_top_k(
        self,
        question: str,
        *,
        display_top_k: int,
    ) -> tuple[int, dict[str, object]]:
        plan = build_query_plan(question)
        analysis = plan.analysis
        target = _ASK_MIN_EVIDENCE_TOP_K
        reason = "simple_fact"
        if plan.answer_shape == "list" and (
            plan.set_semantics == "all" or analysis.expects_complete_list
        ):
            target = 12
            reason = "complete_list"
        elif analysis.intent in {"cause", "comparison"}:
            target = 10
            reason = analysis.intent
        elif analysis.intent == "time" and len(analysis.subjects) > 2:
            target = 10
            reason = "multi_subject_time"
        elif plan.answer_shape == "list" or analysis.intent == "procedure":
            target = 8
            reason = "list" if plan.answer_shape == "list" else analysis.intent
        evidence_top_k = min(
            self.settings.max_top_k,
            max(display_top_k, target),
        )
        return evidence_top_k, {
            "mode": "adaptive",
            "intent": analysis.intent,
            "reason": reason,
            "target": min(target, self.settings.max_top_k),
            "display_top_k": display_top_k,
            "effective_top_k": evidence_top_k,
        }

    async def _generate_answer(
        self,
        question: str,
        sources: list[SourceItem],
        *,
        plan: QueryPlan,
        subject: str,
        relations: tuple[str, ...] = (),
        trusted_evidence: bool = False,
        timeout: float | None = None,
    ) -> str:
        async def generate() -> str:
            if isinstance(self.generator, EvidenceAnswerGenerator):
                return await self.generator.generate(
                    question,
                    sources,
                    subject=subject,
                    relations=relations,
                    trusted_evidence=trusted_evidence,
                    answer_shape=plan.answer_shape,
                    set_semantics=plan.set_semantics,
                    fields=tuple(
                        (field.field_id, field.question, field.relations)
                        for field in plan.fields
                    ),
                )
            return await self.generator.generate(question, sources)

        if timeout is not None:
            if timeout <= 0:
                raise TimeoutError("request answer-generation budget exhausted")
            return await asyncio.wait_for(generate(), timeout=timeout)
        return await generate()

    async def _generate_with_trace(
        self,
        question: str,
        sources: list[SourceItem],
        *,
        plan: QueryPlan,
        trusted_evidence: bool,
        timeout: float | None = None,
    ) -> GenerationResult:
        async def generate() -> GenerationResult:
            if isinstance(self.generator, EvidenceAnswerGenerator):
                return await self.generator.generate_with_trace(
                    question,
                    sources,
                    subject=plan.subject,
                    relations=plan.relations,
                    trusted_evidence=trusted_evidence,
                    answer_shape=plan.answer_shape,
                    set_semantics=plan.set_semantics,
                    fields=tuple(
                        (field.field_id, field.question, field.relations)
                        for field in plan.fields
                    ),
                )
            answer = await self.generator.generate(question, sources)
            return GenerationResult(answer=answer, prompt="", raw_output=answer)

        if timeout is not None:
            if timeout <= 0:
                raise TimeoutError("request answer-generation budget exhausted")
            return await asyncio.wait_for(generate(), timeout=timeout)
        return await generate()

    @staticmethod
    def _writer_trace(
        result: GenerationResult,
        sources: list[SourceItem],
    ) -> dict[str, object]:
        return {
            "prompt": result.prompt,
            "raw_output": result.raw_output,
            "prompt_sha256": result.prompt_sha256,
            "raw_output_sha256": result.raw_output_sha256,
            "source_ids": [source.id for source in sources],
            "source_span_hashes": [
                sha256(source.snippet.encode("utf-8")).hexdigest()
                for source in sources
            ],
            "output_modified": False,
        }

    def _model_raw_output(self, answer: str) -> str:
        return answer

    @staticmethod
    def _trace_stages(
        *,
        request: SearchRequest,
        evidence_response: SearchResponse,
        answer_sources: list[SourceItem],
        field_extraction: EvidenceExtractionResult | None,
        active_retrieval: dict[str, object],
    ) -> list[dict[str, object]]:
        def source_ids(items: list[SourceItem]) -> list[str]:
            return [item.id for item in items]

        return [
            {"stage": "request", "question": request.question},
            {
                "stage": "retrieval",
                "source_ids": source_ids(evidence_response.results),
                "count": len(evidence_response.results),
                "retrieval": evidence_response.retrieval,
            },
            {
                "stage": "active_retrieval",
                "enabled": active_retrieval.get("enabled", False),
                "rounds": active_retrieval.get("rounds", []),
            },
            {
                "stage": "resolver",
                "source_ids": source_ids(answer_sources),
                "count": len(answer_sources),
                "candidate_count": len(field_extraction.candidates) if field_extraction else 0,
                "errors": list(field_extraction.errors) if field_extraction else [],
            },
            {"stage": "writer", "source_ids": source_ids(answer_sources)},
        ]

    @staticmethod
    def _deterministic_answer(
        question: str,
        plan: QueryPlan,
        sources: list[SourceItem],
    ) -> str | None:
        analysis = plan.analysis
        answer: str | None = None
        if plan.answer_shape == "list":
            answer = direct_evidence_answer(question, sources)
            structured_answer = list_evidence_answer(question, sources)
            if answer is None or (
                structured_answer is not None
                and not _answer_covers_subject_topics(answer, plan.subject)
            ):
                answer = structured_answer
        elif analysis.intent == "definition":
            answer = definition_evidence_answer(question, sources)
        elif analysis.intent == "time":
            answer = coordinated_time_evidence_answer(
                sources,
                analysis.subjects,
            ) or time_evidence_answer(question, sources)
        elif analysis.intent == "agent":
            answer = agent_evidence_answer(question, sources, plan.relations)
        elif analysis.intent == "ordinal":
            answer = ordinal_evidence_answer(question, sources)
        elif analysis.intent == "location":
            answer = location_evidence_answer(question, sources)
        elif analysis.intent == "birthplace":
            answer = birthplace_evidence_answer(question, sources)
        if answer is not None:
            return answer
        if plan.answer_shape not in {"single_fact", "list"}:
            return None
        return direct_evidence_answer(question, sources)

    @staticmethod
    def _field_evidence_quote_answer(
        sources: list[SourceItem],
        relations: tuple[str, ...],
    ) -> str | None:
        normalized_relations = {
            normalize_search_text(relation).replace(" ", "")
            for relation in relations
            if len(relation.strip()) >= 2
        }
        if not normalized_relations:
            return None
        quotes: list[str] = []
        for source_index, source in enumerate(sources, start=1):
            units = [
                unit.strip()
                for unit in re.split(r"(?<=[。！？!?；;])|\n+", clean_evidence_text(source.snippet))
                if unit.strip()
            ]
            for unit in units:
                normalized = normalize_search_text(unit).replace(" ", "")
                if not any(relation in normalized for relation in normalized_relations):
                    continue
                quote = unit.rstrip("。！？!?；;")
                quotes.append(f"{quote}[资料 {source_index}]")
                break
            if len(quotes) >= 4:
                break
        if not quotes:
            outcome_markers = (
                "结局", "結局", "结尾", "結尾", "终结", "終結", "结束", "結束",
                "最终", "最終", "最后", "最後", "结果", "結果", "归一", "歸一",
                "一统", "一統", "统一", "統一", "灭亡", "滅亡",
            )
            if not any(
                marker in normalize_search_text(" ".join(relations))
                for marker in outcome_markers
            ):
                return None
            for source_index, source in enumerate(sources, start=1):
                units = [
                    unit.strip()
                    for unit in re.split(r"(?<=[。！？!?；;])|\n+", clean_evidence_text(source.snippet))
                    if unit.strip()
                ]
                for unit in units:
                    normalized = normalize_search_text(unit)
                    if not any(marker in normalized for marker in outcome_markers):
                        continue
                    quotes.append(f"{unit.rstrip('。！？!?；;')}[资料 {source_index}]")
                    if len(quotes) >= 3:
                        break
                if len(quotes) >= 3:
                    break
        if not quotes:
            return None
        return "根据检索资料，可确认：" + "；".join(quotes) + "。"

    @staticmethod
    def _evidence_fallback_answer(
        question: str,
        plan: QueryPlan,
        sources: list[SourceItem],
    ) -> str | None:
        """Render a bounded extractive answer when a model stage fails."""

        if plan.answer_shape == "list":
            structured = structured_list_answer(question, sources)
            if structured is not None:
                return structured
        return SearchService._field_evidence_quote_answer(
            sources,
            plan.relations,
        )

    @staticmethod
    def _extraction_covers_fallback(
        extraction: EvidenceExtractionResult,
        fallback: str,
    ) -> bool:
        candidate_terms = set(query_tokens(
            " ".join(candidate.span for candidate in extraction.candidates)
        ))
        fallback_terms = set(query_tokens(fallback))
        fallback_terms -= {"根据", "检索", "资料", "确认", "可确认"}
        return len(candidate_terms & fallback_terms) >= 3

    async def _extract_field_evidence(
        self,
        question: str,
        plan: QueryPlan,
        sources: list[SourceItem],
        *,
        timeout: float | None = None,
    ) -> EvidenceExtractionResult | None:
        if self.evidence_extractor is None:
            return None
        try:
            if timeout is not None:
                if timeout <= 0:
                    raise TimeoutError("request evidence-extraction budget exhausted")
                result = await asyncio.wait_for(
                    self.evidence_extractor.extract(question, plan, sources),
                    timeout=timeout,
                )
            else:
                result = await self.evidence_extractor.extract(question, plan, sources)
            if (
                not self.settings.semantic_pipeline_enabled
                and result is not None
                and not result.has_candidates
                and plan.answer_shape in {"list", "single_fact", "summary", "narrative"}
            ):
                result = self._deterministic_structured_evidence(result, plan, sources)
            return result
        except Exception as error:
            failed = EvidenceExtractionResult(
                (), 0, 0, (f"{type(error).__name__}: {error}",)
            )
            if (
                not self.settings.semantic_pipeline_enabled
                and plan.answer_shape in {"list", "single_fact", "summary", "narrative"}
            ):
                return self._deterministic_structured_evidence(
                    failed,
                    plan,
                    sources,
                )
            return failed

    @staticmethod
    def _field_extraction_failed(
        extraction: EvidenceExtractionResult | None,
    ) -> bool:
        """Return true only for an extractor error, not a valid empty result."""

        return bool(
            extraction is not None
            and extraction.errors
            and not extraction.has_candidates
        )

    @staticmethod
    def _deterministic_structured_evidence(
        result: EvidenceExtractionResult,
        plan: QueryPlan,
        sources: list[SourceItem],
    ) -> EvidenceExtractionResult:
        subject_terms = {
            normalize_search_text(value).replace(" ", "")
            for value in (plan.subject,)
            if value.strip()
        }
        relation_terms = {
            token
            for token in query_tokens(" ".join((
                *plan.relations,
                *(field.question for field in plan.fields),
            )))
            if len(token.strip()) >= 2
        }
        relation_terms -= {
            token
            for token in query_tokens(plan.subject)
        }
        relation_terms -= {
            "什么", "哪些", "哪个", "哪几个", "怎么", "如何", "多少",
            "主要", "著名", "伟大", "全部", "所有", "列表",
        }
        definition_markers = ("是指", "即", "包括", "包含", "分别是", "分别为")
        summary_terminal_markers = (
            "结局", "結局", "结尾", "結尾", "终结", "終結", "结束", "結束",
            "归一", "歸一", "统一天下", "統一天下", "最终结果", "最終結果",
            "最后结果", "最後結果", "灭亡", "滅亡", "完成",
        )
        scored_candidates: list[tuple[float, int, EvidenceSpan]] = []
        for source_index, source in enumerate(sources):
            normalized_title = normalize_search_text(source.title).replace(" ", "")
            title_matches = any(
                title_matches_subject(source.title, term)
                or title_matches_subject_topic(source.title, term)
                or term in document_aliases(source.title, source.metadata)
                for term in subject_terms
            )
            normalized_body = normalize_search_text(source.snippet).replace(" ", "")
            body_matches = any(term in normalized_body for term in subject_terms)
            transit_match = re.fullmatch(
                r"(?P<network>[\u3400-\u9fff]{2,20}?地铁)(?P<line>\d+号线)",
                next(iter(subject_terms), ""),
            )
            transit_companion_matches = bool(
                plan.answer_shape == "list"
                and transit_match
                and normalized_title == f"{transit_match.group('network')}车站列表"
                and transit_match.group("line") in normalized_body
            )
            if not title_matches and not body_matches and not transit_companion_matches:
                continue
            source_context = " ".join((
                str(source.metadata.get("section") or ""),
                source.title,
            ))
            source_relation_tokens = set(query_tokens(source_context)) & relation_terms
            units = [
                unit.strip()
                for unit in re.split(r"(?<=[。！？!?；;])|\n+", source.snippet)
                if unit.strip()
            ]
            for unit in units:
                normalized = normalize_search_text(unit).replace(" ", "")
                if normalized in subject_terms or normalized == normalized_title:
                    continue
                unit_tokens = set(query_tokens(unit))
                if unit_tokens and unit_tokens <= relation_terms:
                    continue
                if (
                    plan.analysis.intent == "agent"
                    and len(normalized) <= 16
                    and not any(marker in normalized for marker in (
                        "是", "为", "由", "被", "害死", "杀害", "打死", "处死", "发起",
                    ))
                    and not any(term in normalized for term in subject_terms)
                ):
                    continue
                has_requested_relation = bool(unit_tokens & relation_terms)
                has_definition_relation = (
                    plan.analysis.intent == "definition"
                    and any(marker in normalized for marker in definition_markers)
                )
                list_like = bool(
                    source.metadata.get("content_type") in {"table_summary", "table", "list"}
                    or len(re.findall(r"[、，,；;]", unit)) >= 2
                    or re.match(r"^[-*•]\s*", unit)
                    or re.match(r"^[^：:\n]{1,24}[：:]", unit)
                )
                supported_by_structure = bool(
                    plan.answer_shape == "list"
                    and list_like
                    and (source_relation_tokens or has_requested_relation)
                )
                if not (
                    has_requested_relation
                    or has_definition_relation
                    or supported_by_structure
                ):
                    continue
                evidence = EvidenceSpan(
                    field_id=plan.fields[0].field_id,
                    source_index=source_index,
                    span=unit,
                    content_hash=sha256(unit.encode("utf-8")).hexdigest(),
                )
                score = float(has_requested_relation or has_definition_relation)
                if plan.answer_shape in {"summary", "narrative"}:
                    score += sum(
                        5.0
                        for marker in summary_terminal_markers
                        if marker in normalized
                    )
                    score += min(3.0, sum(
                        normalized.count(relation)
                        for relation in relation_terms
                        if len(relation) >= 2
                    ))
                    try:
                        score += min(
                            2.0,
                            int(source.metadata.get("chunk_order") or 0) / 20.0,
                        )
                    except (TypeError, ValueError):
                        pass
                scored_candidates.append((score, source_index, evidence))
                if (
                    plan.answer_shape not in {"summary", "narrative"}
                    and len(scored_candidates) >= 12
                ):
                    break
            if (
                plan.answer_shape not in {"summary", "narrative"}
                and len(scored_candidates) >= 12
            ):
                break
        if plan.answer_shape in {"summary", "narrative"}:
            scored_candidates.sort(key=lambda item: (-item[0], item[1]))
        candidates = [item[2] for item in scored_candidates[:12]]
        if not candidates:
            return result
        signatures = tuple(
            (item.id, sha256(item.snippet.encode("utf-8")).hexdigest())
            for item in sources
        )
        return EvidenceExtractionResult(
            candidates=tuple(candidates),
            attempted_sources=len(sources),
            completed_sources=max(result.completed_sources, len(sources)),
            errors=result.errors,
            source_signatures=signatures,
            strategy=(
                "deterministic_definition_fallback"
                if plan.analysis.intent == "definition"
                else "deterministic_structured_fallback"
            ),
        )

    async def _answer_guided_verification(
        self,
        request: SearchRequest,
        plan: QueryPlan,
        unsupported_terms: tuple[str, ...],
        original_sources: list[SourceItem],
        *,
        evidence_top_k: int,
        deadline: float,
    ) -> tuple[list[SourceItem], EvidenceExtractionResult | None, list[str]]:
        question_terms = set(lexical_tokens(request.question))
        hypotheses = [
            term.strip()
            for term in sorted(
                unsupported_terms,
                key=lambda value: (value in question_terms, -len(value)),
            )
            if len(term.strip()) >= 2
            and term.strip() not in {"根据资料", "无法确定", "问题", "答案"}
        ][:3]
        if not hypotheses:
            return [], None, []
        relation_hints = [
            relation.strip()
            for relation in plan.relations
            if len(relation.strip()) >= 2
            and relation.strip() not in hypotheses
            and relation.strip() not in plan.subject
        ][:3]
        queries = list(dict.fromkeys(
            query
            for hypothesis in hypotheses
            for query in (
                f"{plan.subject} {hypothesis}".strip(),
                *(
                    f"{hypothesis} {relation}".strip()
                    for relation in relation_hints
                ),
            )
            if query
        ))
        responses = await asyncio.gather(*(
            self.search(
                SearchRequest(
                    question=request.question,
                    top_k=evidence_top_k,
                    candidate_k=request.candidate_k,
                    min_score=request.min_score,
                    knowledge_base_id=request.knowledge_base_id,
                ),
                use_model_planner=False,
                query_override=(query,),
            )
            for query in queries
        ), return_exceptions=True)
        groups = [
            self._collapse_verification_chunks(
                response.results,
                hypotheses=hypotheses,
                relations=relation_hints,
            )
            for response in responses
            if isinstance(response, SearchResponse)
        ]
        if not groups:
            return [], None, queries
        merged = self._merge_active_sources(
            groups,
            original_sources,
            limit=self.settings.active_retrieval_max_results,
        )
        verification_plan = replace(
            plan,
            relations=tuple(dict.fromkeys((*hypotheses, *plan.relations))),
            fields=tuple(
                replace(
                    field,
                    relations=tuple(dict.fromkeys((*hypotheses, *field.relations))),
                )
                for field in plan.fields
            ),
        )
        extraction = await self._extract_field_evidence(
            request.question,
            verification_plan,
            merged,
            timeout=self._remaining_budget(deadline),
        )
        if extraction is None or not extraction.has_candidates:
            return [], extraction, queries
        return extraction.answer_sources(merged), extraction, queries

    @staticmethod
    def _collapse_verification_chunks(
        sources: list[SourceItem],
        *,
        hypotheses: list[str],
        relations: list[str],
    ) -> list[SourceItem]:
        verification_terms = set(lexical_tokens(" ".join((*hypotheses, *relations))))
        grouped: OrderedDict[str, list[tuple[int, SourceItem]]] = OrderedDict()
        for index, source in enumerate(sources):
            grouped.setdefault(source.document_id, []).append((index, source))

        output: list[SourceItem] = []
        for candidates in grouped.values():
            _, best = max(
                candidates,
                key=lambda item: (
                    len(verification_terms & set(lexical_tokens(item[1].snippet))),
                    int(bool(re.search(r"[。！？；，,.!?;]", item[1].snippet))),
                    int("category:" not in item[1].snippet.lower()),
                    int(len(item[1].snippet.strip()) >= 80),
                    min(len(item[1].snippet), 1_000),
                    -item[0],
                ),
            )
            output.append(best)
        return output

    def _answer_plan(
        self,
        question: str,
        retrieval: dict[str, object],
    ) -> QueryPlan:
        fallback = build_query_plan(question)
        query_plan = retrieval.get("query_plan")
        if not isinstance(query_plan, dict) or query_plan.get("planner") != "model":
            return fallback
        subject = query_plan.get("subject")
        intent = query_plan.get("intent")
        answer_shape = query_plan.get("answer_shape")
        set_semantics = query_plan.get("set_semantics")
        relations = query_plan.get("relations")
        raw_fields = query_plan.get("fields")
        if not all(isinstance(value, str) and value.strip() for value in (
            subject, intent, answer_shape, set_semantics,
        )):
            return fallback
        if not isinstance(relations, list) or not isinstance(raw_fields, list):
            return fallback
        fields: list[TaskField] = []
        for item in raw_fields:
            if not isinstance(item, dict):
                continue
            field_id = item.get("field_id")
            field_question = item.get("question")
            field_relations = item.get("relations")
            if not isinstance(field_id, str) or not isinstance(field_question, str):
                continue
            if not isinstance(field_relations, list):
                continue
            fields.append(TaskField(
                field_id.strip(),
                field_question.strip(),
                tuple(str(value).strip() for value in field_relations if str(value).strip()),
            ))
        if not fields:
            return fallback
        model_relations = tuple(
            str(value).strip() for value in relations if str(value).strip()
        )
        merged_relations = (
            model_relations
            if self.settings.semantic_pipeline_enabled
            else tuple(dict.fromkeys((*model_relations, *fallback.relations)))
        )
        planned = replace(
            fallback,
            subject=subject.strip(),
            relations=merged_relations,
            fields=tuple(fields),
            answer_shape=answer_shape.strip(),
            set_semantics=set_semantics.strip(),
            analysis=replace(
                fallback.analysis,
                intent=intent.strip(),
                expects_list=answer_shape == "list",
                expects_complete_list=answer_shape == "list" and set_semantics == "all",
            ),
        )
        if (
            not self.settings.semantic_pipeline_enabled
            and fallback.analysis.intent in {"comparison", "cause"}
        ):
            return replace(
                planned,
                subject=fallback.subject,
                relations=fallback.relations,
                fields=fallback.fields,
                answer_shape=fallback.answer_shape,
                set_semantics=fallback.set_semantics,
                analysis=fallback.analysis,
            )
        return planned

    @staticmethod
    def _planned_subject(
        retrieval: dict[str, object],
    ) -> str:
        query_plan = retrieval.get("query_plan")
        if not isinstance(query_plan, dict) or query_plan.get("planner") != "model":
            return ""
        subject = query_plan.get("subject")
        return str(subject).strip() if isinstance(subject, str) else ""

    @staticmethod
    def _planned_relations(
        retrieval: dict[str, object],
    ) -> tuple[str, ...]:
        query_plan = retrieval.get("query_plan")
        if not isinstance(query_plan, dict):
            return ()
        relations = query_plan.get("relations")
        if not isinstance(relations, list):
            return ()
        return tuple(dict.fromkeys(
            str(relation).strip()
            for relation in relations
            if str(relation).strip()
        ))

    @staticmethod
    def _display_sources(
        sources: list[SourceItem],
        answer: str,
        *,
        display_top_k: int,
    ) -> list[SourceItem]:
        required_indexes = {
            int(match.group(1)) - 1
            for match in _CITATION_INDEX_PATTERN.finditer(answer)
            if int(match.group(1)) >= 1
        }
        selected: list[SourceItem] = []
        for index, source in enumerate(sources):
            if index < display_top_k or index in required_indexes:
                selected.append(source)
        return selected

    @staticmethod
    def _answer_cache_key(
        question: str,
        response: SearchResponse,
        plan: QueryPlan,
    ) -> tuple[object, ...]:
        evidence = tuple(
            (source.id, round(source.score, 6), source.snippet[:256])
            for source in response.results
        )
        contract = (
            plan.subject,
            plan.relations,
            plan.analysis.intent,
            plan.answer_shape,
            plan.set_semantics,
        )
        return (
            question,
            response.retrieval.get("knowledge_base_id"),
            contract,
            evidence,
        )

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

    @staticmethod
    def _is_empty_answer_shell(answer: str) -> bool:
        return bool(_EMPTY_ANSWER_SHELL.search(answer.strip()))

    @staticmethod
    def _answer_contract_failed(question: str, answer: str) -> bool:
        answer_without_citations = re.sub(r"\[资料\s*\d+\]", "", answer).strip()
        normalized_answer = normalize_search_text(answer_without_citations).replace(" ", "")
        normalized_question = normalize_search_text(question).replace(" ", "")
        if not normalized_answer or normalized_answer == normalized_question:
            return True
        if re.search(
            r"(?:结局|結局|结尾|結尾|终局|終局|结果|結果|原因|作者|首都|国都)"
            r"(?:是|为|為|：|:)"
            r"(?:最终|最終|结局|結局|结尾|結尾|终局|終局|结果|結果|原因|作者|首都|国都)",
            normalized_answer,
        ):
            return True
        if is_repetitive_garbage(answer_without_citations):
            return True
        list_body = answer_without_citations.split("：", 1)[-1]
        items = [
            normalize_search_text(item).replace(" ", "")
            for item in re.split(r"[、，,；;\n]+", list_body)
            if normalize_search_text(item).replace(" ", "")
        ]
        return len(items) >= 8 and len(set(items)) / len(items) < 0.7

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
        flattened_station_context = await self._flattened_station_list_context(
            question,
            results,
            knowledge_base_id=knowledge_base_id,
        )
        if flattened_station_context:
            context_ids = {result.node_id for result in flattened_station_context}
            context_document_ids = {
                result.document_id for result in flattened_station_context
            }
            return [
                *flattened_station_context,
                results[0],
                *(
                    result
                    for result in results[1:]
                    if result.node_id not in context_ids
                    and result.document_id not in context_document_ids
                ),
            ], True
        deterministic_plan = build_query_plan(question)
        normalized_subject = normalize_search_text(
            deterministic_plan.subject
        ).replace(" ", "")
        subject_document_ids = {
            result.document_id
            for result in results
            if normalized_subject
            and (
                title_matches_subject(
                    str(result.metadata.get("title") or ""),
                    normalized_subject,
                )
                or title_matches_subject_topic(
                    str(result.metadata.get("title") or ""),
                    normalized_subject,
                )
                or normalized_subject in document_aliases(
                    str(result.metadata.get("title") or ""),
                    result.metadata,
                )
            )
        }
        # Structured expansion needs a document that represents the requested
        # topic.  The first BM25 hit can be a broad scope page (for example
        # ``中国``) even when a more specific ``中国朝代``/``中国民族列表``
        # page is present later in the fused candidates.  Pick the best
        # document deterministically before loading its structure; this keeps
        # the behavior independent of query wording and does not alter the
        # normal non-structured ranking path.
        preferred_document_id: str | None = None
        if content_types:
            topic_tokens = self._topic_tokens(
                question,
                deterministic_plan.subject,
            )
            document_order: dict[str, int] = {}
            document_results: dict[str, list[LexicalResult]] = {}
            for index, result in enumerate(results):
                document_id = result.document_id or result.node_id
                document_order.setdefault(document_id, index)
                document_results.setdefault(document_id, []).append(result)

            def topic_priority(document_id: str) -> tuple[float, int]:
                candidates = document_results[document_id]
                title = str(candidates[0].metadata.get("title") or "")
                normalized_title = normalize_search_text(title).replace(" ", "")
                exact_subject = title_matches_subject(title, normalized_subject)
                title_subject_overlap = len(
                    set(lexical_tokens(deterministic_plan.subject))
                    & set(lexical_tokens(title))
                )
                core_tokens = _core_subject_tokens(deterministic_plan.subject)
                title_topic_hits = sum(
                    token in normalized_title
                    for token in topic_tokens
                )
                body_topic_hits = sum(
                    token in normalize_search_text(
                        "\n".join(result.text for result in candidates[:3])
                    )
                    for token in topic_tokens
                )
                return (
                    (8.0 if exact_subject else 0.0)
                    + title_subject_overlap * 4.0
                    + (6.0 if core_tokens and core_tokens[0] in set(lexical_tokens(title)) else 0.0)
                    + title_topic_hits * 10.0
                    + min(body_topic_hits, 3) * 0.25,
                    -document_order[document_id],
                )

            if document_results:
                best_document = max(document_results, key=topic_priority)
                if topic_priority(best_document)[0] > 0:
                    subject_document_ids.add(best_document)
                    preferred_document_id = best_document
        top_document = preferred_document_id or next(
            (
                result.document_id
                for result in results
                if result.document_id in subject_document_ids
            ),
            results[0].document_id,
        )
        candidates = [
            result
            for result in results
            if result.metadata.get("parent_id")
            and result.metadata.get("content_type") in content_types
            and (
                not subject_document_ids
                or result.document_id in subject_document_ids
            )
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
        anchor_candidates = (
            same_document
            if preferred_document_id is not None
            else [*same_document, *candidates]
        )
        anchor_pool = list({
            result.node_id: result
            for result in anchor_candidates
        }.values())
        anchor = max(
            anchor_pool,
            key=lambda result: self._structure_relevance(question, result),
            default=None,
        )
        if anchor is None:
            repair_context = await self._station_repair_context(
                question,
                results,
                document_id=top_document,
                knowledge_base_id=knowledge_base_id,
            )
            if repair_context:
                seen = {result.node_id for result in repair_context}
                return [
                    results[0],
                    *repair_context,
                    *(result for result in results[1:] if result.node_id not in seen),
                ], True
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
        prose_candidates: list[LexicalResult] = []
        if any(marker in question for marker in _CHRONOLOGICAL_LIST_MARKERS):
            analysis = analyze_question(question)
            prose_query = analysis.subjects[0] if analysis.subjects else question
            prose_candidates = await self._document_prose_candidates(
                prose_query,
                document_id=anchor.document_id,
                knowledge_base_id=knowledge_base_id,
            )
        repair_context = await self._station_repair_context(
            question,
            [*expanded, *results],
            document_id=top_document,
            knowledge_base_id=knowledge_base_id,
        )
        if repair_context:
            expanded = [expanded[0], *repair_context, *expanded[1:]]
        expanded_ids = {result.node_id for result in expanded}
        prose_candidates = [
            result for result in prose_candidates if result.node_id not in expanded_ids
        ]
        seen = {*expanded_ids, *(result.node_id for result in prose_candidates)}
        merged = [
            *expanded,
            *prose_candidates,
            *(result for result in results if result.node_id not in seen),
        ]
        return merged, True

    @staticmethod
    def _prioritize_topic_document(
        plan: QueryPlan,
        results: list[LexicalResult],
    ) -> list[LexicalResult]:
        """Move a specific topic page ahead of broad scope pages.

        BM25/RRF can legitimately rank a broad page such as ``中国`` above a
        page titled ``中国民族列表`` because the broad page repeats the scope
        term many times.  When the query contains a distinct topic token, a
        title hit is stronger evidence of the requested object than body
        frequency.  Only whole-document ordering changes; chunk order inside a
        document remains untouched.
        """

        if not results or not plan.subject:
            return results
        subject_tokens = set(lexical_tokens(plan.subject))
        topic_tokens = SearchService._topic_tokens(
            plan.normalized_question,
            plan.subject,
        )
        documents: dict[str, list[LexicalResult]] = {}
        order: dict[str, int] = {}
        for index, result in enumerate(results):
            document_id = result.document_id or result.node_id
            documents.setdefault(document_id, []).append(result)
            order.setdefault(document_id, index)

        def priority(document_id: str) -> tuple[float, int]:
            chunks = documents[document_id]
            title = str(chunks[0].metadata.get("title") or "")
            normalized_title = normalize_search_text(title).replace(" ", "")
            exact_subject = title_matches_subject(title, normalize_search_text(plan.subject).replace(" ", ""))
            title_hits = sum(token in normalized_title for token in topic_tokens)
            query_title_hits = sum(
                token in normalized_title
                for token in query_tokens(" ".join(plan.queries))
                if token not in subject_tokens and len(token) >= 2
            )
            relation_title_hits = sum(
                normalize_search_text(relation).replace(" ", "") in normalized_title
                for relation in plan.relations
                if relation.strip()
            )
            subject_title_hits = len(
                subject_tokens & set(lexical_tokens(title))
            )
            core_tokens = _core_subject_tokens(plan.subject)
            body_hits = sum(
                token in normalize_search_text("\n".join(chunk.text for chunk in chunks[:3]))
                for token in topic_tokens
            )
            score = (
                (8.0 if exact_subject else 0.0)
                + subject_title_hits * 4.0
                + (6.0 if core_tokens and core_tokens[0] in set(lexical_tokens(title)) else 0.0)
                + title_hits * 14.0
                + query_title_hits * 8.0
                + relation_title_hits * 8.0
                + min(body_hits, 3) * 0.25
            )
            return score, -order[document_id]

        best_document = max(documents, key=priority)
        if priority(best_document)[0] <= 0 or best_document == next(iter(documents)):
            return results
        return [
            *documents[best_document],
            *(result for result in results if result.document_id != best_document),
        ]

    async def _flattened_station_list_context(
        self,
        question: str,
        results: list[LexicalResult],
        *,
        knowledge_base_id: str | None,
    ) -> list[LexicalResult]:
        if not _is_station_list_question(question):
            return []
        line_match = re.search(r"\d+号线", normalize_search_text(question))
        if line_match is None:
            line_match = next(
                (
                    match
                    for result in results
                    if (match := re.search(r"\d+号线", normalize_search_text(str(result.metadata.get("title") or ""))))
                ),
                None,
            )
        if line_match is None:
            return []
        companion = next(
            (
                result
                for result in results
                if normalize_search_text(str(result.metadata.get("title") or "")).endswith("车站列表")
            ),
            None,
        )
        search_lookup = getattr(self.index, "search", None)
        if companion is None and search_lookup is not None:
            network = next(
                (
                    match.group("network")
                    for result in results
                    if (
                        match := re.search(
                            r"(?P<network>[\u3400-\u9fff]{2,20}?地铁)\d+号线",
                            normalize_search_text(str(result.metadata.get("title") or "")),
                        )
                    )
                ),
                "",
            )
            if network:
                candidates = await asyncio.to_thread(
                    search_lookup,
                    f"{network}车站列表 {line_match.group(0)}",
                    candidate_k=20,
                    knowledge_base_id=knowledge_base_id,
                )
                expected_title = normalize_search_text(f"{network}车站列表").replace(" ", "")
                companion = next(
                    (
                        result
                        for result in candidates
                        if normalize_search_text(
                            str(result.metadata.get("title") or "")
                        ).replace(" ", "") == expected_title
                    ),
                    None,
                )
        lookup = getattr(self.index, "document_line_section_chunks", None)
        if companion is None or lookup is None:
            return []
        return await asyncio.to_thread(
            lookup,
            line_match.group(0),
            document_id=companion.document_id,
            knowledge_base_id=knowledge_base_id,
            limit=3,
            score=max(companion.score, results[0].score),
        )

    async def _document_prose_candidates(
        self,
        question: str,
        *,
        document_id: str,
        knowledge_base_id: str | None,
    ) -> list[LexicalResult]:
        lookup = getattr(self.index, "document_prose_candidates", None)
        if lookup is None:
            return []
        return await asyncio.to_thread(
            lookup,
            question,
            document_id=document_id,
            knowledge_base_id=knowledge_base_id,
            limit=max(3, self.settings.list_query_max_chunks_per_document),
        )

    async def _station_repair_context(
        self,
        question: str,
        expanded: list[LexicalResult],
        *,
        document_id: str,
        knowledge_base_id: str | None,
    ) -> list[LexicalResult]:
        if not _is_station_list_question(question):
            return []
        lookup = getattr(self.index, "document_term_candidates", None)
        title_lookup = getattr(self.index, "station_title_candidates", None)
        if lookup is None and title_lookup is None:
            return []
        names: list[str] = []
        for result in expanded:
            for match in re.finditer(r"站名[^：:\n]{0,40}?列表[：:]\s*([^\n]+)", result.text):
                for item in re.split(r"[、，,；;]", match.group(1)):
                    name = re.sub(r"\[\d{1,4}\]", "", item).strip(" 。；;，,")
                    if len(name) == 1 and name not in names:
                        names.append(name)
        context: list[LexicalResult] = []
        seen = {result.node_id for result in expanded}
        for name in names[:3]:
            candidates = []
            if lookup is not None:
                candidates = await asyncio.to_thread(
                    lookup,
                    name,
                    document_id=document_id,
                    knowledge_base_id=knowledge_base_id,
                    limit=100,
                )
            for candidate in candidates:
                if candidate.node_id in seen or f"{name}站" not in candidate.text:
                    continue
                seen.add(candidate.node_id)
                context.append(candidate)
                break
            else:
                if title_lookup is None:
                    continue
                title_candidates = await asyncio.to_thread(
                    title_lookup,
                    name,
                    question=question,
                    knowledge_base_id=knowledge_base_id,
                    limit=10,
                )
                if title_candidates:
                    context.append(title_candidates[0])
        return context

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
        all_question_tokens = set(query_tokens(question))
        question_tokens = (
            all_question_tokens - title_tokens - _STRUCTURE_QUESTION_WORDS
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
                elif not any(hint in normalized_context for hint in _LIST_TOPIC_HINTS):
                    list_bonus -= 3.0
                if any(hint in normalized_context for hint in _STATION_STRUCTURE_NOISE):
                    list_bonus -= 4.0
                if any(hint in normalized_context for hint in _TRANSFER_HINTS) and not any(
                    hint in question for hint in _TRANSFER_HINTS
                ):
                    list_bonus -= 2.6
            if any(hint in normalized_context for hint in _EXPLANATORY_SECTION_HINTS):
                list_bonus -= 1.0
        title_overlap = len(title_tokens & all_question_tokens)
        title_bonus = min(2.4, title_overlap * 0.6)
        analysis = analyze_question(question)
        if analysis.intent == "list" and analysis.subjects:
            scope = analysis.subjects[0].split(" ", 1)[0]
            scope_tokens = set(lexical_tokens(scope))
            if scope_tokens and not (scope_tokens & title_tokens):
                title_bonus -= 6.0
        score = coverage * 3.0 + type_bonus + list_bonus + title_bonus
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

    def _max_chunks_per_document(self, question: str, *, top_k: int | None = None) -> int:
        intent = analyze_question(question).intent
        if intent == "list" and any(
            marker in question
            for marker in (*_MULTI_EVIDENCE_MARKERS, "哪几个", "有哪几个")
        ):
            return max(
                self.settings.max_chunks_per_document,
                self.settings.list_query_max_chunks_per_document,
                top_k or 0,
            )
        if intent in {"agent", "time", "cause", "location"}:
            return max(self.settings.max_chunks_per_document, min(top_k or 3, 4))
        return self.settings.max_chunks_per_document

    @staticmethod
    def _trim_post_event_evidence(
        sources: list[SourceItem],
        *,
        subject: str,
        relations: tuple[str, ...],
    ) -> list[SourceItem]:
        normalized_subject = normalize_search_text(subject).replace(" ", "")
        event_terms = [
            normalize_search_text(relation).replace(" ", "")
            for relation in relations
            if relation not in {"原因", "导致"}
        ]
        if not normalized_subject or not event_terms:
            return sources
        patterns = [
            re.compile(
                rf"(?:{re.escape(normalized_subject)})?"
                rf"{re.escape(event)}(?:后|之后|以后)"
            )
            for event in event_terms
        ]
        trimmed: list[SourceItem] = []
        for source in sources:
            normalized = normalize_search_text(source.snippet).replace(" ", "")
            positions = [
                match.start()
                for pattern in patterns
                if (match := pattern.search(normalized)) is not None
            ]
            if not positions:
                trimmed.append(source)
                continue
            cutoff = min(positions)
            snippet = source.snippet[:cutoff].rstrip(" ，,；;。")
            if snippet:
                trimmed.append(source.model_copy(update={"snippet": snippet}))
        return trimmed

    @staticmethod
    def _focus_sources_on_subject(
        plan: QueryPlan,
        sources: list[SourceItem],
    ) -> list[SourceItem]:
        if not plan.subject or plan.analysis.intent == "comparison":
            return sources
        if plan.analysis.intent == "time" and len(plan.analysis.subjects) > 2:
            return sources
        normalized_subject = normalize_search_text(plan.subject).replace(" ", "")
        endpoint_match = (
            _ROUTE_ENDPOINT_PATTERN.search(plan.normalized_question)
            if _ROUTE_ENDPOINT_PATTERN is not None else None
        )
        if endpoint_match and plan.analysis.intent == "list":
            start = normalize_search_text(endpoint_match.group("start")).replace(" ", "")
            end = normalize_search_text(endpoint_match.group("end")).replace(" ", "")
            route_document_ids = {
                source.document_id
                for source in sources
                if start in normalize_search_text(source.snippet).replace(" ", "")
                and end in normalize_search_text(source.snippet).replace(" ", "")
                and bool(re.search(
                    r"(?:线路|路线|号线|由.{0,16}至|起点|终点)",
                    normalize_search_text(f"{source.title}\n{source.snippet}"),
                ))
            }
            if route_document_ids:
                return SearchService._with_station_list_companions(
                    sources,
                    route_document_ids,
                )
        exact_document_ids = {
            source.document_id
            for source in sources
            if (
                title_matches_subject(source.title, normalized_subject)
                        or normalized_subject in SearchService._source_aliases(source)
                        or (
                            plan.analysis.intent == "list"
                            and (
                                title_matches_subject_topic(source.title, normalized_subject)
                                or SearchService._title_contains_core_topic(
                                    source.title,
                                    plan.subject,
                                )
                            )
                        )
            )
        }
        if plan.analysis.intent == "agent" and plan.relations:
            subject_tokens = {
                token for token in lexical_tokens(plan.subject)
                if len(token) >= 2
            }
            relation_documents = {
                source.document_id
                for source in sources
                if subject_tokens
                and all(
                    token in normalize_search_text(f"{source.title}\n{source.snippet}")
                    for token in subject_tokens
                )
                and any(
                    normalize_search_text(relation) in normalize_search_text(source.snippet)
                    for relation in plan.relations
                )
            }
            if relation_documents:
                focused = [
                    source for source in sources
                    if source.document_id in relation_documents
                ]
                return SearchService._sort_relation_sources(
                    focused,
                    subject=plan.subject,
                    relations=plan.relations,
                )
        if plan.analysis.intent == "cause":
            event_terms = tuple(
                normalize_search_text(relation).replace(" ", "")
                for relation in plan.relations
                if len(normalize_search_text(relation).replace(" ", "")) >= 4
                and relation not in {"原因", "因素", "导致", "因由", "缘由"}
            )
            event_document_ids = {
                source.document_id
                for source in sources
                if event_terms
                and any(
                    term in normalize_search_text(
                        f"{source.title}\n{source.snippet}"
                    ).replace(" ", "")
                    for term in event_terms
                )
            }
            if event_document_ids:
                subject_document_ids = {
                    source.document_id
                    for source in sources
                    if title_matches_subject(source.title, normalized_subject)
                }
                return [
                    *(
                        source
                        for source in sources
                        if source.document_id in event_document_ids
                    ),
                    *(
                        source
                        for source in sources
                        if source.document_id in subject_document_ids
                        and source.document_id not in event_document_ids
                    ),
                ]
        if plan.analysis.intent in {"cause", "time", "list"}:
            event_document_ids = {
                source.document_id
                for source in sources
                if title_matches_subject_event(
                    source.title,
                    normalized_subject,
                    plan.normalized_question,
                )
            }
            if event_document_ids:
                if plan.analysis.intent == "list":
                    primary_document_id = SearchService._primary_topic_document_id(
                        sources,
                        event_document_ids,
                        plan.subject,
                    )
                    return SearchService._with_station_list_companions(
                        sources,
                        {primary_document_id},
                    )
                return [source for source in sources if source.document_id in event_document_ids]
        document_ids = exact_document_ids or {
            source.document_id
            for source in sources
            if title_matches_subject(source.title, normalized_subject)
        }
        if not document_ids:
            return sources
        if plan.analysis.intent == "list":
            primary_document_id = SearchService._primary_topic_document_id(
                sources,
                document_ids,
                plan.subject,
            )
            return SearchService._with_station_list_companions(
                sources,
                {primary_document_id},
            )
        focused = [source for source in sources if source.document_id in document_ids]
        if plan.analysis.intent == "agent":
            return SearchService._sort_relation_sources(
                focused,
                subject=plan.subject,
                relations=plan.relations,
            )
        return focused

    @staticmethod
    def _primary_topic_document_id(
        sources: list[SourceItem],
        document_ids: set[str],
        subject: str,
    ) -> str:
        """Choose the most specific topic page among compound-title matches."""

        core_tokens = _core_subject_tokens(subject)
        first_seen: dict[str, int] = {}
        titles: dict[str, str] = {}
        for index, source in enumerate(sources):
            if source.document_id not in document_ids:
                continue
            first_seen.setdefault(source.document_id, index)
            titles.setdefault(
                source.document_id,
                SearchService._result_title(source),
            )

        def priority(document_id: str) -> tuple[int, int, int, int]:
            normalized_title = normalize_search_text(
                titles[document_id]
            ).replace(" ", "")
            title_tokens = set(lexical_tokens(titles[document_id]))
            core_hits = sum(token in title_tokens for token in core_tokens)
            prefix_match = bool(
                core_tokens and normalized_title.startswith(
                    normalize_search_text(core_tokens[0]).replace(" ", "")
                )
            )
            return (
                int(prefix_match),
                core_hits,
                -len(normalized_title),
                -first_seen[document_id],
            )

        return max(document_ids, key=priority)

    @staticmethod
    def _title_contains_core_topic(title: str, subject: str) -> bool:
        core_tokens = _core_subject_tokens(subject)
        if len(core_tokens) < 2:
            return False
        title_tokens = set(lexical_tokens(title))
        return core_tokens[0] in title_tokens

    @staticmethod
    def _result_title(result: object) -> str:
        title = getattr(result, "title", None)
        if title:
            return str(title)
        metadata = getattr(result, "metadata", None)
        if isinstance(metadata, dict):
            return str(metadata.get("title") or "")
        return ""

    @staticmethod
    def _sort_relation_sources(
        sources: list[SourceItem],
        *,
        subject: str,
        relations: tuple[str, ...],
    ) -> list[SourceItem]:
        """Put explicit subject-attribution sentences before loose mentions."""

        normalized_subject = normalize_search_text(subject).replace(" ", "")
        normalized_subject_variants = {
            normalized_subject,
            normalized_subject.replace("事变", "之变").replace("政变", "之变"),
        }
        relation_terms = tuple(
            normalize_search_text(term).replace(" ", "")
            for term in relations
            if term.strip()
        )

        def priority(source: SourceItem) -> tuple[float, float]:
            text = normalize_search_text(source.snippet).replace(" ", "")
            relation_hits = sum(text.count(term) for term in relation_terms)
            explicit_hits = sum(
                bool(re.search(
                    rf"(?:由[^，,。；;（）()\n]{{1,40}}{re.escape(term)}|"
                    rf"{re.escape(term)}(?:是|为|為)[^，,。；;（）()\n]{{1,40}})",
                    text,
                ))
                for term in relation_terms
            )
            if any(
                marker in text
                for marker in ("由", "为首", "為首", "主导", "主導")
            ) and any(term in text for term in relation_terms):
                explicit_hits += 1
            subject_hits = sum(text.count(variant) for variant in normalized_subject_variants if variant)
            return (
                relation_hits * 3.0 + explicit_hits * 8.0 + subject_hits * 0.2,
                -float(source.metadata.get("chunk_order") or 0),
            )

        return sorted(sources, key=priority, reverse=True)

    @staticmethod
    def _source_aliases(source: SourceItem) -> set[str]:
        return document_aliases(source.title, source.metadata)

    @staticmethod
    def _with_station_list_companions(
        sources: list[SourceItem],
        primary_document_ids: set[str],
    ) -> list[SourceItem]:
        primary_sources = [
            source for source in sources if source.document_id in primary_document_ids
        ]
        route_identifiers = {
            identifier
            for source in primary_sources
            for identifier in (
                re.findall(
                    r"\d+号线",
                    normalize_search_text(source.title),
                )
                + list(SearchService._source_aliases(source))
            )
            if identifier
        }
        route_networks = {
            match.group("network")
            for source in primary_sources
            if (
                match := re.search(
                    r"(?P<network>[\u3400-\u9fff]{2,20}?地铁)\d+号线",
                    normalize_search_text(source.title),
                )
            )
        }
        companion_document_ids = {
            source.document_id
            for source in sources
            if normalize_search_text(source.title).replace(" ", "").endswith("车站列表")
            and (
                not route_networks
                or any(
                    normalize_search_text(source.title).replace(" ", "").startswith(network)
                    for network in route_networks
                )
            )
            and any(
                identifier in normalize_search_text(source.snippet).replace(" ", "")
                for identifier in route_identifiers
            )
        }
        companion_source_ids: set[str] = set()
        for document_id in companion_document_ids:
            document_sources = sorted(
                (source for source in sources if source.document_id == document_id),
                key=lambda source: int(source.metadata.get("chunk_order") or 0),
            )
            anchor_order = next(
                (
                    int(source.metadata.get("chunk_order") or 0)
                    for source in document_sources
                    if any(
                        identifier in normalize_search_text(source.snippet).replace(" ", "")
                        for identifier in route_identifiers
                    )
                ),
                None,
            )
            if anchor_order is None:
                continue
            for source in document_sources:
                chunk_order = int(source.metadata.get("chunk_order") or 0)
                normalized_snippet = normalize_search_text(source.snippet).replace(" ", "")
                heading = re.match(r"(?P<line>\d+号线)", normalized_snippet)
                if chunk_order < anchor_order:
                    continue
                if (
                    chunk_order > anchor_order
                    and heading is not None
                    and heading.group("line") not in route_identifiers
                ):
                    break
                companion_source_ids.add(source.id)
        return [
            source for source in sources
            if source.document_id in primary_document_ids
            or source.id in companion_source_ids
        ]

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
            content_type = str(metadata.get("content_type") or "prose")
            snippet_limit = (
                6000
                if content_type in {"table_summary", "table", "list"}
                or int(metadata.get("structure_size") or 1) > 1
                else 900
            )
            snippet = (
                content
                if len(content) <= snippet_limit
                else content[:snippet_limit].rstrip() + "..."
            )
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
