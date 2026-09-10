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
    clean_evidence_text,
)
from .generation import AnswerGenerationError, EvidenceAnswerGenerator, GenerationResult
from .evidence_extraction import (
    EvidenceSpan,
    EvidenceExtractionResult,
    LanguageModelEvidenceExtractor,
)
from .failure_diagnosis import diagnose_failure
from .evidence_gate import (
    evaluate_answer_support,
    evaluate_evidence_gate,
    repair_answer_citations,
)
from .lexical_index import (
    LexicalIndex,
    LexicalResult,
    normalize_search_text,
    query_tokens,
)
from .schemas import AskResponse, SearchRequest, SearchResponse, SourceItem
from .semantic_query_planning import LanguageModelQueryPlanner, QueryPlanningResult
from .qa_analysis import validate_grounding, validate_list_answer
from .query_planning import QueryPlan, TaskField, build_query_plan
from .retrieval_candidates import fuse_document_rrf

_ANSWER_CACHE_TTL_SECONDS = 300.0
_ANSWER_CACHE_MAX_ENTRIES = 128
_ASK_MIN_EVIDENCE_TOP_K = 5
_CITATION_INDEX_PATTERN = re.compile(r"\[资料\s*([1-9]\d*)\]")
_REFUSAL_ANSWERS = {
    "未检索到可用于回答该问题的资料。",
    "根据检索到的资料，无法确定。",
}
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
        bm25_ms = self._elapsed_ms(bm25_started)
        context_started = monotonic()
        results, structure_expanded = await self._expand_structured_results(
            plan.normalized_question,
            results,
            knowledge_base_id=request.knowledge_base_id,
            top_k=top_k,
        )
        relation_context_expanded = False
        # Context expansion can add a more specific topic document after the
        # initial fusion. Reapply the same whole-document ordering before the
        # top-k cut so supplemental chunks cannot put a broad page back first.
        min_score = (
            request.min_score
            if request.min_score is not None
            else self.settings.min_relevance_score
        )
        max_chunks_per_document = self._max_chunks_per_document(
            plan,
            top_k=top_k,
        )
        filtered, selection_trace = self._select_results_with_trace(
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
                "candidate_selection": selection_trace,
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
        fused, _trace = fuse_document_rrf(
            result_groups,
            queries=plan.queries,
            limit=None,
        )
        return fused

    @staticmethod
    def _merge_rank_fusion(
        result_groups: tuple[list[LexicalResult], ...],
        *,
        plan: QueryPlan,
    ) -> list[LexicalResult]:
        fused, _trace = fuse_document_rrf(result_groups, limit=None)
        return fused

    async def _replace_with_lead_chunks(
        self,
        question: str,
        results: list[LexicalResult],
        *,
        knowledge_base_id: str | None,
        intent: str,
    ) -> list[LexicalResult]:
        return results

    async def _expand_document_relation_context(
        self,
        plan: QueryPlan,
        results: list[LexicalResult],
        *,
        knowledge_base_id: str | None,
        top_k: int,
    ) -> tuple[list[LexicalResult], bool]:
        return results, False

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
            limit=min(top_k, self.settings.max_chunks_per_document),
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
            preliminary_sources = list(response.results)
        lexical_gate = evaluate_evidence_gate(
            question,
            analysis,
            preliminary_sources,
            subject=preliminary_plan.subject,
            relations=preliminary_plan.relations,
            field_evidence_available=False,
            field_candidate_count=0,
        )
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
            final_sources = list(current.results)
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
        initial_plan = build_query_plan(request.question)
        evidence_top_k, evidence_policy = self._adaptive_evidence_top_k(
            initial_plan,
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
        verified_structured_answer = None
        if self.settings.semantic_pipeline_enabled:
            verified_structured_answer = self._verified_structured_render(
                question,
                answer_plan,
                answer_sources,
            )
            if verified_structured_answer is not None:
                active_retrieval["verified_structured_render"] = True
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
        raw_model_answer: str | None = None
        retry_model_answer: str | None = None
        verification_model_answer: str | None = None
        verification_queries: list[str] = []
        blocked_answer: str | None = None
        answer_block_reason: str | None = None
        if answer is None:
            if gate.passed:
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
                        fallback = None
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
            else:
                answer = "根据检索到的资料，无法确定。"
                if self.settings.semantic_pipeline_enabled:
                    answer_strategy = "evidence_blocked"
                answer_block_reason = "field_evidence_missing" if (
                    field_extraction and field_extraction.available
                ) else "insufficient_evidence"
        validation = validate_grounding(answer, answer_sources)
        list_validation = validate_list_answer(question, answer, answer_sources)
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
            fallback = None
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
            "ambiguity_candidates": [],
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
            "trace_version": "2",
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
        retrieval_started = monotonic()
        planning = await self._immutable_plan(request.question)
        plan = planning.plan
        evidence_top_k, evidence_policy = self._adaptive_evidence_top_k(
            plan,
            display_top_k=display_top_k,
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
            fused, fusion_trace = fuse_document_rrf(
                result_groups,
                queries=plan.queries,
                limit=evidence_top_k,
            )
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
                    "fusion": fusion_trace,
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
        generation_error_trace: dict[str, object] | None = None
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
                generation_error_trace = getattr(error, "trace", None)
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
            "generation_error_trace": (
                generation_error_trace if generation_error else None
            ),
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
            "trace_version": "2",
            "trace_stages": [
                {"stage": "request", "question": request.question},
                {
                    "stage": "retrieval",
                    "source_ids": [source.id for source in evidence_response.results],
                    "count": len(evidence_response.results),
                    "index": self.settings.opensearch_index,
                    "query_plan": evidence_response.retrieval.get("query_plan", {}),
                    "fusion": evidence_response.retrieval.get("fusion"),
                    "branches": evidence_response.retrieval.get("branches", []),
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
        per_document_limit = max(4, self.settings.max_chunks_per_document)
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
                subject_tokens = set(query_tokens(plan.subject))
                field_core_query = " ".join(
                    token for token in query_tokens(field_question)
                    if token not in subject_tokens
                )
                field_queries = tuple(dict.fromkeys((
                    *plan.queries,
                    plan.subject,
                    *field.relations,
                    field_core_query,
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
                fused, fusion_trace = fuse_document_rrf(
                    groups,
                    queries=field_queries,
                    limit=evidence_top_k,
                )
                field_plan = replace(
                    plan,
                    fields=(field,),
                    relations=field.relations or plan.relations,
                )
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
                    "fusion": fusion_trace,
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
                        "fusion": branch.get("fusion", {}),
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
                    "fusion": branch.get("fusion", {}),
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
    def _focus_complete_list_evidence(
        sources: list[SourceItem],
    ) -> list[SourceItem]:
        return sources

    @staticmethod
    def _verified_structured_render(
        question: str,
        plan: QueryPlan,
        sources: list[SourceItem],
    ) -> str | None:
        return None

    def _adaptive_evidence_top_k(
        self,
        plan: QueryPlan,
        *,
        display_top_k: int,
    ) -> tuple[int, dict[str, object]]:
        target = max(display_top_k, _ASK_MIN_EVIDENCE_TOP_K)
        evidence_top_k = min(self.settings.max_top_k, target)
        return evidence_top_k, {
            "mode": "fixed_minimum",
            "reason": "display_top_k_with_minimum",
            "target": evidence_top_k,
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
            "transport": result.trace,
            "source_ids": [source.id for source in sources],
            "source_span_hashes": [
                sha256(source.snippet.encode("utf-8")).hexdigest()
                for source in sources
            ],
            "output_modified": result.answer != result.raw_output,
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
            return result
        except Exception as error:
            failed = EvidenceExtractionResult(
                (), 0, 0, (f"{type(error).__name__}: {error}",)
            )
            return failed

    @staticmethod
    def _field_extraction_failed(
        extraction: EvidenceExtractionResult | None,
    ) -> bool:
        return bool(
            extraction is not None
            and extraction.errors
            and not extraction.has_candidates
        )

    async def _expand_structured_results(
        self,
        question: str,
        results: list[LexicalResult],
        *,
        knowledge_base_id: str | None,
        top_k: int,
    ) -> tuple[list[LexicalResult], bool]:
        """Restore sibling chunks for a structurally grouped passage."""
        if not results or top_k <= 1:
            return results, False
        lookup = getattr(self.index, "structure_chunks", None)
        if lookup is None:
            return results, False
        anchor = next(
            (
                result for result in results
                if result.metadata.get("parent_id")
                and int(result.metadata.get("structure_size") or 1) > 1
            ),
            None,
        )
        if anchor is None:
            return results, False
        siblings = await asyncio.to_thread(
            lookup,
            str(anchor.metadata["parent_id"]),
            knowledge_base_id=knowledge_base_id,
            limit=min(top_k, self.settings.max_chunks_per_document),
            score=anchor.score,
        )
        if len(siblings) <= 1:
            return results, False
        sibling_ids = {result.node_id for result in siblings}
        return [
            *siblings,
            *(result for result in results if result.node_id not in sibling_ids),
        ], True

    def _select_results_with_trace(
        self,
        results: list[LexicalResult],
        top_k: int,
        min_score: float,
        *,
        max_chunks_per_document: int | None = None,
    ) -> tuple[list[LexicalResult], dict[str, object]]:
        limit = max_chunks_per_document or self.settings.max_chunks_per_document
        selected: list[LexicalResult] = []
        counts: dict[str, int] = {}
        decisions: list[dict[str, object]] = []
        top_score = max((float(item.score or 0) for item in results), default=0.0)
        floor = max(min_score, top_score * self.settings.relative_score_threshold)
        for rank, item in enumerate(results, start=1):
            document_id = item.document_id or item.node_id
            decision = {
                "rank": rank,
                "node_id": item.node_id,
                "document_id": document_id,
                "score": item.score,
                "action": "drop",
            }
            if item.score < floor:
                decision["reason"] = "score_threshold"
            elif counts.get(document_id, 0) >= limit:
                decision["reason"] = "document_limit"
            elif len(selected) >= top_k:
                decision["reason"] = "candidate_limit"
            else:
                counts[document_id] = counts.get(document_id, 0) + 1
                selected.append(item)
                decision.update(action="keep", reason="selected")
            decisions.append(decision)
        return selected, {
            "score_floor": floor,
            "top_k": top_k,
            "document_limit": limit,
            "candidate_count": len(results),
            "selected_count": len(selected),
            "decisions": decisions,
        }

    def _select_results(
        self,
        results: list[LexicalResult],
        top_k: int,
        min_score: float,
        max_chunks_per_document: int | None = None,
    ) -> list[LexicalResult]:
        selected, _ = self._select_results_with_trace(
            results,
            top_k,
            min_score,
            max_chunks_per_document=max_chunks_per_document,
        )
        return selected

    @staticmethod
    def _deterministic_answer(*_args: object, **_kwargs: object) -> str | None:
        return None

    @staticmethod
    def _field_evidence_quote_answer(*_args: object, **_kwargs: object) -> str | None:
        return None

    @staticmethod
    def _evidence_fallback_answer(*_args: object, **_kwargs: object) -> str | None:
        return None

    @staticmethod
    def _extraction_covers_fallback(*_args: object, **_kwargs: object) -> bool:
        return False

    async def _answer_guided_verification(
        self,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[list[SourceItem], EvidenceExtractionResult | None, list[str]]:
        return [], None, []

    def _max_chunks_per_document(
        self,
        plan: QueryPlan | str,
        *,
        top_k: int | None = None,
    ) -> int:
        return self.settings.max_chunks_per_document

    @staticmethod
    def _source_item(result: LexicalResult) -> SourceItem:
        metadata = dict(result.metadata)
        full_answer = str(metadata.pop("full_answer", "")).strip()
        title = str(metadata.pop("title", ""))
        source = str(metadata.pop("source", ""))
        uri = metadata.pop("uri", None)
        content = clean_evidence_text(result.text)
        content_type = str(metadata.get("content_type") or "prose")
        snippet_limit = (
            6000
            if content_type in {"table_summary", "table", "list"}
            or int(metadata.get("structure_size") or 1) > 1
            else 900
        )
        snippet = (
            content
            if content
            else clean_evidence_text(full_answer)
        )
        if len(snippet) > snippet_limit:
            snippet = snippet[:snippet_limit].rstrip() + "..."
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
