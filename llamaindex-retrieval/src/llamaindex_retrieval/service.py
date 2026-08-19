import asyncio
import re
from collections import OrderedDict
from time import monotonic

from .config import Settings
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
    time_evidence_answer,
)
from .generation import EvidenceAnswerGenerator
from .evidence_gate import (
    evaluate_answer_support,
    evaluate_evidence_gate,
    repair_answer_citations,
    title_matches_subject,
    title_matches_subject_event,
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
from .qa_analysis import (
    ambiguity_candidates,
    analyze_question,
    remove_unsupported_number_sentences,
    validate_grounding,
    validate_list_answer,
)
from .query_planning import QueryPlan, build_query_plan

_MULTI_EVIDENCE_MARKERS = (
    "哪些",
    "有哪些",
    "列表",
    "全部",
    "所有",
    "分别",
    "列一下",
    "列出",
    "列举",
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
_LIST_TOPIC_HINTS = ("车站", "站点", "站名")
_EXPLANATORY_SECTION_HINTS = ("问题", "問題", "歷史", "历史", "命名", "更名", "工程", "續建", "续建")
_TRANSFER_HINTS = ("转乘", "轉乘", "换乘", "換乘", "线网转乘", "線網轉乘")
_STATION_STRUCTURE_NOISE = ("列车", "列車", "车型", "車型", "车队", "車隊", "制造商", "製造商")
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
_LEAD_FACT_MARKERS = (
    "是什么",
    "是谁",
    "指的是什么",
    "简要介绍",
    "位于哪里",
    "出生于哪里",
    "哪一年成立",
    "哪一年创建",
    "逝世于哪一年",
    "有什么区别",
    "有何区别",
    "相比如何",
    "比较",
)
_CHRONOLOGICAL_LIST_MARKERS = ("从古至今", "自古至今", "历代", "歷代", "迄今为止")
_ROUTE_ENDPOINT_PATTERN = re.compile(
    r"连接(?P<start>[^，。？?和与]{1,20})(?:和|与)"
    r"(?P<end>[^，。？?的]{1,20})的[^，。？?]{2,32}?(?:线路|路线)"
)


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
        plan = build_query_plan(request.question)
        analysis = plan.analysis
        results = await self._execute_query_plan(
            plan,
            candidate_k=candidate_k,
            knowledge_base_id=request.knowledge_base_id,
        )
        results, relation_context_expanded = await self._expand_document_relation_context(
            plan,
            results,
            knowledge_base_id=request.knowledge_base_id,
            top_k=top_k,
        )
        results, structure_expanded = await self._expand_structured_results(
            plan.normalized_question,
            results,
            knowledge_base_id=request.knowledge_base_id,
            top_k=top_k,
        )
        if analysis.intent == "list" and plan.relations:
            results, list_relation_expanded = await self._expand_document_relation_context(
                plan,
                results,
                knowledge_base_id=request.knowledge_base_id,
                top_k=top_k,
            )
            relation_context_expanded = relation_context_expanded or list_relation_expanded
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
                    "merge_strategy": plan.merge_strategy,
                    "fusion": "weighted_rrf",
                    "context_policy": plan.context_policy,
                },
            },
        )

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
                if document_id not in first_seen:
                    first_seen[document_id] = sequence
                    sequence += 1
        normalized_subject = normalize_search_text(plan.subject).replace(" ", "")
        endpoint_match = _ROUTE_ENDPOINT_PATTERN.search(plan.normalized_question)
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
                aliases = {
                    normalize_search_text(str(alias)).replace(" ", "")
                    for result in document_chunks.values()
                    for alias in (
                        result.metadata.get("aliases")
                        if isinstance(result.metadata.get("aliases"), list)
                        else []
                    )
                    if str(alias).strip()
                }
                if normalized_title == normalized_subject or normalized_title.startswith(
                    (f"{normalized_subject}(", f"{normalized_subject}（")
                ):
                    document_scores[document_id] += 0.12
                elif normalized_subject in aliases:
                    document_scores[document_id] += 0.1
                elif (
                    plan.analysis.intent in {"cause", "time", "list"}
                    and title_matches_subject_event(
                        title,
                        normalized_subject,
                        plan.normalized_question,
                    )
                ):
                    document_scores[document_id] += 0.24
                if plan.relations and any(
                    normalized_subject in normalize_search_text(result.text).replace(" ", "")
                    and any(
                        normalize_search_text(relation).replace(" ", "")
                        in normalize_search_text(result.text).replace(" ", "")
                        for relation in plan.relations
                    )
                    for result in document_chunks.values()
                ):
                    document_scores[document_id] += 0.06
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
        for document_id in ranked_documents:
            ranked_chunks = sorted(
                chunks[document_id].values(),
                key=lambda result: chunk_scores[result.node_id],
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
            "definition", "comparison", "time", "agent", "location", "birthplace", "list",
        } or not results:
            return results
        lookup = getattr(self.index, "document_lead_chunk", None)
        if lookup is None:
            return results
        if intent in {"time", "agent", "location", "birthplace", "list"}:
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
        }:
            return results, False
        lookup = getattr(self.index, "document_relation_candidates", None)
        if lookup is None:
            return results, False
        normalized_subject = normalize_search_text(plan.subject).replace(" ", "")
        document_ids: list[str] = []
        for result in results:
            title = str(result.metadata.get("title") or "")
            if (
                result.document_id not in document_ids
                and (
                    title_matches_subject(title, normalized_subject)
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
        if not document_ids:
            return results, False
        candidate_groups = await asyncio.gather(*(
            asyncio.to_thread(
                lookup,
                plan.normalized_question,
                document_id=document_id,
                relations=plan.relations,
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

    async def ask(self, request: SearchRequest) -> AskResponse:
        display_top_k = min(request.top_k or self.settings.default_top_k, self.settings.max_top_k)
        request_plan = build_query_plan(request.question)
        evidence_top_k = max(display_top_k, _ASK_MIN_EVIDENCE_TOP_K)
        if request_plan.analysis.intent == "time" and len(request_plan.analysis.subjects) > 2:
            evidence_top_k = max(evidence_top_k, 10)
        evidence_request = request.model_copy(
            update={"top_k": evidence_top_k}
        )
        evidence_response = await self.search(evidence_request)
        retrieval = {
            **evidence_response.retrieval,
            "top_k": display_top_k,
            "returned": min(len(evidence_response.results), display_top_k),
            "answer_evidence_top_k": evidence_response.retrieval.get("top_k"),
            "answer_evidence_count": len(evidence_response.results),
        }
        question = str(retrieval.get("normalized_question") or request.question)
        question_analysis = analyze_question(question)
        answer_plan = build_query_plan(question)
        if question_analysis.intent == "cause":
            evidence_response = evidence_response.model_copy(
                update={
                    "results": self._trim_post_event_evidence(
                        evidence_response.results,
                        subject=answer_plan.subject,
                        relations=answer_plan.relations,
                    )
                }
            )
        evidence_response = evidence_response.model_copy(
            update={
                "results": self._focus_sources_on_subject(
                    answer_plan,
                    evidence_response.results,
                )
            }
        )
        gate = evaluate_evidence_gate(
            question,
            question_analysis,
            evidence_response.results,
            subject=answer_plan.subject,
        )
        assessment = gate.assessment
        cache_key = self._answer_cache_key(question, evidence_response)
        answer = self._get_cached_answer(cache_key)
        cache_hit = answer is not None
        answer_strategy = "cache" if cache_hit else "model"
        ambiguity = ambiguity_candidates(question, evidence_response.results)
        if answer is None:
            if ambiguity and gate.passed:
                labels = "、".join(
                    f"{title}[资料 {next(index for index, source in enumerate(evidence_response.results, start=1) if source.title == title)}]"
                    for title in ambiguity
                )
                answer = f"这个名称可能指：{labels}。请补充你想查询的具体对象。"
                answer_strategy = "clarification"
            elif gate.passed and question_analysis.intent == "definition":
                answer = definition_evidence_answer(question, evidence_response.results)
            elif (
                gate.passed
                and question_analysis.intent == "time"
            ):
                answer = coordinated_time_evidence_answer(
                    evidence_response.results,
                    question_analysis.subjects,
                ) or time_evidence_answer(question, evidence_response.results)
            elif gate.passed and question_analysis.intent == "agent":
                answer = agent_evidence_answer(question, evidence_response.results)
            elif gate.passed and question_analysis.intent == "ordinal":
                answer = ordinal_evidence_answer(question, evidence_response.results)
            elif gate.passed and question_analysis.intent == "location":
                answer = location_evidence_answer(question, evidence_response.results)
            elif gate.passed and question_analysis.intent == "birthplace":
                answer = birthplace_evidence_answer(question, evidence_response.results)
            if (
                answer is None
                and gate.passed
                and question_analysis.intent in {"fact", "list"}
            ):
                answer = direct_evidence_answer(question, evidence_response.results)
            if answer is not None:
                answer_strategy = "direct_extract"
            elif gate.passed:
                answer = await self.generator.generate(question, evidence_response.results)
                if (
                    any(
                        marker in answer
                        for marker in ("无法确定", "无法从资料", "资料不足", "不能确定")
                    )
                    and question_analysis.intent == "cause"
                ):
                    fallback = cause_evidence_answer(evidence_response.results)
                    if fallback is not None:
                        answer = fallback
                        answer_strategy = "evidence_fallback"
            else:
                answer = "根据检索到的资料，无法确定。"
        if question_analysis.intent == "cause" and self._is_empty_answer_shell(answer):
            fallback = cause_evidence_answer(evidence_response.results)
            if fallback is not None:
                answer = fallback
                answer_strategy = "evidence_fallback"
        validation = validate_grounding(answer, evidence_response.results)
        if "unsupported_number" in validation.issues and answer_strategy in {"model", "cache"}:
            cleaned_answer = remove_unsupported_number_sentences(answer, validation.unsupported_numbers)
            answer = cleaned_answer or "根据检索到的资料，无法确定。"
            validation = validate_grounding(answer, evidence_response.results)
        list_validation = validate_list_answer(question, answer, evidence_response.results)
        if list_validation.complete is False:
            fallback = list_evidence_answer(question, evidence_response.results)
            answer = fallback or "根据检索到的资料，无法确定完整列表。[资料 1]"
            answer_strategy = "evidence_fallback" if fallback else "incomplete_list_blocked"
            validation = validate_grounding(answer, evidence_response.results)
        if answer_strategy == "model":
            answer = repair_answer_citations(answer, evidence_response.results)
        answer_support = evaluate_answer_support(answer, evidence_response.results)
        if not answer_support.passed and answer_strategy in {"model", "cache"}:
            fallback = (
                cause_evidence_answer(evidence_response.results)
                if question_analysis.intent == "cause"
                else None
            )
            answer = fallback or "根据检索到的资料，无法确定。"
            answer_strategy = "evidence_fallback" if fallback else "answer_grounding_blocked"
            validation = validate_grounding(answer, evidence_response.results)
            answer_support = evaluate_answer_support(answer, evidence_response.results)
        if (
            gate.passed
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
            evidence_response.results,
            answer,
            display_top_k=display_top_k,
        )
        retrieval["returned"] = len(response_results)
        model_name = await self.generator.current_model()
        return AskResponse(
            answer=answer,
            sources=response_results,
            retrieval=retrieval,
            generation={
                "model": model_name,
                "endpoint": self.settings.generation_base_url,
                "evidence_count": len(evidence_response.results),
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
                "citation_required": True,
                "cache_hit": cache_hit,
                "answer_strategy": answer_strategy,
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
            },
        )

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

    @staticmethod
    def _is_empty_answer_shell(answer: str) -> bool:
        return bool(_EMPTY_ANSWER_SHELL.search(answer.strip()))

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
                results[0],
                *flattened_station_context,
                *(
                    result
                    for result in results[1:]
                    if result.node_id not in context_ids
                    and result.document_id not in context_document_ids
                ),
            ], True
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
        anchor_pool = list({result.node_id: result for result in (*same_document, *candidates)}.values())
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
            marker in question for marker in _MULTI_EVIDENCE_MARKERS
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
        endpoint_match = _ROUTE_ENDPOINT_PATTERN.search(plan.normalized_question)
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
                normalize_search_text(source.title).replace(" ", "") == normalized_subject
                or normalized_subject in SearchService._source_aliases(source)
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
                return [
                    source for source in sources
                    if source.document_id in relation_documents
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
                    return SearchService._with_station_list_companions(
                        sources,
                        event_document_ids,
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
            return SearchService._with_station_list_companions(sources, document_ids)
        return [source for source in sources if source.document_id in document_ids]

    @staticmethod
    def _source_aliases(source: SourceItem) -> set[str]:
        aliases = source.metadata.get("aliases")
        if not isinstance(aliases, list):
            return set()
        return {
            normalize_search_text(str(alias)).replace(" ", "")
            for alias in aliases
            if str(alias).strip()
        }

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
            snippet_limit = 6000 if content_type in {"table_summary", "table", "list"} else 900
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
