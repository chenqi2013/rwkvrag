import asyncio
from dataclasses import dataclass
from hashlib import sha256
import json
import re

import httpx

from .config import Settings
from .generation import EvidenceAnswerGenerator
from .lexical_index import lexical_tokens, normalize_search_text
from .query_planning import QueryPlan
from .schemas import SourceItem


@dataclass(frozen=True)
class EvidenceSpan:
    field_id: str
    source_index: int
    span: str
    content_hash: str


@dataclass(frozen=True)
class EvidenceExtractionResult:
    candidates: tuple[EvidenceSpan, ...]
    attempted_sources: int
    completed_sources: int
    errors: tuple[str, ...] = ()
    source_signatures: tuple[tuple[str, str], ...] = ()
    strategy: str = "model"

    @property
    def available(self) -> bool:
        return self.completed_sources > 0

    @property
    def has_candidates(self) -> bool:
        return bool(self.candidates)

    def matches_sources(self, sources: list[SourceItem]) -> bool:
        selected = sources[: self.attempted_sources]
        signatures = tuple(
            (source.id, sha256(source.snippet.encode("utf-8")).hexdigest())
            for source in selected
        )
        return signatures == self.source_signatures

    def answer_sources(self, sources: list[SourceItem]) -> list[SourceItem]:
        grouped: dict[int, list[EvidenceSpan]] = {}
        for candidate in self.candidates:
            grouped.setdefault(candidate.source_index, []).append(candidate)
        output: list[SourceItem] = []
        for source_index in grouped:
            if source_index >= len(sources):
                continue
            source = sources[source_index]
            candidates = grouped[source_index]
            spans = list(dict.fromkeys(candidate.span for candidate in candidates))
            hashes = [candidate.content_hash for candidate in candidates]
            metadata = {
                **source.metadata,
                "evidence_span_hashes": hashes,
                "evidence_field_ids": list(dict.fromkeys(
                    candidate.field_id for candidate in candidates
                )),
            }
            output.append(source.model_copy(update={
                "snippet": "\n".join(spans),
                "metadata": metadata,
            }))
        return output


class LanguageModelEvidenceExtractor:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport

    async def extract(
        self,
        question: str,
        plan: QueryPlan,
        sources: list[SourceItem],
    ) -> EvidenceExtractionResult:
        if not self.settings.evidence_extraction_enabled or not sources:
            return EvidenceExtractionResult((), 0, 0)
        if not self.settings.generation_password:
            return EvidenceExtractionResult(
                (), 0, 0, ("generation_password_not_configured",)
            )
        source_limit = (
            self.settings.evidence_extraction_max_sources
            if plan.answer_shape in {"list", "summary", "narrative"}
            else min(3, self.settings.evidence_extraction_max_sources)
        )
        selected = sources[:source_limit]
        signatures = tuple(
            (source.id, sha256(source.snippet.encode("utf-8")).hexdigest())
            for source in selected
        )
        semaphore = asyncio.Semaphore(self.settings.evidence_extraction_concurrency)

        async def run(source_index: int, source: SourceItem):
            async with semaphore:
                return await self._extract_source(question, plan, source_index, source)

        results = await asyncio.gather(*(
            run(source_index, source)
            for source_index, source in enumerate(selected)
        ), return_exceptions=True)
        candidates: list[EvidenceSpan] = []
        errors: list[str] = []
        completed = 0
        seen: set[tuple[str, int, str]] = set()
        for source_index, result in enumerate(results):
            if isinstance(result, BaseException):
                errors.append(f"source_{source_index + 1}: {type(result).__name__}: {result}")
                continue
            completed += 1
            for candidate in result:
                key = (candidate.field_id, candidate.source_index, candidate.span)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
        strategy = "model"
        if completed:
            had_model_candidates = bool(candidates)
            lexical_candidates = self._lexical_fallback(plan, selected)
            for candidate in lexical_candidates:
                key = (candidate.field_id, candidate.source_index, candidate.span)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
            if lexical_candidates:
                strategy = "model+lexical" if had_model_candidates else "lexical_fallback"
            elif not candidates:
                strategy = "model_empty"
        return EvidenceExtractionResult(
            tuple(candidates),
            attempted_sources=len(selected),
            completed_sources=completed,
            errors=tuple(errors),
            source_signatures=signatures,
            strategy=strategy,
        )

    async def _extract_source(
        self,
        question: str,
        plan: QueryPlan,
        source_index: int,
        source: SourceItem,
    ) -> tuple[EvidenceSpan, ...]:
        prompt = self._prompt(question, plan, source)
        payload = {
            "contents": [prompt],
            "max_tokens": self.settings.evidence_extraction_max_tokens,
            "temperature": 0.1,
            "top_k": 20,
            "top_p": 0.4,
            "alpha_presence": 0.0,
            "alpha_frequency": 0.0,
            "alpha_decay": 0.99,
            "stream": True,
            "password": self.settings.generation_password,
        }
        endpoint = f"{self.settings.generation_base_url.rstrip('/')}/chat/completions"
        async with httpx.AsyncClient(
            timeout=self.settings.evidence_extraction_timeout,
            transport=self.transport,
        ) as client:
            async with client.stream("POST", endpoint, json=payload) as response:
                response.raise_for_status()
                raw = await EvidenceAnswerGenerator._read_stream(
                    response,
                    total_timeout=self.settings.evidence_extraction_timeout,
                )
        return self._parse(raw, plan, source_index, source)

    def _prompt(self, question: str, plan: QueryPlan, source: SourceItem) -> str:
        fields = [
            {
                "field_id": field.field_id,
                "question": field.question,
                "relations": list(field.relations),
            }
            for field in plan.fields
        ]
        text = self._attention_window(plan, source)
        contract = {
            "subject": plan.subject,
            "answer_shape": plan.answer_shape,
            "set_semantics": plan.set_semantics,
            "fields": fields,
        }
        return f"""你是证据抽取器，不回答问题，也不使用常识。只处理当前这一份资料。
任务契约：{json.dumps(contract, ensure_ascii=False)}
对每个字段查找能够直接支持“所求具体值”的最小事实单位。span 必须同时包含具体值及其关系，且逐字复制资料正文，不能只复制标题、字段名或问题中的文字，不能改写、概括或拼接不连续文本。
候选必须属于任务对象；同名异物、导航、分类、页眉页脚和仅仅提到关键词的背景文字都不要提取。
没有直接证据时 candidates 输出空数组。只输出固定 JSON：
{{"candidates":[{{"field_id":"f1","span":"资料中的逐字原文"}}]}}

问题：{question}
资料标题：{source.title}
资料正文：
{text}
JSON："""

    def _attention_window(self, plan: QueryPlan, source: SourceItem) -> str:
        limit = self.settings.evidence_extraction_max_source_characters
        if len(source.snippet) <= limit:
            return source.snippet
        terms = {
            term
            for term in lexical_tokens(" ".join((
                plan.subject,
                *(field.question for field in plan.fields),
                *plan.relations,
            )))
            if len(term.strip()) >= 2
        }
        units = [
            value.strip()
            for value in re.split(r"(?<=[。！？!?；;])|\n+", source.snippet)
            if value.strip()
        ]
        ranked: list[tuple[float, int, str]] = []
        normalized_subject = normalize_search_text(plan.subject).replace(" ", "")
        for index, unit in enumerate(units):
            normalized = normalize_search_text(unit).replace(" ", "")
            unit_terms = set(lexical_tokens(unit))
            score = float(len(terms & unit_terms))
            if normalized_subject and normalized_subject in normalized:
                score += 3.0
            score += sum(
                1.5
                for relation in plan.relations
                if normalize_search_text(relation).replace(" ", "") in normalized
            )
            ranked.append((score, index, unit))
        selected = sorted(
            sorted(ranked, key=lambda item: (-item[0], item[1]))[:8],
            key=lambda item: item[1],
        )
        output: list[str] = []
        used = 0
        for _, _, unit in selected:
            remaining = limit - used
            if remaining <= 0:
                break
            value = unit[:remaining]
            output.append(value)
            used += len(value) + 1
        return "\n".join(output)

    def _lexical_fallback(
        self,
        plan: QueryPlan,
        sources: list[SourceItem],
    ) -> tuple[EvidenceSpan, ...]:
        field_id = plan.fields[0].field_id if plan.fields else "f1"
        candidates: list[EvidenceSpan] = []
        ranked_sources = sorted(
            enumerate(sources),
            key=lambda item: (
                -self._source_relevance(plan, item[1]),
                item[0],
            ),
        )
        for source_index, source in ranked_sources:
            if not self._source_contains_subject(plan, source):
                continue
            window = self._attention_window(plan, source)
            units = [
                value.strip()
                for value in re.split(r"(?<=[。！？!?；;])|\n+", window)
                if value.strip()
            ]
            source_candidate_count = 0
            for unit in units[:8]:
                normalized = normalize_search_text(unit).replace(" ", "")
                normalized_title = normalize_search_text(source.title).replace(" ", "")
                if (
                    not normalized
                    or normalized == normalized_title
                    or normalized.startswith("category:")
                    or normalized.startswith("thumb|")
                ):
                    continue
                if plan.relations:
                    normalized_relations = {
                        normalize_search_text(relation).replace(" ", "")
                        for relation in plan.relations
                        if len(relation.strip()) >= 2
                    }
                    if normalized_relations and not any(
                        relation in normalized
                        for relation in normalized_relations
                    ):
                        continue
                span = unit[:1_000]
                candidates.append(EvidenceSpan(
                    field_id=field_id,
                    source_index=source_index,
                    span=span,
                    content_hash=sha256(span.encode("utf-8")).hexdigest(),
                ))
                source_candidate_count += 1
                if len(candidates) >= 18:
                    return tuple(candidates)
                if source_candidate_count >= 6:
                    break
        return tuple(candidates)

    @staticmethod
    def _source_relevance(plan: QueryPlan, source: SourceItem) -> float:
        title = normalize_search_text(source.title).replace(" ", "").replace("的", "")
        snippet = normalize_search_text(source.snippet).replace(" ", "").replace("的", "")
        subject = normalize_search_text(plan.subject).replace(" ", "").replace("的", "")
        score = 3.0 if subject and subject in title else 0.0
        for relation in plan.relations:
            normalized = normalize_search_text(relation).replace(" ", "").replace("的", "")
            if not normalized:
                continue
            if normalized in title:
                score += 4.0
            elif normalized in snippet:
                score += 0.5
        return score

    @staticmethod
    def _source_contains_subject(plan: QueryPlan, source: SourceItem) -> bool:
        normalized_subject = normalize_search_text(plan.subject).replace(" ", "").replace("的", "")
        normalized_title = normalize_search_text(source.title).replace(" ", "").replace("的", "")
        normalized_relations = {
            normalize_search_text(relation).replace(" ", "").replace("的", "")
            for relation in plan.relations
            if len(relation.strip()) >= 2
        }
        aliases = source.metadata.get("aliases")
        title_values = {normalized_title}
        if isinstance(aliases, list):
            title_values.update(
                normalize_search_text(str(alias)).replace(" ", "").replace("的", "")
                for alias in aliases
                if str(alias).strip()
            )
        evidence = normalize_search_text(f"{source.title}\n{source.snippet}").replace(" ", "").replace("的", "")
        title_match = any(
            normalized_subject == title
            or title.endswith(f"·{normalized_subject}")
            or any(
                title == relation
                or title.startswith(f"{normalized_subject}{relation}")
                for relation in normalized_relations
            )
            for title in title_values
        )
        relation_title_match = any(
            title == relation or title.endswith(f"·{relation}")
            for title in title_values
            for relation in normalized_relations
        )
        body_match = bool(
            normalized_subject
            and normalized_subject in evidence
            and any(relation in evidence for relation in normalized_relations)
        )
        return len(normalized_subject) >= 2 and (
            title_match or relation_title_match or body_match
        )

    @staticmethod
    def _parse(
        raw: str,
        plan: QueryPlan,
        source_index: int,
        source: SourceItem,
    ) -> tuple[EvidenceSpan, ...]:
        if not LanguageModelEvidenceExtractor._source_contains_subject(plan, source):
            return ()
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "")
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("extractor response does not contain a JSON object")
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as error:
            raise ValueError("extractor response contains invalid JSON") from error
        if not isinstance(payload, dict) or set(payload) != {"candidates"}:
            raise ValueError("extractor response must contain only candidates")
        values = payload["candidates"]
        if not isinstance(values, list):
            raise ValueError("extractor candidates must be a list")
        field_ids = {field.field_id for field in plan.fields}
        normalized_title = normalize_search_text(source.title).replace(" ", "")
        contract_text = normalize_search_text(" ".join((
            plan.subject,
            *(field.question for field in plan.fields),
        ))).replace(" ", "")
        explicit_relations = {
            normalize_search_text(relation).replace(" ", "")
            for relation in plan.relations
            if len(relation.replace(" ", "")) >= 4
        }
        candidates: list[EvidenceSpan] = []
        for item in values[:32]:
            if not isinstance(item, dict) or set(item) != {"field_id", "span"}:
                continue
            field_id = str(item["field_id"]).strip()
            span = str(item["span"]).strip()
            if field_id not in field_ids or not span:
                continue
            if len(span) > 2_000 or span not in source.snippet:
                continue
            normalized_span = normalize_search_text(span).replace(" ", "")
            if normalized_span == normalized_title or (
                len(normalized_span) >= 4 and normalized_span in contract_text
            ):
                continue
            if explicit_relations and not any(
                relation in normalized_span
                for relation in explicit_relations
            ):
                continue
            candidates.append(EvidenceSpan(
                field_id=field_id,
                source_index=source_index,
                span=span,
                content_hash=sha256(span.encode("utf-8")).hexdigest(),
            ))
        return tuple(candidates)
