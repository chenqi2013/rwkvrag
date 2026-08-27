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


_RELATION_EQUIVALENTS = {
    "老板": ("老板", "创始人", "创办人", "创办者", "负责人", "首席执行官", "CEO"),
    "负责人": ("负责人", "老板", "创始人", "创办人", "创办者", "负责"),
    "首席执行官": ("首席执行官", "CEO", "负责人", "老板"),
}


def _relation_variants(relations: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        variant
        for relation in relations
        for variant in _RELATION_EQUIVALENTS.get(relation, (relation,))
    ))


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

    def remap_sources(
        self,
        sources: list[SourceItem],
    ) -> "EvidenceExtractionResult | None":
        if not self.candidates:
            return None
        current_signatures = tuple(
            (source.id, sha256(source.snippet.encode("utf-8")).hexdigest())
            for source in sources
        )
        current_indexes = {
            signature: source_index
            for source_index, signature in enumerate(current_signatures)
        }
        remapped: list[EvidenceSpan] = []
        for candidate in self.candidates:
            if candidate.source_index >= len(self.source_signatures):
                continue
            source_index = current_indexes.get(
                self.source_signatures[candidate.source_index]
            )
            if source_index is None:
                continue
            remapped.append(EvidenceSpan(
                field_id=candidate.field_id,
                source_index=source_index,
                span=candidate.span,
                content_hash=candidate.content_hash,
            ))
        if not remapped:
            return None
        return EvidenceExtractionResult(
            candidates=tuple(remapped),
            attempted_sources=len(sources),
            completed_sources=min(self.completed_sources, len(sources)),
            errors=self.errors,
            source_signatures=current_signatures,
            strategy=f"{self.strategy}_remapped",
        )

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
        strategy = "model" if candidates else "model_empty"
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
        sentence_units = self._attention_units(plan, source)
        prompt = self._prompt(
            question,
            plan,
            source,
            sentence_units=sentence_units,
        )
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
        return self._parse(
            raw,
            plan,
            source_index,
            source,
            sentence_units=sentence_units,
        )

    def _prompt(
        self,
        question: str,
        plan: QueryPlan,
        source: SourceItem,
        *,
        sentence_units: tuple[str, ...] | None = None,
    ) -> str:
        fields = [
            {
                "field_id": field.field_id,
                "question": field.question,
                "relations": list(field.relations),
            }
            for field in plan.fields
        ]
        units = sentence_units or self._attention_units(plan, source)
        text = "\n".join(
            f"[s{index}] {unit}"
            for index, unit in enumerate(units, start=1)
        )
        contract = {
            "subject": plan.subject,
            "answer_shape": plan.answer_shape,
            "set_semantics": plan.set_semantics,
            "fields": fields,
        }
        field_targets = "；".join(
            f"{field.field_id}：{field.question}"
            for field in plan.fields
        )
        return f"""你是证据抽取器，不回答问题，也不使用常识。只处理当前这一份资料。
任务契约：{json.dumps(contract, ensure_ascii=False)}
对每个字段查找能够直接支持“所求具体值”的最小编号句子。先根据字段问题判断所求值的类型，例如人物、地点、时间、数量、名称或列表；所选句子必须实际给出该类型的具体值。只出现关系词、讨论该关系、表达某人的观点，但没有给出字段所求具体值时，必须拒绝。
候选必须属于任务对象；同名异物、导航、分类、页眉页脚和仅仅提到关键词的背景文字都不要提取。
不要复制或改写正文，只输出句子编号。没有直接证据时 candidates 输出空数组。只输出固定 JSON：
{{"candidates":[{{"field_id":"f1","sentence_id":"s2"}}]}}

问题：{question}
资料标题：{source.title}
编号句子：
{text}
当前唯一任务：为字段“{field_targets}”选择能直接填写具体答案的句子编号。选中的句子如果不能直接回答该字段，就必须输出空数组。
JSON："""

    def _attention_window(self, plan: QueryPlan, source: SourceItem) -> str:
        return "\n".join(self._attention_units(plan, source))

    def _attention_units(
        self,
        plan: QueryPlan,
        source: SourceItem,
    ) -> tuple[str, ...]:
        limit = self.settings.evidence_extraction_max_source_characters
        all_units = [
            value.strip()
            for value in re.split(r"(?<=[。！？!?；;])|\n+", source.snippet)
            if value.strip()
        ]
        if len(source.snippet) <= limit:
            return tuple(all_units)
        terms = {
            term
            for term in lexical_tokens(" ".join((
                plan.subject,
                *(field.question for field in plan.fields),
                *plan.relations,
            )))
            if len(term.strip()) >= 2
        }
        ranked: list[tuple[float, int, str]] = []
        normalized_subject = normalize_search_text(plan.subject).replace(" ", "")
        for index, unit in enumerate(all_units):
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
        return tuple(output)

    @staticmethod
    def _source_contains_subject(plan: QueryPlan, source: SourceItem) -> bool:
        normalized_subject = normalize_search_text(plan.subject).replace(" ", "").replace("的", "")
        normalized_title = normalize_search_text(source.title).replace(" ", "").replace("的", "")
        normalized_relations = {
            normalize_search_text(relation).replace(" ", "").replace("的", "")
            for relation in _relation_variants(plan.relations)
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
        normalized_body = normalize_search_text(source.snippet).replace(" ", "").replace("的", "")
        title_match = any(
            normalized_subject == title
            or (len(title) >= 3 and normalized_subject.endswith(title))
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
        body_subject_match = bool(
            normalized_subject and normalized_subject in normalized_body
        )
        body_relation_match = any(
            relation in normalized_body for relation in normalized_relations
        )
        body_match = body_subject_match and body_relation_match
        transit_match = re.fullmatch(
            r"(?P<network>[\u3400-\u9fff]{2,20}?地铁)(?P<line>\d+号线)",
            normalized_subject,
        )
        transit_companion_match = bool(
            plan.answer_shape == "list"
            and transit_match
            and normalized_title == f"{transit_match.group('network')}车站列表"
            and transit_match.group("line") in normalized_body
        )
        if plan.answer_shape == "single_fact":
            return len(normalized_subject) >= 2 and (
                body_match
                or (title_match and (body_subject_match or body_relation_match))
            )
        return len(normalized_subject) >= 2 and (
            title_match or relation_title_match or body_match or transit_companion_match
        )

    @staticmethod
    def _parse(
        raw: str,
        plan: QueryPlan,
        source_index: int,
        source: SourceItem,
        *,
        sentence_units: tuple[str, ...] | None = None,
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
        if not isinstance(payload, dict) or "candidates" not in payload:
            raise ValueError("extractor response must contain candidates")
        values = payload["candidates"]
        if not isinstance(values, list):
            raise ValueError("extractor candidates must be a list")
        field_ids = {field.field_id for field in plan.fields}
        normalized_title = normalize_search_text(source.title).replace(" ", "")
        contract_text = normalize_search_text(" ".join((
            plan.subject,
            *(field.question for field in plan.fields),
        ))).replace(" ", "")
        normalized_relations = {
            normalize_search_text(relation).replace(" ", "")
            for relation in _relation_variants(plan.relations)
            if len(relation.replace(" ", "")) >= 2
        }
        explicit_relations = {
            normalize_search_text(relation).replace(" ", "")
            for relation in _relation_variants(plan.relations)
            if len(relation.replace(" ", "")) >= 4
        }
        requested_relations = {
            normalize_search_text(relation).replace(" ", "")
            for relation in _relation_variants(plan.relations)
            if len(relation.replace(" ", "")) >= 2
        }
        summary_markers = (
            "结局", "結局", "结尾", "結尾", "终结", "終結", "结束", "結束",
            "最终", "最終", "最后", "最後", "结果", "結果", "归一", "歸一",
            "一统", "一統", "统一", "統一", "灭亡", "滅亡", "完成",
        )
        candidates: list[EvidenceSpan] = []
        for item in values[:32]:
            if not isinstance(item, dict) or "field_id" not in item:
                continue
            field_id = str(item["field_id"]).strip()
            span = ""
            selected_by_id = False
            if set(item) == {"field_id", "sentence_id"} and sentence_units is not None:
                sentence_match = re.fullmatch(r"s([1-9]\d*)", str(item["sentence_id"]).strip())
                if sentence_match:
                    sentence_index = int(sentence_match.group(1)) - 1
                    if sentence_index < len(sentence_units):
                        span = sentence_units[sentence_index]
                        selected_by_id = True
            elif set(item) == {"field_id", "span"}:
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
            if (
                plan.analysis.intent == "agent"
                and len(normalized_span) <= 16
                and not any(marker in normalized_span for marker in (
                    "是", "为", "由", "被", "害死", "杀害", "打死", "处死", "发起",
                ))
                and normalize_search_text(plan.subject).replace(" ", "")
                not in normalized_span
            ):
                continue
            if (
                selected_by_id
                and plan.answer_shape == "single_fact"
                and normalized_relations
                and not any(
                    relation in normalized_span
                    for relation in normalized_relations
                )
            ):
                continue
            if not selected_by_id and explicit_relations and not any(
                relation in normalized_span
                for relation in explicit_relations
            ):
                continue
            if (
                plan.answer_shape in {"summary", "narrative"}
                and not any(
                    relation in normalized_span
                    for relation in requested_relations
                )
                and not any(marker in normalized_span for marker in summary_markers)
            ):
                continue
            candidates.append(EvidenceSpan(
                field_id=field_id,
                source_index=source_index,
                span=span,
                content_hash=sha256(span.encode("utf-8")).hexdigest(),
            ))
        if not candidates and plan.answer_shape == "list" and sentence_units:
            subject_terms = {
                normalize_search_text(value).replace(" ", "")
                for value in (plan.subject, source.title)
                if value.strip()
            }
            relation_terms = {
                relation
                for relation in normalized_relations
                if len(relation) >= 2
            }
            for unit in sentence_units:
                normalized_unit = normalize_search_text(unit).replace(" ", "")
                if not any(term in normalized_unit for term in subject_terms):
                    continue
                if not any(relation in normalized_unit for relation in relation_terms):
                    continue
                candidates.append(EvidenceSpan(
                    field_id=next(iter(field_ids)),
                    source_index=source_index,
                    span=unit,
                    content_hash=sha256(unit.encode("utf-8")).hexdigest(),
                ))
                break
        return tuple(candidates)
