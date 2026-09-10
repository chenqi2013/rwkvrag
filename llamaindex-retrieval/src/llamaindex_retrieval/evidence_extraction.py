import asyncio
from dataclasses import dataclass
from hashlib import sha256
import json
import re

import httpx

from .config import Settings
from .generation import EvidenceAnswerGenerator
from .query_planning import QueryPlan
from .schemas import SourceItem


@dataclass(frozen=True)
class EvidenceSpan:
    field_id: str
    source_index: int
    span: str
    content_hash: str


@dataclass(frozen=True)
class _EvidenceUnit:
    source_index: int
    span: str
    content_hash: str
    score: float


@dataclass(frozen=True)
class EvidenceExtractionResult:
    candidates: tuple[EvidenceSpan, ...]
    attempted_sources: int
    completed_sources: int
    errors: tuple[str, ...] = ()
    source_signatures: tuple[tuple[str, str], ...] = ()
    strategy: str = "model"
    trace_events: tuple[dict[str, object], ...] = ()

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
            trace_events=self.trace_events,
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
        source_limit = self.settings.evidence_extraction_max_sources
        selected: list[SourceItem] = []
        for source in sources:
            if len(selected) >= source_limit:
                break
            selected.append(source)
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
        trace_events: list[dict[str, object]] = []
        errors: list[str] = []
        completed = 0
        seen: set[tuple[str, int, str]] = set()
        for source_index, result in enumerate(results):
            if isinstance(result, BaseException):
                errors.append(f"source_{source_index + 1}: {type(result).__name__}: {result}")
                continue
            completed += 1
            source_candidates, source_trace = result
            trace_events.append(source_trace)
            for candidate in source_candidates:
                key = (candidate.field_id, candidate.source_index, candidate.span)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
        strategy = "model" if candidates else "model_empty"
        if self.settings.semantic_pipeline_enabled and completed:
            try:
                adjudicated, adjudication_trace = await self._adjudicate(
                    question,
                    plan,
                    selected,
                    candidates,
                )
                candidates = list(adjudicated)
                trace_events.append(adjudication_trace)
                strategy = "model_map_reduce" if candidates else "model_map_reduce_empty"
            except Exception as error:
                errors.append(f"adjudication: {type(error).__name__}: {error}")
                strategy = "model_map_reduce_error_fallback" if candidates else "model_map_reduce_error"
        return EvidenceExtractionResult(
            tuple(candidates),
            attempted_sources=len(selected),
            completed_sources=completed,
            errors=tuple(errors),
            source_signatures=signatures,
            strategy=strategy,
            trace_events=tuple(trace_events),
        )

    async def _adjudicate(
        self,
        question: str,
        plan: QueryPlan,
        sources: list[SourceItem],
        map_candidates: list[EvidenceSpan],
    ) -> tuple[tuple[EvidenceSpan, ...], dict[str, object]]:
        units = self._adjudication_units(plan, sources, map_candidates)
        if not units:
            return (), {
                "stage": "resolver_adjudication",
                "prompt": "",
                "raw_output": "",
                "selected_count": 0,
            }
        prompt = self._adjudication_prompt(question, plan, sources, units)
        payload = {
            "contents": [prompt],
            "max_tokens": min(self.settings.evidence_extraction_max_tokens, 192),
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
        selections = self._parse_adjudication(
            raw,
            {field.field_id for field in plan.fields},
            len(units),
        )
        selections = selections[:16]
        output: list[EvidenceSpan] = []
        seen: set[tuple[str, int, str]] = set()
        for field_id, unit_index in selections:
            unit = units[unit_index - 1]
            if sha256(unit.span.encode("utf-8")).hexdigest() != unit.content_hash:
                continue
            if unit.source_index >= len(sources):
                continue
            if unit.span not in sources[unit.source_index].snippet:
                continue
            key = (field_id, unit.source_index, unit.span)
            if key in seen:
                continue
            seen.add(key)
            output.append(EvidenceSpan(
                field_id=field_id,
                source_index=unit.source_index,
                span=unit.span,
                content_hash=unit.content_hash,
            ))
        return tuple(output), {
            "stage": "resolver_adjudication",
            "prompt": prompt,
            "raw_output": raw,
            "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
            "raw_output_sha256": sha256(raw.encode("utf-8")).hexdigest(),
            "selected_count": len(output),
        }

    def _adjudication_units(
        self,
        plan: QueryPlan,
        sources: list[SourceItem],
        map_candidates: list[EvidenceSpan],
    ) -> tuple[_EvidenceUnit, ...]:
        """Build a bounded, deterministic view for the resolver model.

        Ordering and limits are transport concerns. No question vocabulary,
        intent marker, title heuristic, or answer-specific bonus is applied.
        """
        raw_units: list[tuple[int, str, bool]] = []
        seen: set[tuple[int, str]] = set()
        for candidate in map_candidates:
            key = (candidate.source_index, candidate.span)
            if key in seen or candidate.source_index >= len(sources):
                continue
            seen.add(key)
            raw_units.append((candidate.source_index, candidate.span, True))
        for source_index, source in enumerate(sources):
            for span in self._attention_units(plan, source):
                key = (source_index, span)
                if key in seen:
                    continue
                seen.add(key)
                raw_units.append((source_index, span, False))
        if not raw_units:
            return ()
        units = [
            _EvidenceUnit(
                source_index=source_index,
                span=span,
                content_hash=sha256(span.encode("utf-8")).hexdigest(),
                score=1.0 if selected_by_map else 0.0,
            )
            for source_index, span, selected_by_map in raw_units
        ]
        return tuple(units[:16])

    @staticmethod
    def _adjudication_prompt(
        question: str,
        plan: QueryPlan,
        sources: list[SourceItem],
        units: tuple[_EvidenceUnit, ...],
    ) -> str:
        contract = {
            "subject": plan.subject,
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
        }
        evidence = "\n".join(
            f"[e{index}] 标题：{sources[unit.source_index].title or '未命名'}｜{unit.span}"
            for index, unit in enumerate(units, start=1)
        )
        return f"""你是证据裁决器，不回答问题。候选均来自知识库原文，代码会按编号取回原文。
严格按任务契约逐字段选择能够直接填写所求具体值的证据。模型负责判断对象身份、关系含义、答案值类型和事件结果是否与字段完全一致。
仅仅提到对象、关系词相同、属于同一篇长文、描述其他事件，或只给出背景和时间，都不是直接证据。询问原因时，候选中的因果结果必须就是字段所问事件；询问人物、地点、时间、数量、名称或列表时，候选必须实际给出对应类型的具体值。
每个字段最多选择 6 条。按支持强度从高到低，只输出“字段编号:证据编号”，例如 f1:e3,f1:e7；只有一个字段时也可简写为 e3,e7。没有任何直接证据只输出 NONE。不要解释，不要改写原文。

任务契约：{json.dumps(contract, ensure_ascii=False)}
问题：{question}
候选证据：
{evidence}
当前唯一决定：为任务契约中的每个字段选择直接证据编号。
选择："""

    @staticmethod
    def _parse_adjudication(
        raw: str,
        field_ids: set[str],
        unit_count: int,
    ) -> tuple[tuple[str, int], ...]:
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "")
        first_line = next(
            (
                line.strip().lstrip(">").strip()
                for line in cleaned.splitlines()
                if line.strip() and line.strip() != ">"
            ),
            "",
        )
        if first_line.upper() in {"NONE", "[]"}:
            return ()
        if len(field_ids) == 1 and re.fullmatch(
            r"e[1-9]\d*(?:\s*[,，;；]\s*e[1-9]\d*)*",
            first_line,
            flags=re.IGNORECASE,
        ):
            field_id = next(iter(field_ids))
            return tuple(
                (field_id, int(unit_id))
                for unit_id in re.findall(
                    r"e([1-9]\d*)",
                    first_line,
                    flags=re.IGNORECASE,
                )
                if int(unit_id) <= unit_count
            )
        contract = r"(?:f[1-9]\d*\s*[:：]\s*e[1-9]\d*(?:\s*[,，;；]\s*)?)+"
        if not re.fullmatch(contract, first_line, flags=re.IGNORECASE):
            compact = re.search(
                r"((?:f[1-9]\d*\s*[:：]\s*e[1-9]\d*"
                r"|e[1-9]\d*)(?:\s*[,，;；]\s*(?:f[1-9]\d*\s*[:：]\s*)?e[1-9]\d*)*)",
                first_line,
                flags=re.IGNORECASE,
            )
            if compact is None:
                raise ValueError("adjudicator response does not match the contract")
            first_line = compact.group(1)
        output: list[tuple[str, int]] = []
        field_groups = re.findall(
            r"(f[1-9]\d*)\s*[:：]\s*"
            r"(e[1-9]\d*(?:\s*[,，;；]\s*e[1-9]\d*)*)",
            first_line,
            flags=re.IGNORECASE,
        )
        for field_id, evidence_ids in field_groups:
            normalized_field_id = field_id.lower()
            for unit_id in re.findall(r"e([1-9]\d*)", evidence_ids, flags=re.IGNORECASE):
                unit_index = int(unit_id)
                if (
                    normalized_field_id not in field_ids
                    or unit_index > unit_count
                    or (normalized_field_id, unit_index) in output
                ):
                    continue
                output.append((normalized_field_id, unit_index))
        if not output:
            raise ValueError("adjudicator selected no valid evidence ids")
        return tuple(output)

    async def _extract_source(
        self,
        question: str,
        plan: QueryPlan,
        source_index: int,
        source: SourceItem,
    ) -> tuple[tuple[EvidenceSpan, ...], dict[str, object]]:
        sentence_units = self._attention_units(plan, source)
        prompt = self._prompt(
            question,
            plan,
            source,
            sentence_units=sentence_units,
        )
        max_tokens = min(
            self.settings.evidence_extraction_max_tokens,
            160 if self.settings.semantic_pipeline_enabled else self.settings.evidence_extraction_max_tokens,
        )
        payload = {
            "contents": [prompt],
            "max_tokens": max_tokens,
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
        candidates = self._parse(
            raw,
            plan,
            source_index,
            source,
            sentence_units=sentence_units,
            semantic_mode=self.settings.semantic_pipeline_enabled,
        )
        return candidates, {
            "stage": "resolver_map",
            "source_id": source.id,
            "source_index": source_index,
            "prompt": prompt,
            "raw_output": raw,
            "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
            "raw_output_sha256": sha256(raw.encode("utf-8")).hexdigest(),
            "selected_count": len(candidates),
        }

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
        return f"""你是证据抽取器，不回答问题，也不使用资料外知识。只处理当前这一份资料。
任务契约：{json.dumps(contract, ensure_ascii=False)}
模型负责判断哪些编号句子能够支持字段问题。只选择资料中逐字存在、足以支撑字段答案的句子；不确定时不要选择。不要复制、改写或补充正文。
只输出“字段编号:句子编号”，例如 f1:s2；多条证据用逗号分隔；没有直接证据只输出 NONE。不要解释。

问题：{question}
资料标题：{source.title}
编号句子：
{text}
当前唯一任务：为字段“{field_targets}”选择能直接填写具体答案的句子编号。选中的句子如果不能直接回答该字段，就必须输出空数组。
选择："""

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
        output: list[str] = []
        used = 0
        for unit in all_units:
            remaining = limit - used
            if remaining <= 0:
                break
            value = unit[:remaining]
            output.append(value)
            used += len(value) + 1
        return tuple(output)

    @staticmethod
    def _parse(
        raw: str,
        plan: QueryPlan,
        source_index: int,
        source: SourceItem,
        *,
        sentence_units: tuple[str, ...] | None = None,
        semantic_mode: bool = False,
    ) -> tuple[EvidenceSpan, ...]:
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "")
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        field_ids = {field.field_id for field in plan.fields}
        if start >= 0 and end > start:
            try:
                payload = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError as error:
                raise ValueError("extractor response contains invalid JSON") from error
            if not isinstance(payload, dict) or "candidates" not in payload:
                raise ValueError("extractor response must contain candidates")
            values = payload["candidates"]
            if not isinstance(values, list):
                raise ValueError("extractor candidates must be a list")
        else:
            values = LanguageModelEvidenceExtractor._parse_compact_candidates(
                cleaned,
                field_ids,
            )
        candidates: list[EvidenceSpan] = []
        for item in values[:32]:
            if not isinstance(item, dict) or "field_id" not in item:
                continue
            field_id = str(item["field_id"]).strip()
            span = ""
            if set(item) == {"field_id", "sentence_id"} and sentence_units is not None:
                sentence_match = re.fullmatch(r"s([1-9]\d*)", str(item["sentence_id"]).strip())
                if sentence_match:
                    sentence_index = int(sentence_match.group(1)) - 1
                    if sentence_index < len(sentence_units):
                        span = sentence_units[sentence_index]
            elif set(item) == {"field_id", "span"}:
                span = str(item["span"]).strip()
            if field_id not in field_ids or not span:
                continue
            if len(span) > 2_000 or span not in source.snippet:
                continue
            candidates.append(EvidenceSpan(
                field_id=field_id,
                source_index=source_index,
                span=span,
                content_hash=sha256(span.encode("utf-8")).hexdigest(),
            ))
        return tuple(candidates)

    @staticmethod
    def _parse_compact_candidates(
        raw: str,
        field_ids: set[str],
    ) -> list[dict[str, str]]:
        cleaned = raw.strip().lstrip(">").strip()
        cleaned = next(
            (line.strip() for line in cleaned.splitlines() if line.strip()),
            "",
        )
        if cleaned.upper() in {"NONE", "[]"}:
            return []
        shared_field = re.fullmatch(
            r"(f[1-9]\d*)\s*[:：]\s*\[?\s*"
            r"(s[1-9]\d*(?:\s*[,，]\s*s[1-9]\d*)*)\s*\]?",
            cleaned,
            flags=re.IGNORECASE,
        )
        if shared_field is not None:
            field_id = shared_field.group(1).lower()
            if field_id not in field_ids:
                return []
            return [
                {"field_id": field_id, "sentence_id": sentence_id.lower()}
                for sentence_id in re.findall(
                    r"s[1-9]\d*",
                    shared_field.group(2),
                    flags=re.IGNORECASE,
                )
            ]
        explicit_contract = (
            r"(?:f[1-9]\d*\s*[:：]\s*\[?s[1-9]\d*\]?\s*[,，;；]?\s*)+"
        )
        explicit = (
            re.findall(
                r"\b(f[1-9]\d*)\s*[:：]\s*\[?(s[1-9]\d*)\]?",
                cleaned,
                flags=re.IGNORECASE,
            )
            if re.fullmatch(explicit_contract, cleaned, flags=re.IGNORECASE)
            else []
        )
        if explicit:
            return [
                {
                    "field_id": field_id.lower(),
                    "sentence_id": sentence_id.lower(),
                }
                for field_id, sentence_id in explicit
                if field_id.lower() in field_ids
            ]
        if len(field_ids) == 1 and re.fullmatch(
            r"(?:\[?s[1-9]\d*\]?\s*[,，;；]?\s*)+",
            cleaned,
            flags=re.IGNORECASE,
        ):
            field_id = next(iter(field_ids))
            return [
                {"field_id": field_id, "sentence_id": sentence_id.lower()}
                for sentence_id in re.findall(r"s[1-9]\d*", cleaned, flags=re.IGNORECASE)
            ]
        raise ValueError("extractor response does not match a supported output contract")
