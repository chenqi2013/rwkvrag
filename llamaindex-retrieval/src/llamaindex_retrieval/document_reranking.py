import asyncio
from dataclasses import dataclass
import json
import re

import httpx

from .config import Settings
from .evidence_utils import clean_evidence_text
from .generation import EvidenceAnswerGenerator
from .query_planning import QueryPlan
from .schemas import SourceItem


@dataclass(frozen=True)
class DocumentDecision:
    document_id: str
    relevant: bool
    score: int
    reason: str


@dataclass(frozen=True)
class DocumentRerankResult:
    sources: tuple[SourceItem, ...]
    decisions: tuple[DocumentDecision, ...]
    errors: tuple[str, ...] = ()
    strategy: str = "model"


class LanguageModelDocumentReranker:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport

    async def rerank(
        self,
        question: str,
        plan: QueryPlan,
        sources: list[SourceItem],
    ) -> DocumentRerankResult:
        if not self.settings.document_reranking_enabled or not sources:
            return DocumentRerankResult(tuple(sources), (), strategy="disabled")
        if not self.settings.generation_password:
            return DocumentRerankResult(
                tuple(sources),
                (),
                errors=("generation_password_not_configured",),
                strategy="unavailable",
            )

        grouped: dict[str, list[SourceItem]] = {}
        for source in sources:
            grouped.setdefault(source.document_id, []).append(source)
        document_limit = min(
            self.settings.document_reranking_max_documents,
            3 if plan.answer_shape == "single_fact" else 5,
        )
        document_ids = tuple(grouped)[:document_limit]
        if self.settings.semantic_pipeline_enabled:
            return await self._rerank_batch(
                question,
                plan,
                sources,
                grouped,
                document_ids,
            )
        semaphore = asyncio.Semaphore(self.settings.document_reranking_concurrency)

        async def score(document_id: str) -> DocumentDecision:
            async with semaphore:
                return await self._score_document(
                    question,
                    plan,
                    document_id,
                    grouped[document_id],
                )

        results = await asyncio.gather(
            *(score(document_id) for document_id in document_ids),
            return_exceptions=True,
        )
        decisions: list[DocumentDecision] = []
        errors: list[str] = []
        for document_id, result in zip(document_ids, results, strict=True):
            if isinstance(result, BaseException):
                errors.append(f"{document_id}: {type(result).__name__}: {result}")
                continue
            decisions.append(result)
        if not decisions:
            return DocumentRerankResult(
                tuple(sources),
                (),
                errors=tuple(errors),
                strategy="model_error_fallback",
            )

        source_order = {document_id: index for index, document_id in enumerate(grouped)}
        selected = [decision for decision in decisions if decision.relevant]
        selected.sort(
            key=lambda decision: (
                -decision.score,
                source_order[decision.document_id],
            )
        )
        if not selected:
            return DocumentRerankResult(
                tuple(sources),
                tuple(decisions),
                errors=tuple(errors),
                strategy="model_no_relevant_document",
            )
        selected_ids = {decision.document_id for decision in selected}
        scored_ids = {decision.document_id for decision in decisions}
        ordered_sources = tuple(
            source
            for decision in selected
            for source in grouped[decision.document_id]
            if source.document_id in selected_ids
        )
        ordered_sources += tuple(
            source
            for source in sources
            if source.document_id not in scored_ids
        )
        return DocumentRerankResult(
            ordered_sources,
            tuple(decisions),
            errors=tuple(errors),
        )

    async def _rerank_batch(
        self,
        question: str,
        plan: QueryPlan,
        sources: list[SourceItem],
        grouped: dict[str, list[SourceItem]],
        document_ids: tuple[str, ...],
    ) -> DocumentRerankResult:
        prompt = self._batch_prompt(question, plan, grouped, document_ids)
        payload = {
            "contents": [prompt],
            "max_tokens": self.settings.document_reranking_max_tokens,
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
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.document_reranking_timeout,
                transport=self.transport,
            ) as client:
                async with client.stream("POST", endpoint, json=payload) as response:
                    response.raise_for_status()
                    raw = await EvidenceAnswerGenerator._read_stream(
                        response,
                        total_timeout=self.settings.document_reranking_timeout,
                    )
            selected_indexes = self._parse_batch_selection(raw, len(document_ids))
        except Exception as error:
            return DocumentRerankResult(
                tuple(sources),
                (),
                errors=(f"{type(error).__name__}: {error}",),
                strategy="model_error_fallback",
            )
        decisions = tuple(
            DocumentDecision(
                document_id=document_id,
                relevant=index in selected_indexes,
                score=(3 if selected_indexes and index == selected_indexes[0] else 2)
                if index in selected_indexes
                else 0,
                reason="model_selected" if index in selected_indexes else "model_rejected",
            )
            for index, document_id in enumerate(document_ids, start=1)
        )
        if not selected_indexes:
            return DocumentRerankResult(
                tuple(sources),
                decisions,
                strategy="model_no_relevant_document",
            )
        selected_ids = [document_ids[index - 1] for index in selected_indexes]
        extracted_evidence = all(
            source.metadata.get("evidence_span_hashes")
            for document_id in document_ids
            for source in grouped[document_id]
        )
        safety_count = (
            0
            if extracted_evidence
            else 2 if plan.answer_shape in {"list", "summary", "narrative"} else 1
        )
        for document_id in document_ids[:safety_count]:
            if document_id not in selected_ids:
                selected_ids.append(document_id)
        scored_ids = set(document_ids)
        ordered_sources = tuple(
            source
            for document_id in selected_ids
            for source in grouped[document_id]
        )
        ordered_sources += tuple(
            source
            for source in sources
            if source.document_id not in scored_ids
        )
        return DocumentRerankResult(
            ordered_sources,
            decisions,
            strategy=(
                "model_batch_evidence"
                if extracted_evidence
                else "model_batch_with_lexical_safety"
            ),
        )

    def _batch_prompt(
        self,
        question: str,
        plan: QueryPlan,
        grouped: dict[str, list[SourceItem]],
        document_ids: tuple[str, ...],
    ) -> str:
        contract = json.dumps(
            {
                "subject": plan.subject,
                "fields": [
                    {
                        "field_id": field.field_id,
                        "question": field.question,
                        "relations": list(field.relations),
                    }
                    for field in plan.fields
                ],
                "answer_shape": plan.answer_shape,
                "set_semantics": plan.set_semantics,
            },
            ensure_ascii=False,
        )
        per_document_limit = max(
            300,
            self.settings.document_reranking_max_characters // max(1, len(document_ids)),
        )
        blocks: list[str] = []
        for index, document_id in enumerate(document_ids, start=1):
            document_sources = grouped[document_id]
            excerpt = "\n".join(
                clean_evidence_text(source.snippet)
                for source in document_sources
            )[:per_document_limit]
            blocks.append(
                f"[d{index}] 标题：{document_sources[0].title or '未命名'}\n{excerpt}"
            )
        return f"""你是知识库候选文档选择器，不回答问题。选择能够直接支持任务字段的完整文档。
标题不必与任务对象相同；列表、目录、人物页只要正文明确标识任务对象并给出所求值，也要选择。仅有同词、同名异物、导航或背景提及时不要选择。
选择所有可能直接支持字段的文档，按相关性从高到低输出，最多 3 个，例如 d2,d1。宁可保留可能直接支持的候选，也不要只因标题不同而漏掉列表页、目录页或人物页。没有任何文档能直接支持时只输出 NONE。不要解释。

任务契约：{contract}
问题：{question}
候选文档：
{chr(10).join(blocks)}
选择："""

    @staticmethod
    def _parse_batch_selection(raw: str, document_count: int) -> tuple[int, ...]:
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "")
        cleaned = cleaned.strip().lstrip(">").strip()
        first_line = next(
            (line.strip() for line in cleaned.splitlines() if line.strip()),
            "",
        )
        if first_line.upper() == "NONE":
            return ()
        if not re.fullmatch(
            r"d[1-9]\d*(?:\s*[,，]\s*d[1-9]\d*)*",
            first_line,
            flags=re.IGNORECASE,
        ):
            raise ValueError("batch reranker response does not match the contract")
        selected: list[int] = []
        for value in re.findall(r"d([1-9]\d*)", first_line, flags=re.IGNORECASE):
            index = int(value)
            if index > document_count or index in selected:
                continue
            selected.append(index)
        if not selected:
            raise ValueError("batch reranker selected no valid document ids")
        return tuple(selected)

    async def _score_document(
        self,
        question: str,
        plan: QueryPlan,
        document_id: str,
        sources: list[SourceItem],
    ) -> DocumentDecision:
        prompt = self._prompt(question, plan, sources)
        payload = {
            "contents": [prompt],
            "max_tokens": self.settings.document_reranking_max_tokens,
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
            timeout=self.settings.document_reranking_timeout,
            transport=self.transport,
        ) as client:
            async with client.stream("POST", endpoint, json=payload) as response:
                response.raise_for_status()
                raw = await EvidenceAnswerGenerator._read_stream(
                    response,
                    total_timeout=self.settings.document_reranking_timeout,
                )
        relevant, score, reason = self._parse(raw)
        return DocumentDecision(document_id, relevant, score, reason)

    def _prompt(
        self,
        question: str,
        plan: QueryPlan,
        sources: list[SourceItem],
    ) -> str:
        used = 0
        excerpts: list[str] = []
        limit = self.settings.document_reranking_max_characters
        for source in sources:
            text = clean_evidence_text(source.snippet)
            remaining = limit - used
            if remaining <= 0:
                break
            excerpts.append(text[:remaining])
            used += min(len(text), remaining)
        contract = json.dumps(
            {
                "subject": plan.subject,
                "fields": [
                    {
                        "field_id": field.field_id,
                        "question": field.question,
                        "relations": list(field.relations),
                    }
                    for field in plan.fields
                ],
                "answer_shape": plan.answer_shape,
                "set_semantics": plan.set_semantics,
            },
            ensure_ascii=False,
        )
        return f"""你是知识库文档相关性裁决器，只判断当前这一篇文档是否值得进入证据抽取，不回答问题。
判断对象、关系和答案字段是否同时可能由本文直接支持。仅仅出现一个相同词、同名异物、导航目录或背景提及，均判为不相关。文档标题不必与任务对象完全相同，不能只因标题不同判为无关；列表、目录或人物页面只要正文明确标识任务对象，并紧接着给出所求列表或具体值，就必须判为 2 或 3。
分数只能是 0、1、2、3：0=无关，1=弱相关，2=可能直接支持，3=明确直接支持。
只输出“分数|理由”，例如“3|直接给出所求事实”或“0|同名异物”。不要输出 JSON，不要解释。

任务契约：{contract}
原问题：{question}
文档标题：{sources[0].title or "未命名"}
文档片段：
{chr(10).join(excerpts)}
选择："""

    @staticmethod
    def _parse(raw: str) -> tuple[bool, int, str]:
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "")
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            compact = cleaned.strip().lstrip(">").strip()
            compact = next(
                (line.strip() for line in compact.splitlines() if line.strip()),
                "",
            )
            match = re.fullmatch(
                r"([0-3])\s*[|｜:：-]\s*([^\n|｜]{1,80})",
                compact,
            )
            if match is None:
                raise ValueError("reranker response does not match a supported contract")
            score = int(match.group(1))
            reason = match.group(2).strip()
            return score >= 2, score, reason[:60]
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as error:
            raise ValueError("reranker response contains invalid JSON") from error
        if not isinstance(payload, dict) or set(payload) != {"relevant", "score", "reason"}:
            raise ValueError("reranker response does not match the contract")
        relevant = payload["relevant"]
        score = payload["score"]
        reason = payload["reason"]
        if not isinstance(relevant, bool):
            raise ValueError("reranker relevant must be boolean")
        if not isinstance(score, int) or score not in {0, 1, 2, 3}:
            raise ValueError("reranker score must be between 0 and 3")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reranker reason must be non-empty")
        if relevant != (score >= 2):
            raise ValueError("reranker relevant and score disagree")
        return relevant, score, reason.strip()[:60]
