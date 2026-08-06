import asyncio
import json
import re
from time import monotonic

import httpx

from .config import Settings
from .lexical_index import lexical_tokens, query_tokens
from .schemas import SourceItem

_NO_EVIDENCE_ANSWER = "未检索到可用于回答该问题的资料。"
_INSUFFICIENT_EVIDENCE_ANSWER = "根据检索到的资料，无法确定。"
_CONTINUATION_MARKERS = (
    "\n用户",
    "\nUser",
    "\n问题",
    "\nQuestion",
    "\n助手",
    "\n[资料",
    "\n资料：",
)
_THINKING_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_CITATION_PATTERN = re.compile(r"\[资料\s*([1-9]\d*)\]")
_GROUNDING_STOP_TOKENS = {"个", "分别", "相关", "内容", "资料", "问题", "请问", "一下"}
_GROUNDING_CONTEXT_TERMS = {"中国", "中华人民共和国"}


class EvidenceAssessment:
    def __init__(self, question_terms: set[str], matched_terms: set[str]) -> None:
        self.question_terms = question_terms
        self.matched_terms = matched_terms

    @property
    def specific_terms(self) -> set[str]:
        return self.question_terms - _GROUNDING_CONTEXT_TERMS

    @property
    def matched_specific_terms(self) -> set[str]:
        return self.specific_terms & self.matched_terms

    @property
    def grounded(self) -> bool:
        if self.specific_terms:
            return bool(self.matched_specific_terms)
        return bool(self.matched_terms)


class AnswerGenerationError(RuntimeError):
    """Raised when the configured text generation service cannot answer."""


class EvidenceAnswerGenerator:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport

    async def generate(self, question: str, sources: list[SourceItem]) -> str:
        if not sources:
            return _NO_EVIDENCE_ANSWER
        if not self.assess_evidence(question, sources).grounded:
            return _INSUFFICIENT_EVIDENCE_ANSWER
        if not self.settings.generation_password:
            raise AnswerGenerationError("RWKVRAG_GENERATION_PASSWORD is not configured")

        payload = {
            "contents": [self._prompt(question, sources)],
            "max_tokens": self.settings.generation_max_tokens,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.6,
            "alpha_presence": 1.0,
            "alpha_frequency": 0.1,
            "alpha_decay": 0.99,
            "stream": True,
            "password": self.settings.generation_password,
        }
        endpoint = f"{self.settings.generation_base_url.rstrip('/')}/chat/completions"
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.generation_timeout,
                transport=self.transport,
            ) as client:
                async with client.stream("POST", endpoint, json=payload) as response:
                    response.raise_for_status()
                    answer = await self._read_stream(
                        response,
                        total_timeout=self.settings.generation_total_timeout,
                    )
        except httpx.HTTPError as error:
            raise AnswerGenerationError(f"generation request failed: {error}") from error

        answer = self._clean_answer(answer)
        if not answer:
            return _INSUFFICIENT_EVIDENCE_ANSWER
        if answer == _INSUFFICIENT_EVIDENCE_ANSWER:
            return answer
        return self._ensure_citation(answer, len(sources))

    @staticmethod
    def assess_evidence(question: str, sources: list[SourceItem]) -> EvidenceAssessment:
        question_terms = {
            token
            for token in query_tokens(question)
            if token not in _GROUNDING_STOP_TOKENS
        }
        evidence_terms = {
            token
            for source in sources
            for token in lexical_tokens(f"{source.title}\n{source.snippet}")
        }
        return EvidenceAssessment(question_terms, question_terms & evidence_terms)

    def _prompt(self, question: str, sources: list[SourceItem]) -> str:
        evidence = self._evidence(sources)
        return f"""用户：根据资料回答问题，只回答答案，不要重复题目或资料。只能使用资料内容；资料不能明确答案时回答“{_INSUFFICIENT_EVIDENCE_ANSWER}”。每个结论句末尾必须标注支持它的资料编号，格式为“[资料 1]”。回答应简洁，不超过500个中文字符。
资料：
{evidence}
问题：{question}
助手："""

    def _evidence(self, sources: list[SourceItem]) -> str:
        blocks: list[str] = []
        used = 0
        limit = self.settings.generation_max_evidence_characters
        for index, source in enumerate(sources, start=1):
            title = source.title or "未命名资料"
            text = source.snippet.strip()
            block = f"[资料 {index}] 标题：{title}\n{text}"
            remaining = limit - used
            if remaining <= 0:
                break
            if len(block) > remaining:
                block = block[:remaining].rstrip() + "…"
            blocks.append(block)
            used += len(block)
            if used >= limit:
                break
        return "\n\n---\n\n".join(blocks)

    @staticmethod
    async def _read_stream(response: httpx.Response, *, total_timeout: float) -> str:
        parts: list[str] = []
        lines = response.aiter_lines()
        deadline = monotonic() + total_timeout
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            try:
                line = await asyncio.wait_for(anext(lines), timeout=remaining)
            except (StopAsyncIteration, TimeoutError):
                break
            if not line.startswith("data:"):
                continue
            event = line.removeprefix("data:").strip()
            if event == "[DONE]":
                break
            try:
                payload = json.loads(event)
                content = payload["choices"][0]["delta"].get("content", "")
            except (IndexError, KeyError, TypeError, json.JSONDecodeError):
                continue
            if isinstance(content, str):
                parts.append(content)
        return "".join(parts)

    @staticmethod
    def _clean_answer(answer: str) -> str:
        cleaned = _THINKING_PATTERN.sub("", answer).strip()
        for prefix in ("Assistant:", "assistant:", "助手："):
            if cleaned.startswith(prefix):
                cleaned = cleaned.removeprefix(prefix).strip()
        for marker in _CONTINUATION_MARKERS:
            position = cleaned.find(marker)
            if position >= 0:
                cleaned = cleaned[:position].rstrip()
        return cleaned

    @staticmethod
    def _has_valid_citation(answer: str, source_count: int) -> bool:
        return any(
            1 <= int(match.group(1)) <= source_count
            for match in _CITATION_PATTERN.finditer(answer)
        )

    @classmethod
    def _ensure_citation(cls, answer: str, source_count: int) -> str:
        if cls._has_valid_citation(answer, source_count):
            return answer
        return f"{answer.rstrip()} [资料 1]"
