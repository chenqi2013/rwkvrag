import asyncio
import json
import re
from time import monotonic
from typing import Iterable

import httpx

from .config import Settings
from .evidence_utils import clean_evidence_text
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
    "\n[助手",
    "\n[assistant",
    "\n[资料",
    "\n资料：",
    "\n[用户",
    "\n[User",
)
_THINKING_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_BRACKETED_THINKING_PATTERN = re.compile(r"\[\s*(?:思考|推理|分析)\s*\].*?(?=\[\s*(?:回答|答案)\s*\]|$)", re.DOTALL)
_BRACKETED_ANSWER_PREFIX_PATTERN = re.compile(r"^\s*\[\s*(?:回答|答案)\s*\]\s*")
_ASSISTANT_PREFIX_PATTERN = re.compile(
    r"^(?:Assistant:|assistant:|助手：|\[助手(?:\s+\d+)?\]|\[assistant(?:\s+\d+)?\])\s*"
)
_ROLE_HEADER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:Assistant:|assistant:|助手：|\[助手(?:\s+\d+)?\]|\[assistant(?:\s+\d+)?\])\s*"
)
_USER_ROLE_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:User:|user:|用户：|\[用户(?:\s+\d+)?\]|\[user(?:\s+\d+)?\])\s*"
)
_CITATION_PATTERN = re.compile(r"\[资料\s*([1-9]\d*)\]")
_EVIDENCE_LABEL_PATTERN = re.compile(r"(?:^|\n)\s*\[资料\s*[1-9]\d*\]\s*标题：")
_PROTOCOL_TAG_PATTERN = re.compile(
    r"</?(?:answer|tool_call|tool_calls|tool_code|tool_result)\b",
    re.IGNORECASE,
)
_GROUNDING_STOP_TOKENS = {"个", "分别", "相关", "内容", "资料", "问题", "请问", "一下"}
_GROUNDING_CONTEXT_TERMS = {"中国", "中华人民共和国"}
_ANCHOR_EQUIVALENTS = {
    "中国": {"中华人民共和国"},
    "中华人民共和国": {"中国"},
}
_ANCHOR_NOISE = {
    *_GROUNDING_STOP_TOKENS,
    "哪个",
    "哪些",
    "哪里",
    "什么",
    "怎么",
    "如何",
    "首都",
    "国都",
    "城市",
    "中心",
    "政治",
    "人口",
    "最多",
    "省",
    "省份",
    "民族",
    "分别",
    "多少",
    "几个",
    "丰功",
    "伟绩",
    "功绩",
    "功业",
    "贡献",
    "成就",
    "站点",
    "车站",
    "列表",
    "全部",
    "所有",
}


class EvidenceAssessment:
    def __init__(
        self,
        question_terms: set[str],
        matched_terms: set[str],
        anchors: set[str] | None = None,
        matched_anchors: set[str] | None = None,
    ) -> None:
        self.question_terms = question_terms
        self.matched_terms = matched_terms
        self.anchors = anchors or set()
        self.matched_anchors = matched_anchors or set()

    @property
    def specific_terms(self) -> set[str]:
        return self.question_terms - _GROUNDING_CONTEXT_TERMS

    @property
    def matched_specific_terms(self) -> set[str]:
        return self.specific_terms & self.matched_terms

    @property
    def grounded(self) -> bool:
        if self.anchors and not self.matched_anchors:
            return False
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

    async def current_model(self) -> str | None:
        try:
            async with httpx.AsyncClient(
                timeout=min(self.settings.generation_timeout, 5),
                transport=self.transport,
            ) as client:
                response = await client.get(self.settings.generation_models_url)
                response.raise_for_status()
        except httpx.HTTPError:
            return None

        try:
            models = response.json().get("data")
        except (AttributeError, json.JSONDecodeError):
            return None
        if not isinstance(models, list):
            return None
        for model in models:
            if not isinstance(model, dict):
                continue
            model_id = model.get("id")
            if isinstance(model_id, str) and model_id.strip():
                return model_id.strip()
        return None

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
        anchors = _entity_anchors(question)
        matched_anchors = _matched_anchor_terms(anchors, evidence_terms)
        return EvidenceAssessment(
            question_terms,
            question_terms & evidence_terms,
            anchors=anchors,
            matched_anchors=matched_anchors,
        )

    def _prompt(self, question: str, sources: list[SourceItem]) -> str:
        evidence = self._evidence(sources)
        return f"""用户：根据资料回答问题，只回答答案，不要重复题目或资料。只能使用资料内容；资料不能明确答案时回答“{_INSUFFICIENT_EVIDENCE_ANSWER}”。每个结论句末尾必须标注支持它的资料编号，格式为“[资料 1]”。如果资料提供完整列表，必须保留列表中的每一项，不能省略、不能自行补充、不能改变顺序。回答应简洁，不超过500个中文字符。
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
            text = clean_evidence_text(source.snippet)
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
        cleaned = _BRACKETED_THINKING_PATTERN.sub("", cleaned).strip()
        cleaned = _BRACKETED_ANSWER_PREFIX_PATTERN.sub("", cleaned).strip()
        role_matches = list(_ROLE_HEADER_PATTERN.finditer(cleaned))
        if role_matches:
            cleaned = cleaned[role_matches[-1].end() :].strip()
        cleaned = _ASSISTANT_PREFIX_PATTERN.sub("", cleaned).strip()
        user_match = _USER_ROLE_PATTERN.search(cleaned)
        if user_match:
            cleaned = cleaned[: user_match.start()].rstrip()
        for marker in _CONTINUATION_MARKERS:
            position = cleaned.find(marker)
            if position >= 0:
                cleaned = cleaned[:position].rstrip()
        if _looks_like_protocol_payload(cleaned):
            return ""
        if _EVIDENCE_LABEL_PATTERN.search(cleaned):
            return ""
        return cleaned

    @staticmethod
    def _has_valid_citation(answer: str, source_count: int) -> bool:
        return any(
            1 <= int(match.group(1)) <= source_count
            for match in _CITATION_PATTERN.finditer(answer)
        )

    @classmethod
    def _ensure_citation(cls, answer: str, source_count: int) -> str:
        sanitized = _sanitize_citations(answer, source_count).strip()
        if cls._has_valid_citation(sanitized, source_count):
            return sanitized
        return f"{sanitized.rstrip()} [资料 1]"


def _entity_anchors(question: str) -> set[str]:
    tokens = [
        token
        for token in query_tokens(question)
        if token not in _ANCHOR_NOISE and len(token.strip()) >= 2
    ]
    anchors = set(_identifier_like(tokens))
    if anchors:
        return anchors
    return set(tokens[:3])


def _identifier_like(tokens: Iterable[str]) -> list[str]:
    output: list[str] = []
    for token in tokens:
        if any(character.isdigit() for character in token):
            output.append(token)
            continue
        if token.isascii() and len(token) >= 3:
            output.append(token)
            continue
        if len(token) >= 3:
            output.append(token)
    return output


def _matched_anchor_terms(anchors: set[str], evidence_terms: set[str]) -> set[str]:
    matched = anchors & evidence_terms
    for anchor in anchors:
        if _ANCHOR_EQUIVALENTS.get(anchor, set()) & evidence_terms:
            matched.add(anchor)
    return matched


def _looks_like_protocol_payload(answer: str) -> bool:
    value = answer.strip()
    if not value:
        return True
    if "根据资料回答问题" in value or "只能使用资料内容" in value:
        return True
    if _PROTOCOL_TAG_PATTERN.search(value):
        return True
    if _USER_ROLE_PATTERN.search(value) or _ROLE_HEADER_PATTERN.search(value):
        return True
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(decoded, dict) or (
        isinstance(decoded, list)
        and not all(isinstance(item, str) and _CITATION_PATTERN.search(item) for item in decoded)
    )


def _sanitize_citations(answer: str, source_count: int) -> str:
    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if 1 <= index <= source_count:
            return match.group(0)
        return ""

    cleaned = _CITATION_PATTERN.sub(replace, answer)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()
