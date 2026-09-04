import asyncio
from dataclasses import dataclass
from hashlib import sha256
import json
import re
from time import monotonic

import httpx

from .config import Settings
from .evidence_utils import clean_evidence_text
from .lexical_index import lexical_tokens, normalize_search_text, query_tokens
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
    r"</?(?:answer|think|no|tool_call|tool_calls|tool_code|tool_result)\b",
    re.IGNORECASE,
)


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
        return self.question_terms

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


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    prompt: str
    raw_output: str

    @property
    def prompt_sha256(self) -> str:
        return sha256(self.prompt.encode("utf-8")).hexdigest()

    @property
    def raw_output_sha256(self) -> str:
        return sha256(self.raw_output.encode("utf-8")).hexdigest()


class EvidenceAnswerGenerator:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self._model_cache: tuple[float, str | None] | None = None

    async def generate(
        self,
        question: str,
        sources: list[SourceItem],
        *,
        subject: str = "",
        relations: tuple[str, ...] = (),
        trusted_evidence: bool = False,
        answer_shape: str = "single_fact",
        set_semantics: str = "specific",
        fields: tuple[tuple[str, str, tuple[str, ...]], ...] = (),
    ) -> str:
        result = await self.generate_with_trace(
            question,
            sources,
            subject=subject,
            relations=relations,
            trusted_evidence=trusted_evidence,
            answer_shape=answer_shape,
            set_semantics=set_semantics,
            fields=fields,
        )
        return result.answer

    async def generate_with_trace(
        self,
        question: str,
        sources: list[SourceItem],
        *,
        subject: str = "",
        relations: tuple[str, ...] = (),
        trusted_evidence: bool = False,
        answer_shape: str = "single_fact",
        set_semantics: str = "specific",
        fields: tuple[tuple[str, str, tuple[str, ...]], ...] = (),
    ) -> GenerationResult:
        if not sources:
            return GenerationResult(_NO_EVIDENCE_ANSWER, "", "")
        if (
            self.settings.generation_output_mode == "legacy"
            and
            not trusted_evidence
            and not self.assess_evidence(question, sources, subject=subject).grounded
        ):
            return GenerationResult(_INSUFFICIENT_EVIDENCE_ANSWER, "", "")
        if not self.settings.generation_password:
            raise AnswerGenerationError("RWKVRAG_GENERATION_PASSWORD is not configured")

        prompt = self._prompt(
                question,
                sources,
                subject=subject,
                relations=relations,
                answer_shape=answer_shape,
                set_semantics=set_semantics,
                fields=fields,
            )
        payload = {
            "contents": [prompt],
            "max_tokens": self.settings.generation_max_tokens,
            "temperature": 0.2,
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
                    raw_answer = await self._read_stream(
                        response,
                        total_timeout=self.settings.generation_total_timeout,
                    )
        except httpx.HTTPError as error:
            raise AnswerGenerationError(f"generation request failed: {error}") from error

        if self.settings.generation_output_mode == "immutable":
            return GenerationResult(raw_answer, prompt, raw_answer)
        answer = self._clean_answer(raw_answer)
        if not answer:
            answer = _INSUFFICIENT_EVIDENCE_ANSWER
        if answer == _INSUFFICIENT_EVIDENCE_ANSWER:
            return GenerationResult(answer, prompt, raw_answer)
        if not self.settings.semantic_pipeline_enabled:
            answer = self._ensure_citation(answer, len(sources))
        return GenerationResult(answer, prompt, raw_answer)

    async def current_model(self) -> str | None:
        if self._model_cache is not None:
            cached_at, model_name = self._model_cache
            if monotonic() - cached_at < 300:
                return model_name
        try:
            async with httpx.AsyncClient(
                timeout=min(self.settings.generation_timeout, 5),
                transport=self.transport,
            ) as client:
                response = await client.get(self.settings.generation_models_url)
                response.raise_for_status()
        except httpx.HTTPError:
            self._model_cache = (monotonic(), None)
            return None

        try:
            models = response.json().get("data")
        except (AttributeError, json.JSONDecodeError):
            self._model_cache = (monotonic(), None)
            return None
        if not isinstance(models, list):
            self._model_cache = (monotonic(), None)
            return None
        for model in models:
            if not isinstance(model, dict):
                continue
            model_id = model.get("id")
            if isinstance(model_id, str) and model_id.strip():
                model_name = model_id.strip()
                self._model_cache = (monotonic(), model_name)
                return model_name
        self._model_cache = (monotonic(), None)
        return None

    @staticmethod
    def assess_evidence(
        question: str,
        sources: list[SourceItem],
        *,
        subject: str = "",
    ) -> EvidenceAssessment:
        question_terms = {
            token for token in query_tokens(question) if len(token.strip()) >= 2
        }
        evidence_text = normalize_search_text("\n".join(
            "\n".join((source.title, source.snippet, *_source_alias_values(source)))
            for source in sources
        ))
        evidence_terms = {
            token
            for token in lexical_tokens(evidence_text)
        }
        anchors = _entity_anchors(question, subject=subject)
        matched_anchors = _matched_anchor_terms(
            anchors,
            evidence_terms,
            evidence_text=evidence_text,
        )
        return EvidenceAssessment(
            question_terms,
            question_terms & evidence_terms,
            anchors=anchors,
            matched_anchors=matched_anchors,
        )

    def _prompt(
        self,
        question: str,
        sources: list[SourceItem],
        *,
        subject: str = "",
        relations: tuple[str, ...] = (),
        answer_shape: str = "single_fact",
        set_semantics: str = "specific",
        fields: tuple[tuple[str, str, tuple[str, ...]], ...] = (),
    ) -> str:
        if self.settings.generation_output_mode == "immutable":
            evidence = self._evidence(sources)
            return f"""system:
知识库问答助手；只能依据资料；不足则说明；关键结论标注 [资料 1]、[资料 2]。同一事实或列表项只输出一次，不要重复。

user:
资料：
{evidence}

问题：{question}

assistant:
"""
        evidence = self._evidence(sources)
        normalized_fields = [
            {
                "field_id": field_id,
                "question": field_question,
                "relations": list(field_relations),
            }
            for field_id, field_question, field_relations in fields
        ] or [{
            "field_id": "f1",
            "question": question,
            "relations": list(relations),
        }]
        task_contract = json.dumps(
            {
                "subject": subject or question,
                "relations": list(relations),
                "question": question,
                "answer_shape": answer_shape,
                "set_semantics": set_semantics,
                "fields": normalized_fields,
            },
            ensure_ascii=False,
        )
        return f"""system:
知识库问答助手；只能依据资料；资料不足时说明；关键结论标注对应资料编号。

user:
问题：{question}
任务契约：{task_contract}
资料：
{evidence}
请只依据资料回答；资料不足时明确说明。

assistant:
"""

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
        cleaned = re.sub(r"^(?:\s*>\s*)+", "", cleaned).strip()
        cleaned = re.sub(r"^[\]}>]+\s*", "", cleaned).strip()
        cleaned = re.sub(r"(?:\n\s*>\s*)+$", "", cleaned).strip()
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


def _entity_anchors(question: str, *, subject: str = "") -> set[str]:
    normalized_subject = normalize_search_text(subject).replace(" ", "")
    normalized_subject = normalized_subject.strip("《》“”\"'，,。；;？?")
    if 2 <= len(normalized_subject) <= 80:
        return {normalized_subject}
    return set()


def _matched_anchor_terms(
    anchors: set[str],
    evidence_terms: set[str],
    *,
    evidence_text: str = "",
) -> set[str]:
    normalized_evidence = evidence_text.replace(" ", "")
    matched = {
        anchor
        for anchor in anchors
        if _anchor_supported(anchor, evidence_terms, normalized_evidence)
    }
    return matched


def _anchor_supported(
    anchor: str,
    evidence_terms: set[str],
    normalized_evidence: str,
) -> bool:
    normalized_anchor = normalize_search_text(anchor).replace(" ", "")
    if normalized_anchor and normalized_anchor in normalized_evidence:
        return True
    anchor_tokens = set(lexical_tokens(normalized_anchor))
    return bool(anchor_tokens) and anchor_tokens <= evidence_terms


def _source_alias_values(source: SourceItem) -> tuple[str, ...]:
    aliases = source.metadata.get("aliases")
    if not isinstance(aliases, list):
        return ()
    return tuple(str(alias).strip() for alias in aliases if str(alias).strip())


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
