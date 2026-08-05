import json
import re

import httpx

from .config import Settings
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
                    answer = await self._read_stream(response)
        except httpx.HTTPError as error:
            raise AnswerGenerationError(f"generation request failed: {error}") from error

        answer = self._clean_answer(answer)
        return answer or _INSUFFICIENT_EVIDENCE_ANSWER

    def _prompt(self, question: str, sources: list[SourceItem]) -> str:
        evidence = self._evidence(sources)
        return f"""用户：根据资料回答问题，只回答答案，不要重复题目或资料。只能使用资料内容；资料不能明确答案时回答“{_INSUFFICIENT_EVIDENCE_ANSWER}”。
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
    async def _read_stream(response: httpx.Response) -> str:
        parts: list[str] = []
        async for line in response.aiter_lines():
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
