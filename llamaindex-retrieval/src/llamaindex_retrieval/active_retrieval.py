import json
import re
from dataclasses import dataclass
import httpx

from .config import Settings
from .evidence_utils import clean_evidence_text
from .generation import EvidenceAnswerGenerator
from .schemas import SourceItem


@dataclass(frozen=True)
class ActiveRetrievalDecision:
    action: str
    queries: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class ActiveRetrievalResult:
    decision: ActiveRetrievalDecision | None
    error: str | None = None


class ActiveRetrievalAgent:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport

    async def decide(
        self,
        question: str,
        sources: list[SourceItem],
        *,
        used_queries: tuple[str, ...],
        round_number: int,
    ) -> ActiveRetrievalResult:
        if not self.settings.active_retrieval_enabled:
            return ActiveRetrievalResult(None, "disabled")
        if not self.settings.generation_password:
            return ActiveRetrievalResult(None, "generation_password_not_configured")
        try:
            raw = await self._request(
                self._prompt(
                    question,
                    sources,
                    used_queries=used_queries,
                    round_number=round_number,
                )
            )
            return ActiveRetrievalResult(self._parse(raw))
        except (httpx.HTTPError, TimeoutError, ValueError) as error:
            return ActiveRetrievalResult(None, f"{type(error).__name__}: {error}")

    async def _request(self, prompt: str) -> str:
        payload = {
            "contents": [prompt],
            "max_tokens": self.settings.active_retrieval_max_tokens,
            "temperature": 0.1,
            "top_k": 30,
            "top_p": 0.5,
            "alpha_presence": 0.2,
            "alpha_frequency": 0.1,
            "alpha_decay": 0.99,
            "stream": True,
            "password": self.settings.generation_password,
        }
        endpoint = f"{self.settings.generation_base_url.rstrip('/')}/chat/completions"
        timeout = min(
            self.settings.generation_timeout,
            self.settings.active_retrieval_timeout,
        )
        async with httpx.AsyncClient(timeout=timeout, transport=self.transport) as client:
            async with client.stream("POST", endpoint, json=payload) as response:
                response.raise_for_status()
                return await EvidenceAnswerGenerator._read_stream(
                    response,
                    total_timeout=self.settings.active_retrieval_timeout,
                )

    def _prompt(
        self,
        question: str,
        sources: list[SourceItem],
        *,
        used_queries: tuple[str, ...],
        round_number: int,
    ) -> str:
        evidence = self._evidence(sources)
        used = json.dumps(used_queries, ensure_ascii=False)
        return f"""你是中文知识库的 BM25 补充查询规划器，不负责回答问题。
当前检索结果不足，请生成 1 至 {self.settings.active_retrieval_max_queries} 条新的短关键词查询。
查询必须以原问题为准并保留核心对象，针对当前错误结果换用可能的百科标题、别名、关系词或结构词；当前证据中的对象若未出现在原问题中，视为可能误召回，不得直接拿它替换原问题对象。
新查询必须与已执行查询在核心关键词上明显不同，不能只是调整空格或语序。若原问题使用昵称、简称、颜色名、旧称或口语称呼，应优先生成可能的标准实体名称作为不同查询；可以生成多个标准名称候选，但不要把问题所求的事实答案写入查询。
后端只会执行只读 BM25 搜索，不提供 Bash、rg、网络或其他工具。

只输出一个 JSON 对象，字段固定：
{{"queries":["查询1","查询2","查询3"]}}

轮次：{round_number}
原问题：{question}
已执行查询：{used}
当前证据：
{evidence}
JSON："""

    def _evidence(self, sources: list[SourceItem]) -> str:
        blocks: list[str] = []
        used = 0
        limit = self.settings.active_retrieval_max_evidence_characters
        for index, source in enumerate(sources, start=1):
            text = clean_evidence_text(source.snippet)
            block = f"[{index}] 标题：{source.title}\n{text}"
            remaining = limit - used
            if remaining <= 0:
                break
            if len(block) > remaining:
                block = block[:remaining]
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks) or "（无证据）"

    def _parse(self, raw: str) -> ActiveRetrievalDecision:
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "")
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("active retrieval response does not contain a JSON object")
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as error:
            raise ValueError("active retrieval response contains invalid JSON") from error
        if not isinstance(payload, dict) or set(payload) != {"queries"}:
            raise ValueError("active retrieval response must contain only queries")
        queries = self._clean_queries(payload["queries"])
        if not queries:
            raise ValueError("bm25_search action must contain queries")
        return ActiveRetrievalDecision(
            "bm25_search",
            queries,
            "模型根据当前证据生成补充查询",
        )

    def _clean_queries(self, value: object) -> tuple[str, ...]:
        if not isinstance(value, list):
            return ()
        cleaned: list[str] = []
        for item in value:
            query = self._clean_string(item, max_length=100)
            if query and query not in cleaned:
                cleaned.append(query)
            if len(cleaned) >= self.settings.active_retrieval_max_queries:
                break
        return tuple(cleaned)

    @staticmethod
    def _clean_string(value: object, *, max_length: int) -> str:
        if not isinstance(value, str):
            return ""
        return " ".join(value.split())[:max_length].strip()
