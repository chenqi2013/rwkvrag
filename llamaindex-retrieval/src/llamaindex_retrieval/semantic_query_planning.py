import json
import re
from collections import OrderedDict
from dataclasses import dataclass, replace
from time import monotonic
from typing import Literal

import httpx

from .config import Settings
from .generation import EvidenceAnswerGenerator
from .query_planning import QueryPlan


PlannerStrategy = Literal["model", "deterministic_fallback"]
_CACHE_MAX_ENTRIES = 512


@dataclass(frozen=True)
class QueryPlanningResult:
    plan: QueryPlan
    strategy: PlannerStrategy
    model_queries: tuple[str, ...] = ()
    error: str | None = None


class LanguageModelQueryPlanner:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self._cache: OrderedDict[str, tuple[float, QueryPlanningResult]] = OrderedDict()

    async def plan(self, question: str, fallback: QueryPlan) -> QueryPlanningResult:
        if not self.settings.model_query_planning_enabled:
            return QueryPlanningResult(fallback, "deterministic_fallback", error="disabled")
        if not self.settings.generation_password:
            return QueryPlanningResult(
                fallback,
                "deterministic_fallback",
                error="generation_password_not_configured",
            )
        cached = self._get_cached(question)
        if cached is not None:
            return cached
        try:
            raw = await self._request(self._prompt(question))
            subject, relations, model_queries = self._parse(raw)
        except (httpx.HTTPError, TimeoutError, ValueError) as error:
            return QueryPlanningResult(
                fallback,
                "deterministic_fallback",
                error=f"{type(error).__name__}: {error}",
            )

        query_limit = self.settings.model_query_planning_max_queries
        queries = list(model_queries[: max(1, query_limit - 1)])
        if fallback.normalized_question not in queries:
            queries.append(fallback.normalized_question)
        plan = replace(
            fallback,
            queries=tuple(queries[:query_limit]),
            subject=subject,
            relations=relations,
        )
        result = QueryPlanningResult(
            plan,
            "model",
            model_queries=model_queries,
        )
        self._store_cached(question, result)
        return result

    def _get_cached(self, question: str) -> QueryPlanningResult | None:
        cached = self._cache.get(question)
        if cached is None:
            return None
        created_at, result = cached
        if monotonic() - created_at > self.settings.model_query_planning_cache_ttl:
            self._cache.pop(question, None)
            return None
        self._cache.move_to_end(question)
        return result

    def _store_cached(self, question: str, result: QueryPlanningResult) -> None:
        if self.settings.model_query_planning_cache_ttl <= 0:
            return
        self._cache[question] = (monotonic(), result)
        self._cache.move_to_end(question)
        while len(self._cache) > _CACHE_MAX_ENTRIES:
            self._cache.popitem(last=False)

    async def _request(self, prompt: str) -> str:
        payload = {
            "contents": [prompt],
            "max_tokens": self.settings.model_query_planning_max_tokens,
            "temperature": 0.2,
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
            self.settings.model_query_planning_timeout,
        )
        async with httpx.AsyncClient(timeout=timeout, transport=self.transport) as client:
            async with client.stream("POST", endpoint, json=payload) as response:
                response.raise_for_status()
                return await EvidenceAnswerGenerator._read_stream(
                    response,
                    total_timeout=self.settings.model_query_planning_timeout,
                )

    @staticmethod
    def _prompt(question: str) -> str:
        return f"""你是中文知识库的 BM25 查询规划器。你的任务不是回答问题，而是生成多组搜索关键词。
请从问题中识别唯一的核心对象、所求关系，并生成不同检索角度的关键词组合。
查询应适合百科全文检索：保留专名，使用原问题可能对应的百科标题、关系词和常见同义表达；不要把你猜测的答案写入查询。
输出 3 到 6 条互补查询，短而具体，不要输出完整解释。

只输出一个 JSON 对象，三个字段全部必填，格式固定：
{{"subject":"核心对象","relations":["关系词"],"queries":["查询1","查询2","查询3"]}}

问题：{question}
JSON："""

    @classmethod
    def _parse(cls, raw: str) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "")
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("planner response does not contain a JSON object")
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as error:
            raise ValueError("planner response contains invalid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("planner response must be a JSON object")
        if set(payload) != {"subject", "relations", "queries"}:
            raise ValueError("planner response must contain subject, relations and queries")

        subject = cls._clean_string(payload["subject"], max_length=80)
        relations = cls._clean_string_list(payload["relations"], max_items=8, max_length=32)
        queries = cls._clean_string_list(payload["queries"], max_items=6, max_length=100)
        if not subject:
            raise ValueError("planner subject must not be empty")
        if not relations:
            raise ValueError("planner relations must not be empty")
        if len(queries) < 3:
            raise ValueError("planner must return at least three queries")
        return subject, relations, queries

    @staticmethod
    def _clean_string(value: object, *, max_length: int) -> str:
        if not isinstance(value, str):
            return ""
        return " ".join(value.split())[:max_length].strip()

    @classmethod
    def _clean_string_list(
        cls,
        value: object,
        *,
        max_items: int,
        max_length: int,
    ) -> tuple[str, ...]:
        if not isinstance(value, list):
            return ()
        cleaned: list[str] = []
        for item in value:
            text = cls._clean_string(item, max_length=max_length)
            if text and text not in cleaned:
                cleaned.append(text)
            if len(cleaned) >= max_items:
                break
        return tuple(cleaned)
