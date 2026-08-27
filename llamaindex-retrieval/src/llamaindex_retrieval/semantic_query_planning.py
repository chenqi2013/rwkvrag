import json
import re
from collections import OrderedDict
from dataclasses import dataclass, replace
from time import monotonic
from typing import Literal

import httpx

from .config import Settings
from .generation import EvidenceAnswerGenerator
from .lexical_index import lexical_tokens, normalize_search_text
from .query_planning import QueryPlan, TaskField
from .qa_analysis import counted_list_size


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
        # Preserve deterministic semantic contracts for intents where changing
        # the object or relation is more damaging than missing a model query.
        # The model may still provide extra queries, but it must not turn a
        # cause/comparison/structured explanation into a generic fact lookup.
        if fallback.analysis.intent in {"cause", "comparison", "procedure"} or (
            fallback.analysis.intent == "definition"
            and fallback.answer_shape == "summary"
        ):
            return QueryPlanningResult(
                fallback,
                "deterministic_fallback",
                error="deterministic_semantic_contract",
            )
        if self._fallback_contract_is_sufficient(fallback):
            return QueryPlanningResult(
                fallback,
                "deterministic_fallback",
                error="structured_fallback_sufficient",
            )
        cached = self._get_cached(question)
        if cached is not None:
            return cached
        try:
            raw = await self._request(self._prompt(question))
            (
                subject,
                intent,
                answer_shape,
                set_semantics,
                fields,
                relations,
                model_queries,
            ) = self._parse(raw)
        except (httpx.HTTPError, TimeoutError, ValueError) as error:
            return QueryPlanningResult(
                fallback,
                "deterministic_fallback",
                error=f"{type(error).__name__}: {error}",
            )

        if not self._contract_subject_is_grounded(question, fallback, subject):
            grounded_subject = self._ground_subject_from_contract(
                question,
                fallback,
                fields,
                model_queries,
            )
            if grounded_subject:
                invalid_subject = normalize_search_text(subject).replace(" ", "")
                relations = tuple(
                    relation
                    for relation in relations
                    if normalize_search_text(relation).replace(" ", "")
                    != invalid_subject
                )
                fields = tuple(
                    replace(
                        field,
                        relations=tuple(
                            relation
                            for relation in field.relations
                            if normalize_search_text(relation).replace(" ", "")
                            != invalid_subject
                        ),
                    )
                    for field in fields
                )
                if relations and all(field.relations for field in fields):
                    subject = grounded_subject

        if not self._contract_subject_is_grounded(question, fallback, subject):
            try:
                repaired_raw = await self._request(
                    self._repair_prompt(question, raw)
                )
                (
                    subject,
                    intent,
                    answer_shape,
                    set_semantics,
                    fields,
                    relations,
                    model_queries,
                ) = self._parse(repaired_raw)
            except (httpx.HTTPError, TimeoutError, ValueError):
                pass

        query_limit = self.settings.model_query_planning_max_queries
        if self._preserve_explicit_list_contract(fallback):
            query_candidates = (*fallback.queries, *model_queries)
        else:
            model_prefix_size = max(1, query_limit // 2)
            query_candidates = (
                *model_queries[:model_prefix_size],
                *fallback.queries,
                *model_queries[model_prefix_size:],
            )
        queries: list[str] = []
        for query in query_candidates:
            if query not in queries:
                queries.append(query)
            if len(queries) >= query_limit:
                break
        if fallback.normalized_question not in queries:
            if len(queries) >= query_limit:
                queries[-1] = fallback.normalized_question
            else:
                queries.append(fallback.normalized_question)
        if not self._contract_subject_is_grounded(question, fallback, subject):
            return QueryPlanningResult(
                replace(fallback, queries=tuple(queries[:query_limit])),
                "deterministic_fallback",
                model_queries=model_queries,
                error="model_subject_not_grounded_in_question",
            )
        if fallback.analysis.expects_list:
            intent = "list"
            answer_shape = "list"
            if fallback.set_semantics == "all":
                set_semantics = "all"
        if fallback.answer_shape in {"summary", "narrative"}:
            answer_shape = fallback.answer_shape
            set_semantics = fallback.set_semantics
        if fallback.analysis.intent == "agent" and fallback.subject:
            subject = fallback.subject
            relations = fallback.relations
            fields = fallback.fields
            intent = "agent"
            answer_shape = "single_fact"
            set_semantics = fallback.set_semantics
        if (
            fallback.analysis.intent == "definition"
            and fallback.answer_shape == "summary"
        ):
            subject = fallback.subject
            relations = fallback.relations
            fields = fallback.fields
            intent = "definition"
            answer_shape = "summary"
            set_semantics = fallback.set_semantics
        if fallback.analysis.expects_list:
            intent = "list"
            answer_shape = "list"
            set_semantics = fallback.set_semantics
            if self._subject_matches_fallback(fallback.subject, subject):
                subject = fallback.subject
                relations = fallback.relations
                fields = fallback.fields
        plan = replace(
            fallback,
            queries=tuple(queries[:query_limit]),
            subject=subject,
            relations=relations,
            analysis=replace(
                fallback.analysis,
                intent=intent,
                expects_list=answer_shape == "list",
                expects_complete_list=answer_shape == "list" and set_semantics == "all",
            ),
            fields=fields,
            answer_shape=answer_shape,
            set_semantics=set_semantics,
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
请先把问题拆成一个可追踪的任务契约，再生成不同检索角度的关键词组合。
查询应适合百科全文检索：保留专名，使用原问题可能对应的百科标题、关系词和常见同义表达。可以把可能的人物、事件结果或标准术语作为多种“检索假设”写入 queries，但不能把这些假设写进 subject；后端会用原文验证假设。
subject 必须是问题中已经出现的待查对象，不能填写你猜测的答案。例如“赤手空拳打死老虎的是谁”不能把人物姓名填入 subject。
如果问题询问“是谁”且描述的是一个事件，queries 中至少一条必须包含你推测的具体人物姓名及其典型事件关键词；不能全部只重复“人物、英雄、主角、人名”等抽象词。该人物只作为待验证的检索假设。
intent 只能是 definition、fact、list、cause、time、location、birthplace、agent、ordinal、comparison、procedure。
answer_shape 只能是 single_fact、list、summary、narrative；set_semantics 只能是 latest、all、partial、specific。
fields 固定至少一项；field_id 从 f1 开始，question 写该字段具体要求，relations 写资料中可能出现的 2 到 6 个简短同义表达。
顶层 relations 与 fields[0].relations 保持一致，每项只写关系短语，不写对象或答案。
输出 3 到 6 条互补查询，短而具体，不要输出完整解释。

只输出一个 JSON 对象，七个字段全部必填，格式固定：
{{"subject":"问题中出现的核心对象","intent":"fact","answer_shape":"single_fact","set_semantics":"specific","fields":[{{"field_id":"f1","question":"要取的具体值","relations":["原关系","同义表达"]}}],"relations":["原关系","同义表达"],"queries":["查询1","查询2","查询3"]}}

问题：{question}
JSON："""

    @staticmethod
    def _repair_prompt(question: str, invalid_output: str) -> str:
        return f"""你要修复一个知识库查询契约。上一次输出把待查询答案误写进了 subject，或改变了原问题中的对象。
subject 必须逐字取自原问题中已经出现的待查对象；不能填写你猜测的人名、地点、时间或其他答案。
relations 只写关系名称及同义表达，不能填写猜测的具体答案。
保留原问题真实意图，重新输出完整 JSON；七个字段全部必填，不要解释：
{{"subject":"原问题中的对象","intent":"fact","answer_shape":"single_fact","set_semantics":"specific","fields":[{{"field_id":"f1","question":"要取的具体值","relations":["关系","同义关系"]}}],"relations":["关系","同义关系"],"queries":["查询1","查询2","查询3"]}}

原问题：{question}
无效输出：{invalid_output[-2000:]}
修复后的 JSON："""

    @classmethod
    def _parse(
        cls,
        raw: str,
    ) -> tuple[
        str,
        str,
        str,
        str,
        tuple[TaskField, ...],
        tuple[str, ...],
        tuple[str, ...],
    ]:
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
        required = {
            "subject", "intent", "answer_shape", "set_semantics",
            "fields", "queries",
        }
        if not required.issubset(payload):
            raise ValueError("planner response does not match the task contract")

        subject = cls._clean_string(payload["subject"], max_length=80)
        intent = cls._clean_enum(
            payload["intent"],
            {
                "definition", "fact", "list", "cause", "time", "location",
                "birthplace", "agent", "ordinal", "comparison", "procedure",
            },
        )
        answer_shape = cls._clean_enum(
            payload["answer_shape"],
            {"single_fact", "list", "summary", "narrative"},
        )
        set_semantics = cls._clean_enum(
            payload["set_semantics"],
            {"latest", "all", "partial", "specific"},
        )
        fields = cls._clean_fields(payload["fields"])
        relations = cls._clean_string_list(
            payload.get("relations"),
            max_items=8,
            max_length=32,
        )
        if not relations:
            relations = tuple(dict.fromkeys(
                relation
                for field in fields
                for relation in field.relations
            ))
        queries = cls._clean_string_list(payload["queries"], max_items=6, max_length=100)
        if not subject:
            raise ValueError("planner subject must not be empty")
        if not fields:
            raise ValueError("planner fields must not be empty")
        if not relations:
            raise ValueError("planner relations must not be empty")
        if not queries:
            raise ValueError("planner must return at least one query")
        return subject, intent, answer_shape, set_semantics, fields, relations, queries

    @staticmethod
    def _clean_enum(value: object, allowed: set[str]) -> str:
        text = str(value).strip() if isinstance(value, str) else ""
        if text not in allowed:
            raise ValueError(f"unsupported planner enum: {text}")
        return text

    @classmethod
    def _clean_fields(cls, value: object) -> tuple[TaskField, ...]:
        if not isinstance(value, list):
            return ()
        fields: list[TaskField] = []
        seen_ids: set[str] = set()
        for item in value[:8]:
            if not isinstance(item, dict) or not {
                "field_id", "question", "relations",
            }.issubset(item):
                continue
            field_id = cls._clean_string(item["field_id"], max_length=24)
            question = cls._clean_string(item["question"], max_length=160)
            relations = cls._clean_string_list(item["relations"], max_items=8, max_length=32)
            if not re.fullmatch(r"f[1-9]\d*", field_id) or field_id in seen_ids:
                continue
            if not question or not relations:
                continue
            seen_ids.add(field_id)
            fields.append(TaskField(field_id, question, relations))
        return tuple(fields)

    @staticmethod
    def _subject_is_supported(question: str, subject: str) -> bool:
        normalized_question = normalize_search_text(question).replace(" ", "")
        normalized_subject = normalize_search_text(subject).replace(" ", "")
        if not normalized_subject:
            return False
        if normalized_subject in normalized_question:
            return True
        question_terms = {
            term for term in lexical_tokens(question)
            if len(term.strip()) >= 2
        }
        subject_terms = {
            term for term in lexical_tokens(subject)
            if len(term.strip()) >= 2
        }
        return bool(question_terms & subject_terms)

    @staticmethod
    def _subject_matches_fallback(fallback_subject: str, model_subject: str) -> bool:
        fallback = normalize_search_text(fallback_subject).replace(" ", "")
        model = normalize_search_text(model_subject).replace(" ", "")
        if not fallback:
            return True
        if not model:
            return False
        return fallback == model or (
            min(len(fallback), len(model)) >= 3
            and (fallback in model or model in fallback)
        )

    @classmethod
    def _contract_subject_is_grounded(
        cls,
        question: str,
        fallback: QueryPlan,
        model_subject: str,
    ) -> bool:
        return cls._subject_is_supported(question, model_subject) and not (
            fallback.analysis.intent == "agent"
            and not cls._subject_matches_fallback(fallback.subject, model_subject)
        )

    @staticmethod
    def _preserve_explicit_list_contract(plan: QueryPlan) -> bool:
        if not plan.analysis.expects_complete_list:
            return False
        if counted_list_size(plan.normalized_question) is not None:
            return True
        return bool(re.search(
            r"(?:是|为)(?:哪几个|哪几种|哪几类|哪几项|哪几篇|哪几部|哪几本)",
            plan.normalized_question,
        ))

    @classmethod
    def _ground_subject_from_contract(
        cls,
        question: str,
        fallback: QueryPlan,
        fields: tuple[TaskField, ...],
        model_queries: tuple[str, ...],
    ) -> str:
        if fallback.subject and cls._subject_is_supported(question, fallback.subject):
            return fallback.subject
        contract_text = " ".join((
            *(field.question for field in fields),
            *model_queries,
        ))
        quoted: list[str] = []
        for match in re.finditer(
            r"《([^》]{1,80})》|“([^”]{1,80})”|\"([^\"]{1,80})\"",
            question,
        ):
            quoted.extend(
                value.strip()
                for value in match.groups()
                if value and value.strip()
            )
        candidates = [
            value
            for value in quoted
            if normalize_search_text(value).replace(" ", "")
            in normalize_search_text(contract_text).replace(" ", "")
        ]
        if candidates:
            return max(candidates, key=len)
        question_terms = {
            term.strip()
            for term in lexical_tokens(question)
            if len(term.strip()) >= 2
        }
        contract_terms = {
            term.strip()
            for term in lexical_tokens(contract_text)
            if len(term.strip()) >= 2
        }
        shared = question_terms & contract_terms
        return max(shared, key=len) if shared else ""

    @staticmethod
    def _fallback_contract_is_sufficient(plan: QueryPlan) -> bool:
        if not plan.subject or not plan.relations:
            return False
        if plan.answer_shape != "single_fact":
            return False
        if plan.relations == ("简介", "定义"):
            return len(plan.subject.replace(" ", "")) <= 10
        if plan.analysis.intent == "agent" and any(
            len(relation.replace(" ", "")) > 6
            for relation in plan.relations
        ):
            return False
        return plan.analysis.intent in {
            "definition", "time", "location", "birthplace", "agent", "ordinal",
        }

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
