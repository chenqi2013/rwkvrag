import json
import re
from collections import OrderedDict
from dataclasses import dataclass, replace
from time import monotonic
from typing import Literal

import httpx

from .config import Settings
from .generation import EvidenceAnswerGenerator
from .lexical_index import normalize_search_text
from .query_planning import QueryPlan, TaskField, build_query_plan
from .qa_analysis import QuestionAnalysis


PlannerStrategy = Literal["model", "deterministic_fallback"]
_CACHE_MAX_ENTRIES = 512
@dataclass(frozen=True)
class QueryPlanningResult:
    plan: QueryPlan
    strategy: PlannerStrategy
    model_queries: tuple[str, ...] = ()
    error: str | None = None
    prompt: str = ""
    raw_output: str = ""


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
            (
                subject,
                intent,
                answer_shape,
                set_semantics,
                fields,
                relations,
                model_queries,
            ) = self._parse(raw)
            # A second, independent pass catches contracts that are valid JSON
            # but silently changed the user's requested field or answer shape.
            # The reviewer receives no documents and cannot answer the task;
            # it may only return a corrected contract.
            reviewed_raw = await self._request(
                self._review_prompt(question, raw)
            )
            (
                subject,
                intent,
                answer_shape,
                set_semantics,
                fields,
                relations,
                model_queries,
            ) = self._parse(reviewed_raw)
            raw = reviewed_raw
        except (httpx.HTTPError, TimeoutError, ValueError) as error:
            return QueryPlanningResult(
                fallback,
                "deterministic_fallback",
                error=f"{type(error).__name__}: {error}",
            )

        query_limit = self.settings.model_query_planning_max_queries
        # The model owns query semantics.  Deterministic fallback queries are
        # used only when the model call fails; they are never mixed into a
        # successful model plan because that reintroduces vocabulary-specific
        # behaviour and can drown out a relevant model query.
        query_candidates = (*model_queries, fallback.normalized_question)
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
        plan = replace(
            fallback,
            queries=tuple(queries[:query_limit]),
            subject=subject,
            relations=relations,
            analysis=replace(
                fallback.analysis,
                intent=intent,
                subjects=(),
                expects_list=answer_shape == "list",
                expects_complete_list=answer_shape == "list" and set_semantics == "all",
            ),
            merge_strategy=(
                "document_interleave"
                if intent in {"comparison", "time"}
                else "rank_fusion"
            ),
            context_policy={
                "cause": "section",
                "procedure": "section",
                "list": "structure",
                "definition": "lead",
                "comparison": "lead_append",
            }.get(intent, "none"),
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

    async def plan_immutable(self, question: str) -> QueryPlanningResult:
        prompt = self._prompt(question)
        try:
            fallback = build_query_plan(question)
            raw = await self._request(prompt)
            (
                subject,
                intent,
                answer_shape,
                set_semantics,
                fields,
                relations,
                model_queries,
            ) = self._parse(raw)
            fields = fields or fallback.fields
            queries = tuple(dict.fromkeys((
                *model_queries,
                subject,
                question,
            )))[
                : self.settings.model_query_planning_max_queries
            ]
            plan = QueryPlan(
                original_question=question,
                normalized_question=normalize_search_text(question),
                analysis=QuestionAnalysis(
                    intent=intent,
                    entity_type="unknown",
                    subjects=(),
                    expects_list=answer_shape == "list",
                    expects_complete_list=(
                        answer_shape == "list" and set_semantics == "all"
                    ),
                ),
                queries=queries,
                subject=subject,
                relations=relations,
                merge_strategy="rank_fusion",
                context_policy="none",
                fields=fields,
                answer_shape=answer_shape,
                set_semantics=set_semantics,
            )
            return QueryPlanningResult(
                plan=plan,
                strategy="model",
                model_queries=model_queries,
                prompt=prompt,
                raw_output=raw,
            )
        except (httpx.HTTPError, TimeoutError, ValueError) as error:
            fallback = build_query_plan(question)
            return QueryPlanningResult(
                plan=fallback,
                strategy="deterministic_fallback",
                error=f"{type(error).__name__}: {error}",
                prompt=prompt,
                raw_output=locals().get("raw", ""),
            )

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
        return f"""你是中文知识库的 BM25 查询规划器。你的任务不是回答问题，而是把原问题转换为可检索的任务契约。
先准确理解用户要查的对象、字段和答案形状，再生成多组彼此互补的查询。查询只能是原问题的忠实改写：保留原问题中的对象、范围、数量和关系，不添加原问题没有表达的答案、子类型、起点、终点、时间或其他事实。不能用猜测替换用户的原始关系，也不能把一个集合问题缩小成单个字段。
查询应适合百科全文检索：保留专名，使用原问题中已有的术语和不改变含义的自然表达；对口语字段可改写为资料中更常见的百科标题术语（尤其是列表、表格、人物归属等字段），但不得添加事实或答案。至少一条查询保留原问题，至少一条查询使用完整对象与字段，另至少一条查询采用“对象 + 字段核心名词”的标题式短查询。对于带范围限定的对象，至少保留一条完整范围查询；不要只保留宽泛范围词，也不要删除范围限定。
如果问题要求列举多个对象、成员、站点、项目、原因或成就，answer_shape 必须为 list，并且 set_semantics 按问题要求选择 all、partial 或 specific；不要把它标成 single_fact。若问题只问一个值，才使用 single_fact。
subject 必须是问题中已经出现的待查实体对象，不能填写你猜测的答案，也不能把“创始人、作者、原因、时间、地点、站点”等所求字段并入对象。例如“某公司创始人是谁”的 subject 只能是“某公司”；“赤手空拳打死老虎的是谁”不能把人物姓名填入 subject。
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

    @staticmethod
    def _review_prompt(question: str, contract: str) -> str:
        return f"""你是查询契约审计器。只检查契约是否忠实表达原问题，不回答原问题。
如果契约已经忠实，原样输出；如果对象、字段、数量范围或答案形状被改变，修正后输出。
必须保留原问题要求的完整集合，不得把多值问题缩成单值；不得添加原问题没有表达的答案或事实。
只输出完整 JSON，不要解释：
原问题：{question}
候选契约：{contract[-6000:]}
JSON："""

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
                "field_id", "question",
            }.issubset(item):
                continue
            field_id = cls._clean_string(item["field_id"], max_length=24)
            question = cls._clean_string(item["question"], max_length=160)
            relations = cls._clean_string_list(
                item.get("relations"),
                max_items=8,
                max_length=32,
            )
            if not re.fullmatch(r"f[1-9]\d*", field_id) or field_id in seen_ids:
                continue
            if not question:
                continue
            seen_ids.add(field_id)
            fields.append(TaskField(field_id, question, relations))
        return tuple(fields)

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
