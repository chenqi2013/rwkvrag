"""Plan the user task in small independent calls before reading source facts."""
import asyncio
import json
from typing import Annotated, Literal
from pydantic import Field
from . import typed_funnel_contract as c

Short = Annotated[str, Field(min_length=1, max_length=80)]


class Overview(c.Contract):
    mode: Literal["fact", "comparison", "selection"]
    objects: list[Short] = Field(min_length=1, max_length=8)
    fields: list[Short] = Field(min_length=1, max_length=8)
    requested_count: int | None = Field(ge=1, le=8)


class ValueType(c.Contract):
    value_type: Literal["boolean", "quantity", "text"]


class UserQuote(c.Contract):
    message_id: str
    quote: str = Field(min_length=1, max_length=400)


class Quotes(c.Contract):
    quotes: list[UserQuote] = Field(max_length=12)


class Meaning(c.Contract):
    state: Literal["active", "withdrawn"]
    kind: Literal["hard", "preference"]
    meaning: str = Field(min_length=1, max_length=240)
    field_names: list[str] = Field(max_length=8)
    lifecycle_message_id: str
    lifecycle_quote: str = Field(min_length=1, max_length=400)


def dump(value):
    return json.dumps(value, ensure_ascii=False)


async def build_task(runner, messages):
    def overview_check(parsed):
        c.unique_subset(parsed.objects, parsed.objects)
        c.unique_subset(parsed.fields, parsed.fields)
        return parsed.model_dump()
    rendered = "\n\n".join(f"第{i}条用户发言：{text}" for i, text in enumerate(messages.values(), 1))
    overview = await runner.node(
        "根据用户发言整理最新任务，只输出JSON。objects填要研究的真实对象名称，"
        "消息序号不是研究对象。fields填所有对象共用的属性名称，不包含任何对象名，"
        "每项只是一种具体属性，不写答案、不写比较结论、不重复。"
        "仅保留用户仍需要的属性；没有指定比较维度时提出少量具体属性，不为填满数组而增加。"
        "历史中未撤回的目标和约束继续有效，最新取消的要求不恢复。"
        "mode为查事实fact、比较comparison或推荐选型selection。"
        "requested_count是明确要求推荐的数量，没有指定则null。\n\n"
        + rendered,
        "funnel_task_overview", Overview, validator=overview_check, max_tokens=768)
    if overview is None:
        return None

    async def classify(question):
        result = await runner.node(
            "给这一个属性问题选择答案类型，只输出value_type。明确是/否的属性为boolean，"
            "单一数值为quantity，日期、区间、多值或描述为text。不回答问题，不判断用户条件是否满足。\n"
            + dump({"question": question}), "funnel_field_type:" + question, ValueType, max_tokens=48)
        return {"name": question, "question": question, **result} if result else None
    dimensions = await asyncio.gather(*(classify(question) for question in overview["fields"]))
    if any(item is None for item in dimensions):
        return None

    def quotes_check(parsed):
        for quote in parsed.quotes:
            if quote.message_id not in messages or quote.quote not in messages[quote.message_id]:
                raise ValueError("condition origin must quote a user message")
        c.unique_subset([(q.message_id, q.quote) for q in parsed.quotes],
                        [(q.message_id, q.quote) for q in parsed.quotes])
        return parsed.model_dump()
    quoted = await runner.node(
        "从用户消息中逐字摘录选型的具体限制、偏好及取消/修改要求的语句。"
        "每段只摘一个独立条件并保留否定、数值和限定，不把整道多条件问题作为一个条件。"
        "不把要求引用、比较动作或回答格式当作产品必须具备的功能。"
        "每段保留message_id和quote；没有选型条件则quotes为空。不解释，不猜新增条件。\n"
        + dump(messages), "funnel_requirement_quotes", Quotes, validator=quotes_check, max_tokens=1024)
    if quoted is None:
        return None

    async def interpret(origin):
        def validate(parsed):
            c.unique_subset(parsed.field_names, overview["fields"])
            if parsed.state == "active" and not parsed.field_names:
                raise ValueError("active condition not covered by planned fields")
            if (parsed.lifecycle_message_id not in messages or
                    parsed.lifecycle_quote not in messages[parsed.lifecycle_message_id]):
                raise ValueError("condition lifecycle must quote a user message")
            return parsed.model_dump()
        schema = Meaning.model_json_schema()
        schema["properties"]["field_names"]["items"] = {"type": "string", "enum": overview["fields"]}
        result = await runner.node(
            "只解释这一段用户条件，并依据全部用户消息判断它现在是否生效。"
            "active为仍生效，withdrawn为取消或暂缓。取消功能要求不等于禁止产品具备该功能。"
            "kind区分必须满足hard和优先考虑preference。meaning保留条件原意和数值/否定。"
            "field_names选择该条件对应的已计划属性，撤回项可为空。"
            "lifecycle_message_id及lifecycle_quote逐字引用决定当前生效状态的用户原话。\n"
            + dump({"condition_origin": origin, "all_user_messages": messages, "fields": overview["fields"]}),
            "funnel_requirement:" + origin["message_id"] + ":" + origin["quote"], Meaning,
            validator=validate, document=schema, max_tokens=512)
        return {**origin, **result} if result else None
    requirements = await asyncio.gather(*(interpret(origin) for origin in quoted["quotes"]))
    if any(item is None for item in requirements):
        return None
    return {"mode": overview["mode"], "objects": overview["objects"], "fields": dimensions,
            "requirements": requirements, "requested_count": overview["requested_count"]}
