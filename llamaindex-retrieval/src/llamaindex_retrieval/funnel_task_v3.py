"""Separate intent and literal user clauses before choosing fact dimensions."""
import asyncio
import re
from typing import Literal
from pydantic import Field
from . import typed_funnel_contract as c
from .funnel_task import Short, ValueType


class Overview(c.Contract):
    objects: list[Short] = Field(min_length=1, max_length=8)
    fields: list[Short] = Field(min_length=1, max_length=8)
    requested_count: int | None = Field(ge=1, le=8)


class Intent(c.Contract):
    mode: Literal['fact', 'comparison', 'selection']


class Clause(c.Contract):
    role: Literal['hard', 'preference', 'context']
    state: Literal['active', 'withdrawn']
    meaning: str = Field(min_length=1, max_length=240)
    field_question: str | None = Field(max_length=80)


def user_clauses(messages):
    """Punctuation boundaries and exact offsets only; no semantic keyword split."""
    for identity, text in messages.items():
        for match in re.finditer(r'[^，；;。！？!?、\n]+[，；;。！？!?、\n]*', text):
            quote = match.group().strip()
            if quote:
                start = match.start() + len(match.group()) - len(match.group().lstrip())
                yield {'message_id': identity, 'quote': quote, 'start': start, 'end': start + len(quote)}


async def build_task(runner, messages):
    text = '\n\n'.join(f'第{i}条用户发言：{value}' for i, value in enumerate(messages.values(), 1))
    intent = await runner.node(
        '只判断用户当前需要哪种回答，输出mode：查具体事实为fact；只列区别为comparison；'
        '问哪个好、如何选择或推荐先试对象为selection。以最新问题及仍有效的历史为准。\n' + text,
        'funnel_intent', Intent, max_tokens=48)
    if intent is None:
        return None
    def validate_overview(parsed):
        c.unique_subset(parsed.objects, parsed.objects)
        c.unique_subset(parsed.fields, parsed.fields)
        return parsed.model_dump()
    overview = await runner.node(
        '列出当前仍需研究的对象和属性，输出JSON。objects只填真实对象名，消息序号不是对象。'
        'fields填所有对象共用的具体属性名，不包含对象名，不填答案或排名，不重复。'
        '保留用户原有仍有效的属性；未指定时只提出少量具体比较属性，不为填满数组扩展。'
        'requested_count填用户明确要求的推荐数量，未指定用null。\n' + text,
        'funnel_task_overview', Overview, validator=validate_overview, max_tokens=768)
    if overview is None:
        return None

    async def interpret(origin):
        def validate(parsed):
            if parsed.role != 'context' and parsed.state == 'active' and not parsed.field_question:
                raise ValueError('active condition requires a concrete attribute question')
            return parsed.model_dump()
        result = await runner.node(
            '只解释下面这一段用户原话，不将其他段的要求混进来。完整发言仅用于指代和后续更正。'
            'role：明确必须满足的条件为hard，优先考虑为preference，其余为context。'
            '列举对象、询问事实、比较动作、要求引用或格式、笼统项目目标都是context，不是产品硬指标。'
            'state：仍有效为active，后来取消或暂缓为withdrawn。取消功能需求不代表禁止产品有该功能。'
            'meaning只解释本段原意；field_question只提一个对象可回答的具体属性，不包含对象名。'
            'context的field_question为null。不要自创或反转用户的肯定/否定。\n'
            '本段原话：' + origin['quote'] + '\n完整用户发言：\n' + text,
            f"funnel_requirement_clause:{origin['message_id']}:{origin['start']}", Clause,
            validator=validate, max_tokens=256)
        return {**origin, **result} if result else None
    clauses = await asyncio.gather(*(interpret(origin) for origin in user_clauses(messages)))
    if any(clause is None for clause in clauses):
        return None
    requirements = [{'message_id': row['message_id'], 'quote': row['quote'], 'meaning': row['meaning'],
                     'state': row['state'], 'kind': row['role'], 'field_names': [row['field_question']] if row['field_question'] else [],
                     'origin_span': {'start': row['start'], 'end': row['end']}}
                    for row in clauses if row['role'] != 'context']
    # Concrete condition questions come from the clause models. General
    # comparison dimensions are used when no active concrete condition exists.
    questions = list(dict.fromkeys(row['field_names'][0] for row in requirements
        if row['state'] == 'active' and row['field_names'])) or overview['fields']
    if len(questions) > 8:
        runner.failures.append({'purpose': 'funnel_task_dimensions', 'status': 'dimension_budget_exceeded',
                                'requested_dimensions': questions})
        return None
    async def classify(question):
        value = await runner.node(
            '只给这个属性选择答案类型value_type。明确是/否为boolean，单一数值为quantity，'
            '日期、区间、多值、描述为text。不回答问题，也不判断是否满足用户要求。\n属性：' + question,
            'funnel_field_type:' + question, ValueType, max_tokens=48)
        return {'name': question, 'question': question, **value} if value else None
    fields = await asyncio.gather(*(classify(question) for question in questions))
    if any(field is None for field in fields):
        return None
    return {'mode': intent['mode'], 'objects': overview['objects'], 'fields': fields,
            'requirements': requirements, 'requested_count': overview['requested_count'],
            'user_clause_analysis': clauses}
