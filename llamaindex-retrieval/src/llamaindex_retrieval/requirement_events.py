"""Model-authored requirement changes, distinct from whether an utterance is active."""
from typing import Literal
from pydantic import Field
from .typed_funnel_contract import Contract


class Event(Contract):
    action: Literal['必须满足', '优先考虑', '不再要求', '其他']
    requirement: str | None = Field(max_length=240)


class BooleanAnswer(Contract):
    answer: bool


async def interpret_event(runner, quote, *, purpose='funnel_requirement_event'):
    def validate(parsed):
        if parsed.action != '其他' and (parsed.requirement is None or not parsed.requirement.strip()):
            raise ValueError('requirement change must identify its content')
        return parsed.model_dump()
    return await runner.node(
        '判断这句话对产品选型条件做了什么变更，输出action和requirement。'
        'action：提出必须满足的具体能力/环境/数值限制为“必须满足”；'
        '提出优先考虑的具体属性为“优先考虑”；取消或暂缓功能需求为“不再要求”；'
        '仅列举产品、提出比较问题、说明笼统项目目标、要求回答格式为“其他”。'
        'requirement只写这句话的具体需求内容，保留否定和数值，不扩展；其他用null。'
        '取消需求不等于要求产品不能具备该功能。\n原话：' + quote,
        purpose, Event, validator=validate, max_tokens=256)


async def needs_selection(runner, messages):
    return await runner.node(
        '判断用户是否希望获得选型建议、推荐对象或推荐先试哪些，而不仅是事实或区别列表。'
        '只输出JSON的answer布尔值。\n用户发言：\n' + messages,
        'funnel_selection_intent', BooleanAnswer, max_tokens=24)
