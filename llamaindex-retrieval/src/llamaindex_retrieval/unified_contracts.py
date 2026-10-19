"""Versioned RWKV retrieval contracts. Validation is structural, never entailment."""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, model_validator

PROTOCOL = "rwkv-unified-v2"
Text = Annotated[str, StringConstraints(min_length=1, max_length=8000, pattern=r"\S")]
Channel = Literal["knowledge_base", "web"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class UnifiedPolicy(Contract):
    """One opt-in candidate, with resource limits rather than semantic switches."""
    reading_protocol: Literal["answer_v1", "span_ids_v1"] = "answer_v1"
    control_protocol: Literal["single_direction_v1", "review_cycle_v1"] = "single_direction_v1"
    plan_contract: Literal["static_v1", "bounded_v1"] = "static_v1"
    max_plan_revisions: int = Field(default=1, ge=0, le=2)
    max_rounds: int = Field(default=2, ge=1, le=4)
    max_actions: int = Field(default=6, ge=1, le=32)
    max_units: int = Field(default=24, ge=1, le=128)
    max_calls: int = Field(default=96, ge=2, le=512)
    stage_concurrency: int = Field(default=1, ge=1, le=32)
    timeout_seconds: float = Field(default=600.0, gt=0, le=3600, allow_inf_nan=False)
    unit_characters: int = Field(default=2400, ge=256, le=8000)
    control_tokens: int = Field(default=2048, ge=128, le=4096)
    leaf_tokens: int = Field(default=768, ge=64, le=4096)
    merge_tokens: int = Field(default=1024, ge=64, le=4096)

    @model_validator(mode="after")
    def compatible_control(self):
        if self.stage_concurrency > 1 and self.reading_protocol != "span_ids_v1":
            raise ValueError("concurrent_stages_require_span_ids_v1")
        if self.control_protocol == "review_cycle_v1" and self.reading_protocol != "span_ids_v1":
            raise ValueError("review_cycle_requires_span_ids_v1")
        if self.plan_contract == "bounded_v1" and self.control_protocol != "review_cycle_v1":
            raise ValueError("bounded_plan_requires_review_cycle_v1")
        return self


class MessageRef(Contract):
    message_id: Text
    quote: Text


class GroundedText(Contract):
    id: Text
    text: Text
    refs: list[MessageRef] = Field(min_length=1, max_length=16)


class Task(Contract):
    # Task interpretation is not a transport/capability decision. All item
    # categories share one shape; no duplicate question or mode/channels pair.
    goals: list[GroundedText] = Field(min_length=1, max_length=32)
    constraints: list[GroundedText] = Field(max_length=32)
    retracted: list[GroundedText] = Field(max_length=32)
    unresolved: list[Text] = Field(max_length=32)

    def check(self, messages):
        for group in (self.goals, self.constraints, self.retracted):
            unique([item.id for item in group])
        by_id = {message["id"]: message["content"] for message in messages}
        for item in [*self.goals, *self.constraints, *self.retracted]:
            for ref in item.refs:
                if ref.message_id not in by_id or ref.quote not in by_id[ref.message_id]:
                    raise ValueError("unbound_conversation_quote")
        if not any(ref.message_id == messages[-1]["id"]
                   for goal in self.goals for ref in goal.refs):
            raise ValueError("latest_request_not_referenced")
        # A current-turn quote is a structural anchor, not proof that an object,
        # retraction or condition has been understood correctly.


class Direction(Contract):
    choice: Literal["provided", "knowledge_base", "web", "hybrid", "clarify"]

    def check(self, allowed):
        if not set(self.channels()) <= set(allowed):
            raise ValueError("channel_not_permitted")

    def channels(self):
        return {"provided": [], "clarify": [], "knowledge_base": ["knowledge_base"],
                "web": ["web"], "hybrid": ["knowledge_base", "web"]}[self.choice]


class Action(Contract):
    channel: Channel
    query: Annotated[str, StringConstraints(min_length=1, max_length=2000, pattern=r"\S")]
    goal_ids: list[Text] = Field(min_length=1, max_length=32)


class Plan(Contract):
    actions: list[Action] = Field(max_length=32)
    unplanned_goal_ids: list[Text] = Field(max_length=32)

    def check(self, task, channels, remaining_actions, remaining_web, previous, *, exclusive=False):
        goals = {goal.id for goal in task.goals}
        unique(self.unplanned_goal_ids)
        covered = set(self.unplanned_goal_ids)
        signatures = []
        if len(self.actions) > remaining_actions:
            raise ValueError("action_budget_exceeded")
        if sum(action.channel == "web" for action in self.actions) > remaining_web:
            raise ValueError("web_budget_exceeded")
        for action in self.actions:
            unique(action.goal_ids)
            if action.channel not in channels or not set(action.goal_ids) <= goals:
                raise ValueError("action_outside_task")
            signature = (action.channel, action.query)
            if signature in previous:
                raise ValueError("repeated_action")
            signatures.append(signature)
            covered.update(action.goal_ids)
        unique(signatures)
        if exclusive and set(self.unplanned_goal_ids) & {g for a in self.actions for g in a.goal_ids}:
            raise ValueError("planned_and_unplanned_goal_ids")
        if covered != goals:
            raise ValueError("unaccounted_goal_ids")


class Reading(Contract):
    # This is model-authored derived text, never a verbatim-source representation.
    answer: Text
    evidence_ids: list[Text] = Field(max_length=128)
    missing: list[Text] = Field(max_length=32)

    def check(self, allowed_evidence):
        unique(self.evidence_ids)
        if not set(self.evidence_ids) <= set(allowed_evidence):
            raise ValueError("evidence_outside_node")


class Review(Contract):
    next: Literal["search", "stop"]
    missing: list[Text] = Field(max_length=32)
    reason: Text


def unique(values):
    if len(values) != len(set(values)):
        raise ValueError("duplicate_identity")


def digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def parse(raw, contract):
    """Reject fences, duplicate keys and coercion; retain raw separately."""
    def pairs(items):
        unique([key for key, _ in items])
        return dict(items)
    return contract.model_validate(json.loads(raw, object_pairs_hook=pairs), strict=True)


def parse_issues(error):
    """Safe, actionable diagnostics without copying raw values into errors."""
    if isinstance(error, ValidationError):
        return [{"location": list(item["loc"]), "type": item["type"]}
                for item in error.errors(include_input=False, include_context=False)]
    if isinstance(error, json.JSONDecodeError):
        return [{"type": "json_syntax", "line": error.lineno, "column": error.colno}]
    return [{"type": "duplicate_identity" if isinstance(error, ValueError)
             else "invalid_raw_type"}]


def task_input(messages):
    # Transport role/order only; never programmatically recover an object,
    # choose which condition was withdrawn, or rewrite the latest question.
    return {"history": messages[:-1], "latest_user_message": messages[-1]}


def reduction_calls(count):
    total = 0
    while count > 1:
        count = (count + 2) // 3
        total += count
    return total


def permitted_channels(mode):
    # API mode is an explicit capability restriction, not a semantic classifier.
    return {"knowledge_base": ["knowledge_base"], "web": ["web"],
            "hybrid": ["knowledge_base", "web"],
            "auto": ["knowledge_base", "web"], "materials": []}[mode]


_INSTRUCTIONS = {
    "task": "整理用户当前任务：goals列问题目标，constraints列用户要求，retracted列撤回内容，"
            "unresolved列理解任务所需的澄清。refs用消息ID和逐字原话标明依据。",
    "direction": "为当前任务选择资料方向：provided为给定材料，knowledge_base为知识库，"
                 "web为网络，hybrid为两者，clarify为澄清。",
    "plan": "根据任务、已执行动作和剩余预算生成本轮检索计划；"
            "动作标明goal_ids，未安排的目标列入unplanned_goal_ids。",
    "leaf": "依据本次原文回答任务相关的部分，保留资料ID和缺口。"
            "正文引用用[资料 N]对应E编号。",
    "review": "根据当前任务、阅读结果和剩余预算，决定继续检索search或停止stop，说明尚缺的信息。",
    "merge": "根据原始对话合并本组子输出，保留任务相关事实、已有资料ID和缺口。",
    "clarify": "就当前任务提出需要用户澄清的问题。",
}


_CYCLE_INSTRUCTIONS = {
    "plan_revision": "参考rejected_plan中的被拒计划及校验错误，重新生成本轮完整检索计划。"
                     + _INSTRUCTIONS["plan"],
}


def prompt(purpose, payload, contract=None, *, protocol=PROTOCOL, contract_schema=None):
    instruction = (_CYCLE_INSTRUCTIONS[purpose] if protocol == "rwkv-unified-v4"
                   and purpose in _CYCLE_INSTRUCTIONS else _INSTRUCTIONS[purpose])
    output = ("\n只输出符合以下结构的JSON，不要代码围栏：" + dumps(contract.model_json_schema() if contract_schema is None else contract_schema)
              if contract is not None else "\n直接输出澄清问题。")
    return (f"协议：{protocol}；用途：{purpose}\n{instruction}"
            + output + "\n输入数据：\n" + dumps(payload))
