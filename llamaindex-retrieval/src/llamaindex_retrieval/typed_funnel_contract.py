"""Typed, source-bound funnel contracts. Validation never supplies an answer.

Models own field selection, negation, requirement revisions and eligibility.
Python checks types, provenance and the completeness of dependency sets only.
"""
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Dimension(Contract):
    name: str = Field(min_length=1, max_length=80)
    question: str = Field(min_length=1, max_length=240)
    value_type: Literal["boolean", "quantity", "text"]


class Requirement(Contract):
    message_id: str
    quote: str = Field(min_length=1, max_length=400)
    meaning: str = Field(min_length=1, max_length=240)
    state: Literal["active", "withdrawn"]
    kind: Literal["hard", "preference"]
    field_names: list[str] = Field(max_length=8)


class Task(Contract):
    mode: Literal["fact", "comparison", "selection"]
    objects: list[str] = Field(min_length=1, max_length=8)
    fields: list[Dimension] = Field(min_length=1, max_length=8)
    requirements: list[Requirement] = Field(max_length=12)
    requested_count: int | None = Field(ge=1, le=8)


def unique_subset(values, allowed, *, complete=False):
    if len(values) != len(set(values)) or not set(values) <= set(allowed):
        raise ValueError("duplicate or foreign dependency")
    if complete and set(values) != set(allowed):
        raise ValueError("missing dependency")


def validate_task(task, user_messages):
    unique_subset(task.objects, task.objects)
    if any(not name.strip() for name in task.objects):
        raise ValueError("blank object")
    names = [field.name for field in task.fields]
    unique_subset(names, names)
    for field in task.fields:
        if not field.name.strip() or not field.question.strip():
            raise ValueError("blank field")
    for requirement in task.requirements:
        if (requirement.message_id not in user_messages or
                requirement.quote not in user_messages[requirement.message_id]):
            raise ValueError("requirement must quote a user message verbatim")
        unique_subset(requirement.field_names, names)
        if requirement.state == "active" and not requirement.field_names:
            raise ValueError("active condition needs a concrete field")
    return task.model_dump()


class SourceObjects(Contract):
    objects: list[str] = Field(max_length=8)


class Atomic(Contract):
    # Explicit presence distinguishes an observed false/zero from absent data.
    observed: bool
    unit_ids: list[str] = Field(max_length=3)
    quote: str | None = Field(max_length=400)
    value: bool | str | None
    unit: str | None = Field(max_length=80)
    scope: str | None = Field(max_length=160)


def atomic_schema(field, unit_ids):
    schema = Atomic.model_json_schema()
    value_type = "boolean" if field["value_type"] == "boolean" else "string"
    schema["properties"]["value"] = {"anyOf": [{"type": value_type}, {"type": "null"}]}
    schema["properties"]["unit_ids"]["items"] = {"type": "string", "enum": list(unit_ids)}
    return schema


def validate_atomic(parsed, field, units):
    unique_subset(parsed.unit_ids, units)
    if not parsed.observed:
        if parsed.unit_ids or any(value is not None for value in
                (parsed.quote, parsed.value, parsed.unit, parsed.scope)):
            raise ValueError("unobserved result must not carry a fact")
        return parsed.model_dump()
    if not parsed.unit_ids or not parsed.quote or parsed.value is None:
        raise ValueError("observed fact needs value and exact source provenance")
    texts = [units[identity].text for identity in parsed.unit_ids]
    if not any(parsed.quote in text for text in texts):
        raise ValueError("fact quote is not verbatim in a selected unit")
    kind = field["value_type"]
    if kind == "boolean":
        if type(parsed.value) is not bool or parsed.unit is not None:
            raise ValueError("boolean field requires a boolean and no physical unit")
    else:
        if not isinstance(parsed.value, str) or not parsed.value.strip():
            raise ValueError("text/quantity field requires nonempty string")
        if kind == "quantity":
            try:
                number = Decimal(parsed.value)
            except InvalidOperation as error:
                raise ValueError("quantity must be a decimal without the unit") from error
            if not number.is_finite():
                raise ValueError("nonfinite quantity")
            if parsed.value not in parsed.quote:
                raise ValueError("quantity must preserve the source number; no conversion")
            if parsed.unit is not None and (not parsed.unit or parsed.unit not in parsed.quote):
                raise ValueError("unit must be verbatim in the source quote")
        elif parsed.unit is not None:
            raise ValueError("text field must not carry a separate numeric unit")
    return parsed.model_dump()


class Verification(Contract):
    verdict: Literal["supported", "mismatch", "uncertain"]
    explanation: str = Field(min_length=1, max_length=240)


class Cell(Contract):
    status: Literal["supported", "unknown", "conflict"]
    fact_ids: list[str] = Field(max_length=12)
    explanation: str = Field(min_length=1, max_length=320)


def validate_cell(cell, facts):
    unique_subset(cell.fact_ids, [fact["id"] for fact in facts])
    if cell.status == "supported" and not cell.fact_ids:
        raise ValueError("supported cell requires selected facts")
    if cell.status == "conflict" and len(cell.fact_ids) < 2:
        raise ValueError("conflict requires multiple observations")
    return cell.model_dump()


class Judgment(Contract):
    status: Literal["satisfied", "not_satisfied", "unknown", "conflict"]
    cell_ids: list[str] = Field(max_length=8)
    explanation: str = Field(min_length=1, max_length=320)


class Candidate(Contract):
    status: Literal["eligible", "ineligible", "unresolved", "not_applicable"]
    judgment_ids: list[str] = Field(max_length=12)
    explanation: str = Field(min_length=1, max_length=320)


def validate_candidate(candidate, judgments):
    unique_subset(candidate.judgment_ids, [j["id"] for j in judgments], complete=True)
    statuses = {j.get("status") for j in judgments}
    if candidate.status == "eligible" and (not judgments or statuses != {"satisfied"}):
        raise ValueError("eligible requires all active hard conditions satisfied")
    if candidate.status == "ineligible" and "not_satisfied" not in statuses:
        raise ValueError("ineligible requires a failed hard condition")
    if candidate.status == "not_applicable" and judgments:
        raise ValueError("hard conditions require an eligibility judgment")
    if candidate.status == "unresolved" and (not judgments or statuses == {"satisfied"} or "not_satisfied" in statuses):
        raise ValueError("unresolved contradicts hard-condition labels")
    return candidate.model_dump()


class Relation(Contract):
    cell_ids: list[str] = Field(min_length=1, max_length=8)
    summary: str = Field(min_length=1, max_length=600)


class Decision(Contract):
    recommended_ids: list[str] = Field(max_length=8)
    explanation: str = Field(min_length=1, max_length=600)


def validate_decision(decision, candidates, requested_count):
    # These labels were produced by the qualification model, not inferred here.
    allowed = [c["object_id"] for c in candidates if c.get("status") in {"eligible", "not_applicable"}
               and c.get("execution_status") == "completed"]
    unique_subset(decision.recommended_ids, allowed)
    if requested_count is not None and len(decision.recommended_ids) > requested_count:
        raise ValueError("recommendation exceeds requested count")
    return decision.model_dump()
