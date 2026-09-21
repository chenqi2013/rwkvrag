"""One nullable typed value; no second truth label competing with bool false.

Quantity parsing splits a model-selected verbatim scalar into number and unit.
It performs no arithmetic, unit conversion or field/entity inference.
"""
import re
from pydantic import Field
from .typed_funnel_contract import *
from . import typed_funnel_contract as base


class Atomic(base.Contract):
    evidence_ids: list[str] = Field(max_length=3)
    quote: str | None = Field(max_length=400)
    value: bool | str | None
    source_scope: str | None = Field(max_length=160)


def atomic_schema(field, unit_ids):
    schema = Atomic.model_json_schema()
    kind = "boolean" if field["value_type"] == "boolean" else "string"
    schema["properties"]["value"] = {"anyOf": [{"type": kind}, {"type": "null"}]}
    schema["properties"]["evidence_ids"]["items"] = {"type": "string", "enum": list(unit_ids)}
    return schema


def validate_atomic(parsed, field, units):
    base.unique_subset(parsed.evidence_ids, units)
    texts = [units[i].text for i in parsed.evidence_ids]
    if parsed.quote is not None and (not parsed.quote or not any(parsed.quote in text for text in texts)):
        raise ValueError("quote must be verbatim in selected evidence")
    if parsed.source_scope is not None and (not parsed.source_scope.strip() or not any(parsed.source_scope in text for text in texts)):
        raise ValueError("scope must be verbatim in selected evidence")
    if parsed.value is not None and (not parsed.quote or not parsed.evidence_ids):
        raise ValueError("a value requires source provenance")
    value, unit = parsed.value, None
    if value is not None:
        if field["value_type"] == "boolean":
            if type(value) is not bool:
                raise ValueError("boolean field requires boolean value")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError("text/quantity field requires nonempty string")
        elif field["value_type"] == "quantity":
            if value not in parsed.quote:
                raise ValueError("quantity must be a verbatim source scalar including its unit")
            match = re.fullmatch(r"([+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))\s*([^0-9\s].*)?", value)
            if not match:
                raise ValueError("invalid decimal scalar representation")
            value, unit = match.group(1), match.group(2)
    return {"observed": parsed.value is not None, "unit_ids": parsed.evidence_ids,
            "quote": parsed.quote, "value": value, "unit": unit, "scope": parsed.source_scope}
