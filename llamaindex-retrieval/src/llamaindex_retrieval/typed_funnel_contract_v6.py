"""V6 wire names disambiguate source IDs, physical units and known values.

The normalized dependency graph retains v5's field names. Raw model output is
unchanged; aliases are format parsing, not semantic correction.
"""
from pydantic import Field
from .typed_funnel_contract import *  # same task, cell and qualification contracts
from . import typed_funnel_contract as v5


class Atomic(v5.Atomic):
    observed: bool = Field(alias="known_value")
    unit_ids: list[str] = Field(alias="evidence_ids", max_length=3)
    unit: str | None = Field(alias="quantity_unit", max_length=80)
    scope: str | None = Field(alias="source_scope", max_length=160)


def atomic_schema(field, unit_ids):
    schema = Atomic.model_json_schema()
    kind = "boolean" if field["value_type"] == "boolean" else "string"
    schema["properties"]["value"] = {"anyOf": [{"type": kind}, {"type": "null"}]}
    schema["properties"]["evidence_ids"]["items"] = {"type": "string", "enum": list(unit_ids)}
    if field["value_type"] != "quantity":
        schema["properties"]["quantity_unit"] = {"type": "null"}
    return schema


def validate_atomic(parsed, field, units):
    result = v5.validate_atomic(parsed, field, units)
    if parsed.scope is not None and (not parsed.scope.strip() or not any(
            parsed.scope in units[identity].text for identity in parsed.unit_ids)):
        raise ValueError("scope must be verbatim in selected source evidence")
    return result
