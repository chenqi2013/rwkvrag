from types import SimpleNamespace
import pytest
from llamaindex_retrieval import typed_funnel_contract_v6 as c


def test_wire_aliases_preserve_false_and_disallow_nonphysical_unit_for_boolean():
    field = {"value_type": "boolean"}
    schema = c.atomic_schema(field, ["E1"])
    assert schema["properties"]["quantity_unit"] == {"type": "null"}
    wire = {"known_value": True, "evidence_ids": ["E1"], "quote": "甲不支持离线",
            "value": False, "quantity_unit": None, "source_scope": None}
    value = c.validate_atomic(c.Atomic.model_validate(wire), field, {"E1": SimpleNamespace(text=wire["quote"])})
    assert value["observed"] is True and value["value"] is False
    assert wire["value"] is False and "known_value" in wire  # parsing does not edit raw


def test_scope_is_source_bound_not_an_evidence_identifier():
    wire = {"known_value": True, "evidence_ids": ["E1"], "quote": "甲容量8L",
            "value": "8", "quantity_unit": "L", "source_scope": "E1"}
    with pytest.raises(ValueError):
        c.validate_atomic(c.Atomic.model_validate(wire), {"value_type": "quantity"},
            {"E1": SimpleNamespace(text=wire["quote"])})
