from types import SimpleNamespace
import pytest
from llamaindex_retrieval import typed_funnel_contract_v7 as c


@pytest.mark.parametrize("raw,kind,number,unit", [(False,"boolean",False,None), ("0 次","quantity","0","次"), ("11 L/min","quantity","11","L/min"), (None,"quantity",None,None)])
def test_single_value_preserves_negation_zero_unit_and_missing_provenance(raw,kind,number,unit):
    quote = "不支持离线；故障0 次；流量11 L/min；停机时长未记载"
    wire = {"evidence_ids":["E1"],"quote":quote,"value":raw,"source_scope":None}
    parsed = c.validate_atomic(c.Atomic.model_validate(wire), {"value_type":kind}, {"E1":SimpleNamespace(text=quote)})
    assert parsed["value"] == number and parsed["unit"] == unit
    assert parsed["observed"] is (raw is not None)
    assert parsed["quote"] == quote
    assert wire["value"] == raw
