import copy
import json
from hashlib import sha256

import pytest

from llamaindex_retrieval.comparison_contract import parse_comparison


def record():
    text = "甲标准模式10小时；乙标准模式12小时。"
    h = sha256(text.encode()).hexdigest()
    return {
        "protocol": "comparison-contract-v1",
        "snapshots": [{"id": "S1", "kind": "source", "version": "1", "text": text, "sha256": h}],
        "object_ids": ["a", "b"],
        "facts": [{"id": "F" + obj, "object_id": obj, "field": "续航", "scope": "标准模式",
                   "claim": claim, "assessment": "supported",
                   "source_spans": [{"snapshot_id": "S1", "snapshot_sha256": h, "start": 0, "end": len(text)}]}
                  for obj, claim in [("a", "10小时"), ("b", "12小时")]],
        "relations": [{"id": "R1", "left_object_id": "a", "right_object_id": "b", "field": "续航",
                       "scope": "标准模式", "left_fact_ids": ["Fa"], "right_fact_ids": ["Fb"],
                       "status": "less", "explanation": "甲较短"}],
        "resolver_selected_ids": ["S1"], "writer_source_ids": ["S1"],
        "writer_relation_ids": ["R1"], "raw_answer": "甲较短[S1]",
    }


def validate(data):
    return parse_comparison(json.dumps(data))


def test_valid_and_immutable():
    d = record()
    before = copy.deepcopy(d)
    assert validate(d).raw_answer == d["raw_answer"]
    assert d == before


@pytest.mark.parametrize("mutation", [
    lambda d: d["relations"][0].update(right_fact_ids=[]),
    lambda d: d["relations"][0].update(right_fact_ids=["Fa"]),
    lambda d: d["facts"][1].update(scope="省电模式"),
    lambda d: d["facts"][1].update(assessment="uncertain"),
    lambda d: d["snapshots"][0].update(text="被修改"),
    lambda d: d.update(resolver_selected_ids=[]),
    lambda d: d.update(writer_source_ids=[]),
    lambda d: d.update(writer_source_ids=["S1", "S2"]),
    lambda d: d["facts"].append(copy.deepcopy(d["facts"][0])),
])
def test_invalid_dependencies(mutation):
    d = record()
    mutation(d)
    with pytest.raises(ValueError):
        validate(d)


def test_unknown_missing_side_is_preserved():
    d = record()
    d["relations"][0].update(status="unknown", right_fact_ids=[])
    assert validate(d).relations[0].status == "unknown"


def test_semantic_error_is_not_falsely_claimed_detected():
    d = record()
    d["relations"][0].update(status="greater", explanation="甲更长")
    d["raw_answer"] = "甲更长[S1]"
    assert validate(d).raw_answer == "甲更长[S1]"


def test_duplicate_json_key_rejected():
    raw = json.dumps(record())
    raw = raw.replace('"raw_answer":', '"raw_answer": "hidden", "raw_answer":')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_comparison(raw)
