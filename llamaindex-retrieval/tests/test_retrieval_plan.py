import json

import pytest
from pydantic import ValidationError

from llamaindex_retrieval.retrieval_plan import RetrievalPlanV1, parse_plan, training_prompt


def plan(objects=None, dimensions=None, **changes):
    objects = objects or ["A", "B", "C"]
    dimensions = dimensions or ["部署", "硬件", "优化"]
    value = {"coverage": "grid", "objects": objects, "dimensions": dimensions,
             "conditions": [], "listed_pairs": [],
             "initial_queries": [{"object": obj, "query": f"{obj} 部署 硬件 优化 官方文档"}
                                 for obj in objects]}
    return {**value, **changes}


def test_three_by_three_requirements_need_only_three_initial_queries():
    result = RetrievalPlanV1.model_validate(plan())
    assert len(result.cells()) == 9
    assert len(result.initial_queries) == 3
    assert [cell["id"] for cell in result.cells()] == [f"c{i}" for i in range(1, 10)]


def test_four_by_four_and_explicit_listed_pairs():
    result = RetrievalPlanV1.model_validate(plan(objects=list("ABCD"), dimensions=list("1234")))
    assert len(result.cells()) == 16
    listed = plan(objects=["A", "B"], dimensions=["部署", "价格"], coverage="listed",
                  listed_pairs=[{"object": "A", "dimension": "部署"},
                                {"object": "B", "dimension": "价格"}])
    assert [(c["object"], c["dimension"]) for c in RetrievalPlanV1.model_validate(listed).cells()] == [
        ("A", "部署"), ("B", "价格")]


def test_missing_object_query_or_requirement_capacity_fails_instead_of_truncating():
    value = plan()
    value["initial_queries"] = value["initial_queries"][:2]
    with pytest.raises(ValidationError):
        RetrievalPlanV1.model_validate(value)
    with pytest.raises(ValidationError, match="continuation protocol required"):
        RetrievalPlanV1.model_validate(plan(objects=list("ABCDEF"), dimensions=list("12345")))


def test_duplicate_json_keys_and_extra_facts_are_rejected():
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_plan('{"coverage":"grid","coverage":"listed"}')
    with pytest.raises(ValidationError):
        parse_plan(json.dumps({**plan(), "answer": "A 最好"}))


def test_training_prompt_uses_exact_batch_prefix():
    prompt = training_prompt("比较 A、B 的部署", [])
    assert prompt.startswith("User: 只规划检索")
    assert prompt.endswith("\n\nAssistant: <think></think>")
