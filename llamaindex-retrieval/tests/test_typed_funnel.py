import asyncio
import json
from types import SimpleNamespace

import pytest

from llamaindex_retrieval import typed_funnel_contract as c
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.model_client import model_client_class
from llamaindex_retrieval.native_rwkv import NativeRWKVResult
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline
from llamaindex_retrieval.schemas import SourceItem
from llamaindex_retrieval.structured_native import StructuredNativeRWKVClient


def field(kind):
    return {"name": "属性", "question": "属性是什么？", "value_type": kind}


def atomic(value, quote, unit=None):
    return c.Atomic(observed=True, unit_ids=["E1"], quote=quote, value=value, unit=unit, scope=None)


@pytest.mark.parametrize("value,kind,quote,unit", [
    (False, "boolean", "不支持离线", None), (True, "boolean", "支持离线", None),
    ("0", "quantity", "故障0次", "次"), ("0", "quantity", "停机0小时", "小时"),
    ("11", "quantity", "流量11L/min", "L/min"), ("-2.5", "quantity", "温度-2.5°C", "°C"),
    ("无外部依赖", "text", "无外部依赖", None),
])
def test_known_values_preserve_zero_negation_unit(value, kind, quote, unit):
    result = c.validate_atomic(atomic(value, quote, unit), field(kind), {"E1": SimpleNamespace(text=quote)})
    assert result["observed"] and result["value"] == value and result["unit"] == unit


@pytest.mark.parametrize("value,kind,quote,unit", [
    ("8", "boolean", "容量8L", "L"), (False, "quantity", "不支持", None),
    ("8 L", "quantity", "容量8 L", None), ("NaN", "quantity", "NaN", None),
    ("1000", "quantity", "1m", "mm"), ("1", "quantity", "1m", "mm"),
])
def test_wrong_type_or_changed_source_numbers_rejected(value, kind, quote, unit):
    with pytest.raises(ValueError):
        c.validate_atomic(atomic(value, quote, unit), field(kind), {"E1": SimpleNamespace(text=quote)})


def test_unknown_is_distinct_from_zero_and_false():
    unknown = c.Atomic(observed=False, unit_ids=[], quote=None, value=None, unit=None, scope=None)
    assert c.validate_atomic(unknown, field("boolean"), {})["value"] is None
    with pytest.raises(ValueError):
        c.validate_atomic(unknown.model_copy(update={"value": False}), field("boolean"), {})


def test_quote_must_belong_to_selected_unit():
    with pytest.raises(ValueError):
        c.validate_atomic(atomic(False, "不支持离线"), field("boolean"), {"E1": SimpleNamespace(text="容量8L")})


def test_task_records_withdrawal_without_inventing_a_hard_ban():
    task = c.Task(mode="selection", objects=["甲", "乙"], fields=[c.Dimension(**field("boolean"))],
        requested_count=1, requirements=[c.Requirement(message_id="U2", quote="先不做Wiki", meaning="Wiki暂缓",
        state="withdrawn", kind="hard", field_names=[])])
    assert c.validate_task(task, {"U2": "先不做Wiki"})["requirements"][0]["state"] == "withdrawn"
    with pytest.raises(ValueError):
        c.validate_task(task, {"U2": "没有这句话"})


@pytest.mark.parametrize("status,judgments", [
    ("eligible", [{"id": "J1", "status": "unknown"}]),
    ("eligible", [{"id": "J1", "status": "not_satisfied"}]),
    ("eligible", []), ("ineligible", [{"id": "J1", "status": "unknown"}]),
    ("not_applicable", [{"id": "J1", "status": "satisfied"}]),
    ("unresolved", [{"id": "J1", "status": "satisfied"}]),
])
def test_qualification_cannot_contradict_model_condition_labels(status, judgments):
    candidate = c.Candidate(status=status, judgment_ids=[j["id"] for j in judgments], explanation="判断")
    with pytest.raises(ValueError):
        c.validate_candidate(candidate, judgments)


def test_qualification_must_cover_every_condition_and_never_cross_objects():
    candidate = c.Candidate(status="eligible", judgment_ids=["O2:R1"], explanation="判断")
    with pytest.raises(ValueError):
        c.validate_candidate(candidate, [{"id": "O1:R1", "status": "satisfied"}])


def test_ranking_cannot_revive_excluded_candidate_or_exceed_requested_count():
    candidates = [{"object_id": "O1", "status": "ineligible", "execution_status": "completed"},
                  {"object_id": "O2", "status": "eligible", "execution_status": "completed"}]
    with pytest.raises(ValueError):
        c.validate_decision(c.Decision(recommended_ids=["O1"], explanation="推荐"), candidates, 1)
    assert c.validate_decision(c.Decision(recommended_ids=["O2"], explanation="推荐"), candidates, 1)


def settings():
    return Settings(native_writer_pipeline="typed_funnel_v5", native_completion_protocol="g1j_plain",
        native_planner_prefill="<think></think", native_resolver_prefill="<think></think",
        native_writer_prefill="<think></think", generation_output_mode="immutable")


def test_runtime_explicitly_selects_structured_client_and_rejects_old_transport():
    assert model_client_class(settings()) is StructuredNativeRWKVClient
    with pytest.raises(ValueError):
        Settings(native_writer_pipeline="typed_funnel_v5")


class Model:
    def __init__(self, reject=False):
        self.calls = []
        self.reject = reject

    async def complete(self, messages, **kwargs):
        prompt = messages[0]["content"]
        self.calls.append((prompt, kwargs))
        if prompt.startswith("根据按时间"):
            value = {"mode": "comparison", "objects": ["甲"], "fields": [field("boolean")],
                "requirements": [{"message_id": "U1", "quote": "Wiki先不做", "meaning": "暂缓Wiki",
                    "state": "withdrawn", "kind": "hard", "field_names": []}], "requested_count": None}
        elif prompt.startswith("选择原文"):
            value = {"objects": ["甲"]}
        elif prompt.startswith("阅读资料"):
            value = atomic(False, "甲不支持离线").model_dump()
        elif prompt.startswith("核验一个事实"):
            value = {"verdict": "mismatch" if self.reject else "supported", "explanation": "核验记录"}
        elif prompt.startswith("归并一个对象"):
            value = {"status": "unknown" if self.reject else "supported", "fact_ids": [] if self.reject else ["A1"], "explanation": "归并记录"}
        elif prompt.startswith("汇总该候选"):
            value = {"status": "not_applicable", "judgment_ids": [], "explanation": "没有硬条件"}
        elif prompt.startswith("比较同一属性"):
            value = {"cell_ids": ["O1:F1"], "summary": "属性比较"}
        else:
            value = "保留原始答案[资料 1]。"
        raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return NativeRWKVResult("completed", raw, "stop", {"stage": kwargs["stage"], "raw_text": raw,
            "status": "completed", "completion_attempted": True, "prompt_protocol": "g1j_plain",
            "envelope": {"valid": True, "answer_span": {"start": 0, "end": len(raw), "unit": "unicode_code_points"}}})


@pytest.mark.parametrize("reject", [False, True])
def test_application_uses_funnel_and_keeps_rejected_fact_and_raw_answer(reject):
    model = Model(reject)
    pipeline = RWKVPipeline(settings(), None, model=model)
    source = SourceItem(id="s1", document_id="d1", source="test", title="甲", snippet="甲不支持离线", score=1, metadata={})
    result = asyncio.run(pipeline.ask_materials("甲如何，Wiki先不做", [source]))
    graph = result.retrieval["funnel"]
    assert result.answer == "保留原始答案[资料 1]。"
    assert graph["facts"][0]["value"] is False  # immutable proposal, even if rejected
    assert graph["conditions"] == []  # withdrawal never reaches condition judging
    assert graph["writer_source_ids"] == ([] if reject else ["s1"])
    assert not graph["failures"]
    assert len(model.calls) == 8
    assert all(kwargs.get("structured_schema") for _, kwargs in model.calls[:-1])
    assert "structured_schema" not in model.calls[-1][1]
