"""Canned model decisions test plumbing only, not history/condition accuracy."""
import asyncio
import copy
import json
from hashlib import sha256

import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.current_question import correction_prompt
from llamaindex_retrieval.native_rwkv import NativeRWKVResult
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline, conversation
from llamaindex_retrieval.schemas import ConversationMessage, SearchRequest, SourceItem
from llamaindex_retrieval.typed_funnel_v10 import write_funnel
from test_current_question import CorrectionModel
from test_rwkv_pipeline import FakeIndex, hit, native_result


@pytest.mark.parametrize("latest,current", [
    ("不要容器，只问本机。", "设备甲在本机如何持久化？"),
    ("只比较第二、第三种。", "比较设备乙与设备丙的温度。"),
    ("撤销采购条件，只查维护。", "设备甲如何维护？"),
])
def test_model_authored_task_bytes_reach_every_stage_withdrawals_not_remerged(latest, current):
    old = "比较设备甲乙丙，要求容器部署并采购。"
    model = CorrectionModel(json.dumps({"question": current}, ensure_ascii=False),
                            plan={"queries": [current], "fields": [current]},
                            writer_raw=">done</think>  原始回答[资料 1]\n")
    index = FakeIndex({current: [hit("one", "供测试使用的逐字材料。", "document")]})
    pipe = RWKVPipeline(Settings(_env_file=None, native_history_protocol="current-question-v1"),
                        index, model=model)
    request = SearchRequest(question=latest, history=[ConversationMessage(role="user", content=old),
        ConversationMessage(role="assistant", content="未被用户采纳的建议：增加采购范围。")])
    before = request.model_dump()
    result = asyncio.run(pipe.ask(request))
    assert request.model_dump() == before
    assert {stage[0] for stage in index.calls} == {current}
    for call in model.calls:
        assert current in call["prompt"] and old not in call["prompt"] and latest not in call["prompt"]
        assert "增加采购范围" not in call["prompt"]
    event = result.generation["model_calls"][0]
    assert event["parsed_question"] == current
    assert event["raw_text"] == ">done</think>" + model.correction
    assert "增加采购范围" in event["original_task"]
    assert result.generation["raw_model_answer"] == model.writer_raw


def test_user_only_correction_protocol_exposes_its_assistant_reference_limitation():
    history = [ConversationMessage(role="user", content="列出两个方案"),
               ConversationMessage(role="assistant", content="第一种是甲；第二种是乙。")]
    prompt = correction_prompt("就选你说的第二种", history)
    data = json.loads(prompt.split("\n", 1)[1])
    assert data == {"previous_user_questions": ["列出两个方案"], "correction": "就选你说的第二种"}
    # This is a documented missing input, NOT proof that the model resolves it.
    assert "第二种是乙" not in prompt


@pytest.mark.parametrize("status", ["length", "timeout", "invalid_response"])
def test_failed_current_question_is_not_accepted_even_with_parseable_raw_json(status):
    class Model(CorrectionModel):
        async def complete(self, messages, **kwargs):
            return native_result("planner", '>done</think>{"question":"不可采信的完整问题"}', status)
    model = Model("unused")
    index = FakeIndex({})
    pipe = RWKVPipeline(Settings(_env_file=None, native_history_protocol="current-question-v1"),
                        index, model=model)
    result = asyncio.run(pipe.ask(SearchRequest(question="改一下", history=[
        ConversationMessage(role="user", content="旧问题")])))
    assert not index.calls and result.answer == ""
    assert result.generation["status"] == "current_question_failed"
    assert result.generation["model_calls"][0]["status"] == status
    assert "不可采信" in result.generation["model_calls"][0]["raw_text"]


class ConditionModel:
    def __init__(self, values, statuses, *, bad_status=None, missing=False):
        self.values, self.statuses = values, statuses
        self.bad_status, self.missing = bad_status, missing
        self.calls = []

    async def complete(self, messages, **kwargs):
        prompt = messages[0]["content"]
        data = json.loads(prompt.split("\n", 1)[1]) if kwargs["stage"] != "writer" else None
        status = "completed"
        if prompt.startswith("选择原文"):
            value = {"objects": [data["title"]]}
        elif prompt.startswith("阅读原文"):
            quote = next(iter(data["evidence"].values()))
            value = {"evidence_ids": ["E1"], "quote": quote,
                     "value": None if self.missing else self.values[data["object"]], "source_scope": None}
        elif prompt.startswith("从给出的原文中逐字摘录"):
            value = {"value": self.values[data["object"]]}
        elif prompt.startswith("核验一个事实"):
            value = {"verdict": "supported", "explanation": "固定的测试核验输出"}
        elif prompt.startswith("归并一个对象"):
            value = {"status": "supported" if data["facts"] else "unknown",
                     "fact_ids": [f["id"] for f in data["facts"]], "explanation": "固定归并输出"}
        elif prompt.startswith("只判断一个候选"):
            value = {"status": self.statuses[data["object"]], "explanation": "原始条件判定；不改写。"}
            status = self.bad_status or "completed"
        elif prompt.startswith("汇总该候选"):
            judgments = data["judgments"]
            selected = "unresolved" if any(j["execution_status"] == "failed" for j in judgments) else {
                "satisfied": "eligible", "not_satisfied": "ineligible", "unknown": "unresolved",
                "conflict": "unresolved"}[judgments[0]["status"]]
            value = {"status": selected, "explanation": "固定资格输出"}
        elif prompt.startswith("比较同一属性"):
            value = {"summary": "固定测试摘要；不表示语义验证。"}
        else:
            assert kwargs["stage"] == "writer"
            value = "  原始Writer输出，不在代码中修复[资料 99]。\n"
        raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        trace = {"stage": kwargs["stage"], "raw_text": raw, "status": status, "prompt": prompt,
                 "prompt_sha256": sha256(prompt.encode()).hexdigest(),
                 "evidence_ids": list(kwargs.get("evidence_ids", ())), "prompt_protocol": "g1j_plain",
                 "envelope": {"valid": True, "answer_span": {"start": 0, "end": len(raw),
                                                             "unit": "unicode_code_points"}}}
        self.calls.append(trace)
        return NativeRWKVResult(status, raw, "length" if status == "length" else "stop", trace)


@pytest.mark.parametrize("values,kind,labels,missing,bad_status", [
    ({"甲": "2 GiB", "乙": "1024 MiB"}, "quantity", {"甲": "satisfied", "乙": "not_satisfied"}, False, None),
    ({"甲": "0 L", "乙": "0 L"}, "quantity", {"甲": "satisfied", "乙": "satisfied"}, False, None),
    ({"甲": False, "乙": False}, "boolean", {"甲": "not_satisfied", "乙": "not_satisfied"}, False, None),
    ({"甲": None, "乙": None}, "quantity", {"甲": "unknown", "乙": "unknown"}, True, None),
    ({"甲": "0 L", "乙": "0 L"}, "quantity", {"甲": "conflict", "乙": "conflict"}, False, None),
    ({"甲": None, "乙": None}, "quantity", {"甲": "satisfied", "乙": "satisfied"}, True, None),
    ({"甲": "0 L", "乙": "0 L"}, "quantity", {"甲": "satisfied", "乙": "satisfied"}, False, "length"),
])
def test_conditions_keep_object_dependencies_units_and_failed_vs_unknown(monkeypatch, values, kind, labels,
                                                                        missing, bad_status):
    spec = {"mode": "comparison", "objects": ["甲", "乙"], "requested_count": None,
            "fields": [{"name": "属性", "question": "属性是什么？", "value_type": kind}],
            "requirements": [{"kind": "hard", "state": state, "field_names": ["属性"],
                              "meaning": text, "quote": text, "message_id": "U1",
                              "lifecycle_message_id": "U1", "lifecycle_quote": text}
                             for state, text in [("active", "用户仍生效的原始条件"),
                                                 ("withdrawn", "已撤回的旧条件")]]}
    async def fixed_task(*args):
        return copy.deepcopy(spec)
    monkeypatch.setattr("llamaindex_retrieval.funnel_task_v2.build_task", fixed_task)
    model = ConditionModel(values, labels, missing=missing, bad_status=bad_status)
    settings = Settings(_env_file=None, native_writer_pipeline="typed_funnel_v10",
        native_completion_protocol="g1j_plain", native_quantity_binding="verbatim",
        native_planner_prefill="<think></think", native_resolver_prefill="<think></think",
        native_writer_prefill="<think></think")
    pipe = RWKVPipeline(settings, None, model=model)
    sources = [SourceItem(id=name, document_id="d-" + name, source="test", title=name, score=1,
                         snippet=f"{name}的原始记录：{value}。") for name, value in values.items()]
    result = asyncio.run(write_funnel(pipe, conversation("按现有条件比较甲乙", []), sources))
    graph = result.trace["funnel"]
    assert len(graph["cells"]) == len(graph["conditions"]) == len(graph["candidates"]) == 2
    assert {c["requirement_id"] for c in graph["conditions"]} == {"R1"}
    failed = bool(bad_status) or (missing and "satisfied" in labels.values())
    for row in graph["conditions"]:
        assert row["execution_status"] == ("failed" if failed else "completed")
        if not failed:
            name = spec["objects"][int(row["object_id"][1:]) - 1]
            assert row["status"] == labels[name]
            assert row["cell_ids"] == [row["object_id"] + ":F1"]
        else:
            assert "status" not in row  # failure is not silently rewritten as unknown/NO
    judgments = [c for c in model.calls if c["prompt"].startswith("只判断一个候选")]
    assert len(judgments) == 2
    for call in judgments:
        data = json.loads(call["prompt"].split("\n", 1)[1])
        assert all(c["object"] == data["object"] for c in data["cells"])
        assert "已撤回的旧条件" not in call["prompt"]
        assert "原始条件判定；不改写。" in call["raw_text"]
    if not missing:
        assert [fact["value"] for fact in graph["facts"]] == list(values.values())
    assert result.raw_text == "  原始Writer输出，不在代码中修复[资料 99]。\n"
    assert graph["writer_source_ids"] == ([] if missing else ["甲", "乙"])
