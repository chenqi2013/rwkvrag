import asyncio
import json
import pytest
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.native_rwkv import NativeRWKVResult
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline, conversation
from llamaindex_retrieval.schemas import SourceItem
from llamaindex_retrieval.typed_funnel_v9 import write_funnel


class Model:
    def __init__(self, version=9):
        self.calls = []
        self.version = version

    async def complete(self, messages, **kwargs):
        prompt = messages[0]["content"]
        self.calls.append((prompt, kwargs))
        if prompt.startswith("选择原文"):
            value = {"objects": ["甲", "乙"]}
        elif prompt.startswith("阅读原文"):
            value = {"evidence_ids": ["E1"], "quote": "甲支持离线和加密，乙不支持离线和加密。", "value": False, "source_scope": None}
        elif prompt.startswith("核验一个事实"):
            value = {"verdict": "supported", "explanation": "测试核验记录"}
        elif prompt.startswith("归并一个对象"):
            facts = json.loads(prompt.split("\n", 1)[1])["facts"]
            value = {"status": "supported" if facts else "unknown", "fact_ids": [f["id"] for f in facts], "explanation": "测试归并记录"}
        elif prompt.startswith("汇总该候选"):
            value = {"status": "not_applicable", "judgment_ids": [], "explanation": "无硬条件"}
        elif prompt.startswith("比较同一属性"):
            cells = json.loads(prompt.split("\n", 1)[1])["cells"]
            value = {"summary": "测试比较记录"}
            if self.version == 9:
                value["cell_ids"] = [c["id"] for c in cells]
        else:
            value = "模型的原始回答[资料 1]。"
        raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return NativeRWKVResult("completed", raw, "stop", {"stage": kwargs["stage"], "raw_text": raw,
            "status": "completed", "completion_attempted": True, "prompt_protocol": "g1j_plain",
            "envelope": {"valid": True, "answer_span": {"start": 0, "end": len(raw), "unit": "unicode_code_points"}}})


@pytest.mark.parametrize("version", [9, 10])
def test_full_downstream_chain_runs_when_atomic_jobs_exceed_call_budget(monkeypatch, version):
    async def task(*args):
        return {"mode": "comparison", "objects": ["甲", "乙"], "requirements": [], "requested_count": None,
                "fields": [{"name": name, "question": name, "value_type": "boolean"} for name in ["离线支持", "加密支持"]]}
    monkeypatch.setattr("llamaindex_retrieval.funnel_task.build_task" if version == 9 else "llamaindex_retrieval.funnel_task_v2.build_task", task)
    settings = Settings(native_writer_pipeline="typed_funnel_v8", native_completion_protocol="g1j_plain",
        native_planner_prefill="<think></think", native_resolver_prefill="<think></think",
        native_writer_prefill="<think></think", native_funnel_max_calls=28)
    model = Model(version)
    pipeline = RWKVPipeline(settings, None, model=model)
    sources = [SourceItem(id=f"s{i}", document_id=f"d{i}", source="test", title="资料",
        snippet="甲支持离线和加密，乙不支持离线和加密。", score=1, metadata={}) for i in range(6)]
    if version == 10:
        from llamaindex_retrieval.typed_funnel_v10 import write_funnel as writer
    else:
        writer = write_funnel
    result = asyncio.run(writer(pipeline, conversation("比较甲乙", []), sources))
    graph = result.trace["funnel"]
    assert len(model.calls) <= 28
    assert result.raw_text == "模型的原始回答[资料 1]。"
    assert len(graph["cells"]) == 4 and all(c["execution_status"] == "completed" for c in graph["cells"])
    assert len(graph["candidates"]) == (2 if version == 9 else 0)
    assert all(c["execution_status"] == "completed" for c in graph["candidates"])
    assert len(graph["field_summaries"]) == 2 and all(c["execution_status"] == "completed" for c in graph["field_summaries"])
    assert graph["call_budget"]["requested_jobs"] == 24
    assert len(graph["call_budget"]["unexamined_jobs"]) == 20
    assert {f["status"] for f in graph["failures"]} == {"call_budget_unexamined"}


def test_task_failure_retains_failed_trace_and_independent_writer_source_map(monkeypatch):
    async def failed_task(runner, messages):
        runner.calls.append({"stage": "resolver", "status": "invalid_response", "raw_text": "原始失败标签"})
        runner.failures.append({"purpose": "task", "status": "invalid_response"})
        return None
    monkeypatch.setattr("llamaindex_retrieval.funnel_task_v2.build_task", failed_task)
    model = Model(10)
    settings = Settings(native_writer_pipeline="typed_funnel_v10", native_completion_protocol="g1j_plain",
        native_planner_prefill="<think></think", native_resolver_prefill="<think></think", native_writer_prefill="<think></think")
    pipeline = RWKVPipeline(settings, None, model=model)
    source = SourceItem(id="one", document_id="d", source="test", title="原文", snippet="原文事实", score=1, metadata={})
    result = asyncio.run(pipeline.ask_materials("查这个事实", [source]))
    assert result.answer == result.generation["raw_model_answer"] == "模型的原始回答[资料 1]。"
    assert result.retrieval["funnel"]["status"] == "task_failed_independent_writer"
    assert result.generation["status"] == "funnel_partial_failure"
    assert [s.id for s in result.sources] == ["one"]
    assert result.generation["model_calls"][0]["raw_text"] == "原始失败标签"
    assert sum(not kwargs.get("check_only", False) for _, kwargs in model.calls) == 1
