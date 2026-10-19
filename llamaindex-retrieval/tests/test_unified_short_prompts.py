"""Short input guidance must not change model output or conversation bytes."""
import json

import pytest

from llamaindex_retrieval.unified_contracts import Task, prompt, task_input
from llamaindex_retrieval.unified_retrieval import UnifiedRetrievalPipeline
from llamaindex_retrieval.unified_span_reading import SpanCatalog, span_prompt, synthesis_prompt
from test_unified_retrieval import material
from test_unified_span_pipeline import SpanModel, settings


def test_short_task_guidance_keeps_full_conversation_and_schema():
    messages = [{"id": "M1", "role": "assistant", "content": "旧答案\r\n{原样}"},
                {"id": "M2", "role": "user", "content": "改问另一个对象。"}]
    payload = task_input(messages)
    text = prompt("task", payload, Task, protocol="rwkv-unified-v4")
    guidance, tail = text.split("\n只输出符合以下结构的JSON，不要代码围栏：", 1)
    schema, raw_input = tail.split("\n输入数据：\n", 1)
    assert len(guidance) < 150
    assert json.loads(schema) == Task.model_json_schema()
    assert json.loads(raw_input) == payload
    assert payload["history"][0]["content"] == "旧答案\r\n{原样}"


@pytest.mark.parametrize("terminal", [False, True])
def test_short_synthesis_preserves_children_verbatim(terminal):
    payload = {"history": [{"role": "assistant", "content": "旧答案"}],
               "latest_user_message": {"role": "user", "content": "当前问题"},
               "children": [{"node_id": "N1", "raw_text": ' \n{"answer":"错误也保留"}\r\n'}],
               "inherited_span_bindings": {"E1.S1": {"leaf_call_id": "unchanged"}}}
    text = synthesis_prompt(payload, terminal=terminal)
    assert len(text.split("\n输出结构：", 1)[0]) < 150
    assert json.loads(text.split("\n输入数据：\n", 1)[1]) == payload


def test_readers_use_original_request_not_duplicate_control_interpretations():
    from test_unified_span_reading import material as original_material
    payload = {"history": [], "latest_user_message": {"role": "user", "content": "原问题"},
               "task": {"unresolved": ["派生解释"]}, "remaining": {"calls": 3}, "direction": "provided",
               "original_material": original_material()}
    before = json.dumps(payload)
    catalog = SpanCatalog.build(payload["original_material"])
    data = json.loads(span_prompt(payload, catalog).split("\n输入数据：\n", 1)[1])
    assert data == {"history": [], "latest_user_message": payload["latest_user_message"],
                    "original_material": catalog.model_material()}
    assert json.dumps(payload) == before


def test_synthesis_keeps_raw_and_execution_gaps_without_control_state():
    payload = {"history": [], "latest_user_message": {"role": "user", "content": "原问题"},
               "task": {"unresolved": ["派生解释"]}, "remaining": {"calls": 3},
               "children": [{"raw_text": ' \n{"answer":"task/remaining等字样也不删"}\r\n'}],
               "inherited_span_bindings": {"E1.S1": {"leaf_call_id": "unaltered"}},
               "execution_scope": {"unread_units": 2, "failed_nodes": ["N2"],
                                   "task_unresolved": ["派生解释"], "round_budgets": []}}
    before = json.dumps(payload)
    data = json.loads(synthesis_prompt(payload, terminal=True).split("\n输入数据：\n", 1)[1])
    assert set(data) == {"history", "latest_user_message", "children", "inherited_span_bindings", "execution_scope"}
    assert data["children"] == payload["children"]
    assert data["inherited_span_bindings"] == payload["inherited_span_bindings"]
    assert data["execution_scope"] == {"unread_units": 2, "failed_nodes": ["N2"]}
    assert json.dumps(payload) == before


@pytest.mark.parametrize("control", ["single_direction_v1", "review_cycle_v1"])
@pytest.mark.parametrize("output", [
    ' \n{"answer":"没有依据的错误回答，原样保留。","span_ids":["E1.S1"],"missing":[]}\r\n',
    '```json\n{"answer":"不能剥掉围栏修复"}\n```',
    ("length", '{"answer":"截断的输出也不补全'),
])
async def test_raw_root_reaches_api_unchanged_even_when_wrong_or_invalid(control, output):
    model = SpanModel(overrides={"final": output})
    response = await UnifiedRetrievalPipeline(settings(control_protocol=control), None, model=model).ask_materials(
        "问题", [material()])
    raw = output[1] if isinstance(output, tuple) else output
    root = next(n for n in response.generation["nodes"] if n["id"] == response.generation["root_node_id"])
    call = next(c for c in response.generation["model_calls"] if c["purpose"] == "final")
    assert response.answer == response.generation["raw_model_answer"] == root["raw_text"] == call["raw_text"] == raw
    assert model.counts["final"] == 1
