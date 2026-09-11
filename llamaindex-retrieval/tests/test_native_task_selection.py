import json

import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.lexical_index import LexicalResult
from llamaindex_retrieval.native_rwkv import NativeRWKVResult
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline, parse_task_selections
from llamaindex_retrieval.schemas import SearchRequest


@pytest.mark.parametrize("text, expected", [
    ("NONE", []), (" E1 \n", [1]), ("E2,E1,E2", [2, 1]),
    ("E2，E1", [2, 1]), ("E1;\nE2", [1, 2]),
])
def test_task_selection_only_interprets_actual_ids(text, expected):
    assert parse_task_selections(text, 2) == expected


@pytest.mark.parametrize("text", [
    "", "E0", "E3", "E01", "E1,", "E1,NONE", "NONE,E1", "f1: E1",
    "E1 because it is relevant", "25000", "E1E2", "E1\nE2", "E1: answer",
    "[E1]", "E1 E2", "E１", "NONE explanation",
])
def test_bad_task_selection_cannot_leak_valid_prefix_to_writer(text):
    with pytest.raises(ValueError):
        parse_task_selections(text, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ["E1", "NONE", "E1,E2"])
async def test_task_selection_binds_source_and_never_claims_field_coverage(selection):
    raw_answer = ">原样思考</think>\n原样回答，不由代码补充。"
    calls = []

    class Index:
        def search_chunks(self, query, **kwargs):
            assert query == "甲对象的完整问题"
            return [LexicalResult(node_id="source", document_id="doc", text="原文中的事实。",
                metadata={"title": "甲", "source": "fixture"}, score=1)]

    class Model:
        async def complete(self, messages, *, stage, evidence_ids, assistant_prefill, **kwargs):
            prompt = messages[0]["content"]
            calls.append({"stage": stage, "prompt": prompt, "prefill": assistant_prefill})
            body = '["甲对象的完整问题"]' if stage == "planner" else selection
            raw = ">" + body if stage != "writer" else raw_answer
            return NativeRWKVResult(status="completed", raw_text=raw, finish_reason="stop",
                trace={"stage": stage, "status": "completed", "raw_text": raw,
                       "prefill": assistant_prefill})

    settings = Settings(native_plan_protocol="shared_tasks", native_planner_prefill="<think></think",
        native_resolver_protocol="task_units", native_resolver_prefill="<think></think")
    response = await RWKVPipeline(settings, Index(), Model()).ask(SearchRequest(question="当前问题"))
    assert response.answer == raw_answer
    reader, writer = calls[1:]
    assert reader["prefill"] == "<think></think" and writer["prefill"] == "<think"
    assert json.loads(reader["prompt"].rsplit("原文：", 1)[1]) == {"E1": "原文中的事实。"}
    assert "E2" not in reader["prompt"]
    assert response.retrieval["uncovered_fields"] is None
    assert response.retrieval["field_coverage_assessed"] is False
    visible = json.loads(writer["prompt"].split("逐字证据：", 1)[1])
    if selection == "E1":
        assert [item["snippet"] for item in [source.model_dump() for source in response.sources]] == ["原文中的事实。"]
        assert response.sources[0].metadata["field_ids"] == []
        assert response.sources[0].metadata["selection_scope"] == "task_units"
        assert [item["text"] for item in visible] == ["原文中的事实。"]
        assert response.generation["model_calls"][1]["selected_units"] == ["E1"]
    else:
        assert response.sources == [] and visible == []
    assert response.generation["status"] == (
        "resolver_partial_failure" if selection == "E1,E2" else "completed")


@pytest.mark.asyncio
async def test_individual_tasks_keep_history_deduplicate_spans_and_reject_bad_selection():
    from llamaindex_retrieval.schemas import SourceItem
    calls = []
    task = json.dumps({"history": [{"role": "user", "content": "先看旧版"}],
        "latest_question": "改为新版，分别查价格和保修"}, ensure_ascii=False)
    source = SourceItem(id="s", document_id="doc", title="新版", source="fixture", score=1, snippet="价格37元。保修两年。")
    class Model:
        async def complete(self, messages, *, stage, **kwargs):
            prompt = messages[0]["content"]
            assert f"任务：{task}\n" in prompt
            group = json.JSONDecoder().raw_decode(prompt.split("子问题：", 1)[1])[0]
            calls.append(group)
            raw = ">" + ("E1,E1" if group == ["新版价格"] else "E1 because yes")
            return NativeRWKVResult(status="completed", raw_text=raw, finish_reason="stop",
                trace={"stage":stage, "status":"completed", "raw_text":raw, "prefill":"<think></think"})
    settings = Settings(native_resolver_protocol="task_units", native_resolver_prefill="<think></think",
        native_resolver_task_grouping="individual")
    evidence, events = await RWKVPipeline(settings, None, Model())._resolve(task, ["新版价格", "新版保修"], [source])
    assert calls == [["新版价格"], ["新版保修"]]
    assert len(evidence) == 1 and evidence[0].snippet == source.snippet
    assert evidence[0].metadata["field_ids"] == []
    assert events[0]["selected_units"] == ["E1"]
    assert "parse_error" in events[1] and "selected_units" not in events[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("first_status, second, count, selected", [
    ("completed", "E1", 2, True), ("completed", "1", 2, False),
    ("length", "E1", 1, False), ("completed", "NONE", 2, False),
])
async def test_format_repair_is_one_independent_call_and_never_coerces_raw(first_status, second, count, selected):
    from llamaindex_retrieval.schemas import SourceItem
    prompts = []
    class Model:
        async def complete(self, messages, *, stage, **kwargs):
            prompts.append(messages[0]["content"])
            i = len(prompts)
            raw = ">" + ("1" if i == 1 else second)
            status = first_status if i == 1 else "completed"
            return NativeRWKVResult(status=status, raw_text=raw, finish_reason="stop",
                trace={"call_id":str(i),"stage":stage,"status":status,"raw_text":raw,"prefill":"<think></think"})
    settings = Settings(native_resolver_protocol="task_units",native_resolver_prefill="<think></think",
        native_resolver_format_repair=True)
    source = SourceItem(id="s",document_id="doc",title="资料",source="fixture",score=1,snippet="原文事实。")
    evidence, events = await RWKVPipeline(settings,None,Model())._resolve("原始问题",["原始子问题"],[source])
    assert len(prompts) == count and len(events) == count
    assert bool(evidence) == selected
    assert events[0]["raw_text"] == ">1" and "parse_error" in events[0]
    if count == 2:
        assert events[1]["raw_text"] == ">" + second
        assert events[1]["format_repair_of_call_id"] == "1"
        assert prompts[0].split("\n",1)[1] == prompts[1].split("\n",1)[1]
        assert events[0].get("format_repair_succeeded",False) == (second in {"E1","NONE"})
        if second == "1": assert "parse_error" in events[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("scope, evidence_count", [("global", 0), ("per_query", 2)])
async def test_query_budget_reaches_deeper_evidence_without_cross_query_reader_calls(scope, evidence_count):
    reads = []

    class Index:
        def search_chunks(self, query, **kwargs):
            return [LexicalResult(node_id=query + suffix, document_id=query,
                text=query + text, metadata={"title":query}, score=score)
                for suffix, text, score in [("-heading", "标题", 2), ("-fact", "明确事实", 1)]]

    class Model:
        async def complete(self, messages, *, stage, assistant_prefill, **kwargs):
            prompt = messages[0]["content"]
            if stage == "planner":
                body = '["甲问题", "乙问题"]'
            elif stage == "resolver":
                query = json.JSONDecoder().raw_decode(prompt.split("子问题：", 1)[1])[0]
                unit = json.loads(prompt.split("原文：", 1)[1])["E1"]
                reads.append((query, unit))
                if scope == "per_query":
                    assert unit.startswith(query[0])
                body = "E1" if unit.endswith("明确事实") else "NONE"
            else:
                body = "原始回答"
            raw = ">" + body
            return NativeRWKVResult(status="completed", raw_text=raw, finish_reason="stop",
                trace={"stage":stage,"status":"completed","raw_text":raw,"prefill":assistant_prefill})

    settings = Settings(native_plan_protocol="shared_tasks", native_task_source="queries",
        native_candidate_order="query_round_robin", native_resolver_sources=2,
        native_resolver_protocol="task_units", native_resolver_task_grouping="individual",
        native_resolver_budget_scope=scope, native_planner_prefill="<think></think",
        native_resolver_prefill="<think></think", native_writer_prefill="<think></think")
    response = await RWKVPipeline(settings, Index(), Model()).ask(SearchRequest(question="两个问题"))
    assert len(reads) == 4
    assert len(response.sources) == evidence_count
    assert all(source.snippet.endswith("明确事实") for source in response.sources)
    assert response.answer == ">原始回答"
    assert response.retrieval["field_coverage_assessed"] is False


@pytest.mark.parametrize("invalid", [
    {"native_task_source":"fields"}, {"native_resolver_task_grouping":"joint"},
    {"native_resolver_protocol":"fields"},
])
def test_per_query_budget_requires_matching_query_task_route(invalid):
    args = dict(native_resolver_budget_scope="per_query", native_task_source="queries",
        native_resolver_task_grouping="individual", native_resolver_protocol="task_units")
    with pytest.raises(ValueError, match="Per-query Reader budget"):
        Settings(**(args | invalid))
