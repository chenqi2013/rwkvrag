"""Offline contracts for the native RAG chain; no model or remote index calls."""

import asyncio
from dataclasses import replace
import json

import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.lexical_index import LexicalResult
from llamaindex_retrieval.native_rwkv import NativeRWKVResult
from llamaindex_retrieval.rwkv_pipeline import (
    RWKVPipeline,
    conversation,
    digest,
    evidence_units,
    fuse_chunks,
    parse_plan,
    parse_selections,
    source_from_hit,
    structured_body,
)
from llamaindex_retrieval.schemas import ConversationMessage, SearchRequest


def hit(identity: str, text: str, document: str = "shared-document") -> LexicalResult:
    return LexicalResult(
        node_id=identity, document_id=document, text=text,
        metadata={"title": f"Document {document}", "source": "fixture",
                  "uri": f"https://example.test/{document}", "revision": 7},
        score=10.0,
    )


def native_result(stage: str, raw: str | None, status: str = "completed") -> NativeRWKVResult:
    return NativeRWKVResult(
        status=status, raw_text=raw,
        finish_reason="stop" if status == "completed" else "length" if status == "length" else None,
        trace={"stage": stage, "status": status, "raw_text": raw},
    )


def read_task(prompt: str) -> dict:
    return json.JSONDecoder().raw_decode(prompt.split("任务：", 1)[1])[0]


class FakeIndex:
    def __init__(self, groups):
        self.groups = groups
        self.calls = []

    def search_chunks(self, query, *, candidate_k, knowledge_base_id=None):
        self.calls.append((query, candidate_k, knowledge_base_id))
        return list(self.groups[query])


class FakeModel:
    def __init__(
        self, *, plan=None, resolver_outputs=None, writer_raw=None,
        planner_result=None, writer_status="completed",
    ):
        self.plan = plan or {"queries": ["alpha query", "beta query"], "fields": ["所求字段"]}
        self.resolver_outputs = resolver_outputs or {}
        self.writer_raw = writer_raw
        self.planner_result = planner_result
        self.writer_status = writer_status
        self.calls = []
        self.resolver_active = 0
        self.resolver_peak = 0
        self.closed = False
        self.last_writer_raw = None

    async def complete(
        self, messages, *, max_tokens, stage, evidence_ids=(), assistant_prefill="<think",
    ):
        assert len(messages) == 1 and messages[0]["role"] == "user"
        prompt = messages[0]["content"]
        self.calls.append({"stage": stage, "prompt": prompt,
                           "evidence_ids": tuple(evidence_ids), "max_tokens": max_tokens})
        if stage == "planner":
            return self.planner_result or native_result(
                stage, ">规划思考</think>\n" + json.dumps(self.plan, ensure_ascii=False),
            )
        if stage == "resolver":
            self.resolver_active += 1
            self.resolver_peak = max(self.resolver_peak, self.resolver_active)
            try:
                await asyncio.sleep(0.005)
                override = self.resolver_outputs.get(evidence_ids[0])
                if isinstance(override, NativeRWKVResult):
                    return override
                if override is not None:
                    return native_result(stage, ">选择思考</think>\n" + override)
                units = json.loads(prompt.split("原文：", 1)[1])
                body = "\n".join(
                    f"f{i}: " + ",".join(units)
                    for i in range(1, len(self.plan["fields"]) + 1)
                )
                return native_result(stage, ">选择思考</think>\n" + body)
            finally:
                self.resolver_active -= 1
        assert stage == "writer"
        assert self.resolver_active == 0
        evidence = json.loads(prompt.split("逐字证据：", 1)[1])
        raw = self.writer_raw
        if raw is None and self.writer_status == "completed":
            raw = ">写作思考</think>\n结论。[资料 " + str(len(evidence)) + "]\n"
        self.last_writer_raw = raw
        return native_result(stage, raw, self.writer_status)

    async def aclose(self):
        self.closed = True


def settings(**overrides):
    return Settings(
        native_resolver_window_characters=512,
        native_resolver_overlap_characters=80,
        native_resolver_batch_characters=1000,
        native_resolver_sources=20,
        **overrides,
    )


def test_completed_native_protocol_is_interpreted_without_mutating_raw_text():
    raw = '> 原样思考 </think>\n {"queries":["q"],"fields":["f"]} \n'
    result = native_result("planner", raw)
    assert structured_body(result) == '{"queries":["q"],"fields":["f"]}'
    assert result.raw_text == raw


def test_fenced_plan_is_interpreted_without_changing_the_model_output():
    raw = '> 原样思考 </think>\n```json\n{"queries":["q"],"fields":["f"]}\n```'
    result = native_result("planner", raw)
    assert parse_plan(structured_body(result), settings()) == {"queries": ["q"], "fields": ["f"]}
    assert result.raw_text == raw


@pytest.mark.parametrize("raw", [
    '说明\n```json\n{"queries":["q"],"fields":["f"]}\n```',
    '```json\n{"queries":["q"],"fields":["f"]}\n```\n额外说明',
    '```json\n{"queries":["q"],"fields":["f"]}\n```\n```json\n{}\n```',
    '```python\n{"queries":["q"],"fields":["f"]}\n```',
    '```json\n{"queries":[],"fields":["f"]}\n```',
])
def test_fenced_plan_does_not_admit_prose_multiple_blocks_or_invalid_schema(raw):
    with pytest.raises(ValueError):
        parse_plan(raw, settings())


def test_fenced_plan_preserves_query_budget():
    raw = "```json\n" + json.dumps({"queries": ["q"] * 7, "fields": ["f"]}) + "\n```"
    with pytest.raises(ValueError, match="invalid query count"):
        parse_plan(raw, settings(native_max_queries=6))


@pytest.mark.parametrize("status", ["length", "budget_exceeded", "http_error", "timeout"])
def test_unfinished_model_output_cannot_be_used_as_structured_selection(status):
    with pytest.raises(ValueError):
        structured_body(native_result("resolver", ">think</think>f1: E1", status))


@pytest.mark.parametrize("body", [
    'f1: E1\nexplanation', 'f1: E1\nf1: E2', 'f2: E1', 'f1: E0', 'f1: E3',
])
def test_bad_resolver_selection_is_rejected_as_a_whole(body):
    with pytest.raises(ValueError):
        parse_selections(body, fields=1, units=2)


def test_selection_requires_all_fields_and_accepts_explicit_none():
    assert parse_selections("f1: E2,E1,E2\nf2: NONE", 2, 2) == [(1, 2), (1, 1)]
    with pytest.raises(ValueError, match="omitted"):
        parse_selections("f1: E1", 2, 2)


@pytest.mark.parametrize("separator", ["\n", ", ", "; ", ",\r\n", ";\n\n", "\r\n"])
def test_selection_v2_accepts_unambiguous_assignment_separators(separator):
    raw = separator.join(["f1: E1", "f2: E1", "f3: NONE"])
    assert parse_selections(raw, fields=3, units=1) == [(1, 1), (2, 1)]


def test_selection_v2_distinguishes_evidence_commas_from_field_commas():
    raw = "\n\tf3: E2, E1,E2, f1 : NONE;\n f2: E1,\nE3\n"
    assert parse_selections(raw, fields=3, units=3) == [(3, 2), (3, 1), (2, 1), (2, 3)]


def test_selection_v2_all_none_is_a_complete_valid_empty_selection():
    assert parse_selections("f1: NONE, f2: NONE; f3: NONE", 3, 0) == []


@pytest.mark.parametrize("separator", ["\n", ", ", "; "])
def test_selection_v2_rejects_repeated_fields_even_when_they_select_nothing(separator):
    with pytest.raises(ValueError, match="repeated"):
        parse_selections(separator.join(["f1: NONE", "f2: E1", "f1: NONE"]), 2, 1)


@pytest.mark.parametrize("raw", [
    "", " \t\r\n", "f1:", "f1: E1,", "f1: E1;\n", "f1: E1,, f2: NONE",
    "f1: E1;; f2: NONE", "f1: E1,; f2: NONE", "f1: E1 f2: NONE",
    "f1: E1E2, f2: NONE", "f1: E1\nE2, f2: NONE", "f1: E1; E2, f2: NONE",
    "f1: NONE,E1, f2: NONE", "f1: E1,NONE, f2: NONE",
    "f1: NONE,NONE, f2: NONE", "f1: 25000, f2: NONE", "f1: £25,000, f2: NONE",
    "f1: E1, f2: NONE because no evidence", "f1: E1; f2: NONE\n说明",
    "f1: E1, f2: NONE, E1", "Here are the IDs: f1: E1, f2: NONE",
    "```\nf1: E1\nf2: NONE\n```", "f1: E1, f2: NONE.", "f1: E1, f2: NONE\x00",
    "f1: E1，f2: NONE", "f1: E1\u200bf2: NONE", "f1: E1 # comment\nf2: NONE",
])
def test_selection_v2_consumes_every_character_and_rejects_ambiguous_or_extra_text(raw):
    with pytest.raises(ValueError):
        parse_selections(raw, fields=2, units=2)


@pytest.mark.parametrize("raw", [
    "f0: E1, f2: NONE", "f01: E1, f2: NONE", "f1: E1, f3: NONE",
    "f1: E0, f2: NONE", "f1: E01, f2: NONE", "f1: E3, f2: NONE",
    "f1: E1,E3, f2: NONE", "f1: E1, f2: E3", "f1: E1, f2: E２",
    "F1: E1, f2: NONE", "f1: e1, f2: NONE", "f1: E1, f2: none",
])
def test_selection_v2_preserves_canonical_ids_and_all_range_checks(raw):
    with pytest.raises(ValueError):
        parse_selections(raw, fields=2, units=2)


@pytest.mark.asyncio
async def test_selection_v2_only_hands_valid_ids_to_writer_and_preserves_raw_output():
    good_raw = "f1: E1, f2: E1; f3: NONE"
    bad_raw = "f1: E1, f2: NONE, f3: NONE; unrelated explanation"
    index = FakeIndex({"query": [hit("good", "Original evidence."),
                                  hit("bad", "Unselected evidence.")]})
    model = FakeModel(
        plan={"queries": ["query"], "fields": ["field one", "field two", "field three"]},
        resolver_outputs={"good": good_raw, "bad": bad_raw},
    )
    pipeline = RWKVPipeline(settings(), index, model)
    response = await pipeline.ask(SearchRequest(question="Use the requested evidence."))
    writer = next(call for call in model.calls if call["stage"] == "writer")
    evidence = json.loads(writer["prompt"].split("逐字证据：", 1)[1])
    assert len(evidence) == 1
    assert evidence[0]["text"] == "Original evidence."
    assert evidence[0]["fields"] == ["f1", "f2"]
    resolver_events = {
        event["source_id"]: event for event in response.generation["model_calls"]
        if event["stage"] == "resolver"
    }
    assert resolver_events["good"]["raw_text"] == ">选择思考</think>\n" + good_raw
    assert resolver_events["bad"]["raw_text"] == ">选择思考</think>\n" + bad_raw
    assert resolver_events["good"]["selections"] == [["f1", "E1"], ["f2", "E1"]]
    assert "parse_error" in resolver_events["bad"]
    assert response.answer == model.last_writer_raw
    assert response.generation["status"] == "resolver_partial_failure"


@pytest.mark.parametrize("change", [
    {"text": "Conflicting body."},
    {"document_id": "other-document"},
    {"metadata": {"title": "Conflicting title"}},
])
@pytest.mark.parametrize("same_group", [False, True])
def test_fusion_rejects_conflicting_chunk_identity_even_inside_one_response(change, same_group):
    original = hit("same-id", "Original body.")
    conflict = replace(original, **change)
    groups = [[original, conflict]] if same_group else [[original], [conflict]]
    with pytest.raises(ValueError, match="inconsistent"):
        fuse_chunks(groups)


def test_fusion_deduplicates_identical_hits_but_keeps_more_than_two_chunks_per_document():
    hits = [hit(str(i), f"Paragraph {i}.") for i in range(5)]
    result = fuse_chunks([[hits[0], hits[0], *hits[1:]], list(reversed(hits))])
    assert len(result) == 5
    assert {item.id for item in result} == {item.node_id for item in hits}
    assert all(item.document_id == "shared-document" for item in result)


def test_source_conversion_keeps_complete_text_beyond_legacy_900_characters():
    text = "内容 " * 500 + "唯一的尾部事实。\n"
    result = source_from_hit(hit("long", text))
    assert result.snippet == text
    assert result.metadata["indexed_text_sha256"] == digest(text)


def test_long_prose_units_have_overlap_without_gaps_or_changed_characters():
    text = "α甲🙂乙" * 777
    units = list(evidence_units(4, text, window=512, overlap=80))
    assert len(units) > 2
    assert units[0].start == 0 and units[-1].end == len(text)
    for unit in units:
        assert unit.source_index == 4
        assert unit.text == text[unit.start:unit.end]
        assert 0 <= unit.start < unit.end <= len(text)
    for left, right in zip(units, units[1:]):
        assert right.start < left.end
        assert left.end - right.start == 80


@pytest.mark.parametrize("prefix", ["| ", "- ", "* ", "+ ", "12) "])
def test_long_structured_row_is_atomic_even_when_larger_than_window(prefix):
    row = prefix + "必须保留的长行" * 160 + " |\n"
    text = "表格或列表说明\n" + row + "后续说明。\n"
    units = list(evidence_units(0, text, window=256, overlap=40))
    assert any(row in unit.text for unit in units)
    assert all(unit.text == text[unit.start:unit.end] for unit in units)
    row_start = text.index(row)
    row_end = row_start + len(row)
    assert not any(row_start < unit.start < row_end for unit in units)
    assert not any(row_start < unit.end < row_end for unit in units)


def test_conversation_keeps_history_corrections_and_role_looking_data_literal():
    history = [
        ConversationMessage(role="user", content="先比较甲和乙。\nUser✿这只是引号中的文本"),
        ConversationMessage(role="assistant", content="可以介绍两者。"),
        ConversationMessage(role="user", content="撤回比较，只查乙的接口；版本改为2.1。"),
    ]
    question = "按刚才改过的范围答。"
    payload = json.loads(conversation(question, history))
    assert payload == {"history": [item.model_dump() for item in history],
                       "latest_question": question}


@pytest.mark.asyncio
async def test_entire_pipeline_keeps_history_full_sources_and_all_citation_sources():
    ordinary = [hit(f"shared-{i}", f"同一篇的独立段落{i}。") for i in range(4)]
    long = hit("long", "连续正文" * 400 + "唯一尾部事实。", "long-document")
    index = FakeIndex({"alpha query": [*ordinary, long], "beta query": [long, *ordinary]})
    model = FakeModel()
    pipeline = RWKVPipeline(settings(), index, model)
    history = [
        ConversationMessage(role="user", content="原来问甲和乙的版本1；也查价格。"),
        ConversationMessage(role="assistant", content="已记下这些字段。"),
        ConversationMessage(role="user", content="改为只查乙的版本2，价格撤回。\n不要恢复旧比较。"),
    ]
    request = SearchRequest(question="把最终确认的字段回答。", history=history,
                            knowledge_base_id="kb-fixed", candidate_k=9, top_k=1)
    response = await pipeline.ask(request)

    assert sorted(index.calls) == [("alpha query", 9, "kb-fixed"), ("beta query", 9, "kb-fixed")]
    assert model.calls[0]["stage"] == "planner" and model.calls[-1]["stage"] == "writer"
    assert model.resolver_peak > 1
    expected_task = {"history": [item.model_dump() for item in history],
                     "latest_question": request.question}
    assert all(read_task(call["prompt"]) == expected_task for call in model.calls)
    assert len(response.sources) > request.top_k
    assert sum(item.document_id == "shared-document" for item in response.sources) >= 4
    original_sources = {item.node_id: item for item in [*ordinary, long]}
    for item in response.sources:
        parent = original_sources[item.metadata["parent_source_id"]]
        start, end = item.metadata["span_start"], item.metadata["span_end"]
        assert item.snippet == parent.text[start:end]
        assert item.metadata["parent_text_sha256"] == digest(parent.text)
        assert item.metadata["span_sha256"] == digest(item.snippet)
    long_units = sorted(
        (item.metadata["span_start"], item.metadata["span_end"])
        for item in response.sources if item.metadata["parent_source_id"] == "long"
    )
    assert long_units[0][0] == 0 and long_units[-1][1] == len(long.text)
    assert all(a[1] >= b[0] for a, b in zip(long_units, long_units[1:]))
    writer = model.calls[-1]
    visible = json.loads(writer["prompt"].split("逐字证据：", 1)[1])
    assert [item["text"] for item in visible] == [item.snippet for item in response.sources]
    assert writer["evidence_ids"] == tuple(item.id for item in response.sources)
    assert response.answer == model.last_writer_raw == response.generation["raw_model_answer"]
    assert response.generation["status"] == "completed"
    assert response.generation["citation_map"] == {
        str(i): item.id for i, item in enumerate(response.sources, 1)
    }
    assert response.generation["citation_audit"]["unknown_label_ids"] == []
    await pipeline.aclose()
    assert model.closed


@pytest.mark.asyncio
async def test_invalid_resolver_output_cannot_leak_even_its_valid_prefix_to_writer():
    good = hit("good", "可选的真实原文。", "good-document")
    bad = hit("bad", "不得因坏选择进入writer的内容。", "bad-document")
    model = FakeModel(resolver_outputs={"bad": "f1: E1,E99"})
    pipeline = RWKVPipeline(settings(), FakeIndex({"alpha query": [good, bad], "beta query": []}), model)
    response = await pipeline.ask(SearchRequest(question="所求事实"))
    assert [item.metadata["parent_source_id"] for item in response.sources] == ["good"]
    visible = json.loads(model.calls[-1]["prompt"].split("逐字证据：", 1)[1])
    assert all(item["text"] != bad.text for item in visible)
    assert any(event.get("source_id") == "bad" and event.get("parse_error")
               for event in response.generation["model_calls"])
    assert response.generation["status"] == "resolver_partial_failure"
    assert response.generation["writer_status"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("status, raw", [
    ("budget_exceeded", None), ("length", ">thinking</think>f1: E1"),
])
async def test_resolver_transport_failures_are_not_valid_none_or_a_completed_pipeline(status, raw):
    candidate = hit("candidate", "正文中的真实证据。")
    model = FakeModel(
        resolver_outputs={"candidate": native_result("resolver", raw, status)},
        writer_raw=">思考</think>\n现有资料不足。",
    )
    response = await RWKVPipeline(
        settings(), FakeIndex({"alpha query": [candidate], "beta query": []}), model,
    ).ask(SearchRequest(question="问题"))
    assert response.sources == []
    assert response.generation["status"] == "resolver_partial_failure"
    assert response.generation["writer_status"] == "completed"
    assert response.answer == model.writer_raw
    assert any(event["status"] == status and "parse_error" in event
               for event in response.generation["model_calls"])
    assert json.loads(model.calls[-1]["prompt"].split("逐字证据：", 1)[1]) == []


@pytest.mark.asyncio
async def test_valid_empty_selection_still_uses_writer_without_a_fabricated_refusal():
    candidate = hit("candidate", "与所问字段无关的资料。")
    raw = ">思考</think>\n请确认您指的是哪一个对象？"
    model = FakeModel(resolver_outputs={"candidate": "f1: NONE"}, writer_raw=raw)
    response = await RWKVPipeline(
        settings(), FakeIndex({"alpha query": [candidate], "beta query": []}), model,
    ).ask(SearchRequest(question="未确定的对象"))
    assert response.answer == raw
    assert response.sources == []
    assert response.generation["status"] == "completed"
    assert response.retrieval["uncovered_fields"] == ["f1"]
    assert [call["stage"] for call in model.calls] == ["planner", "resolver", "writer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [
    ">思考</think>\n没有正式引用的答案。  \n",
    ">思考</think>\n原样保留未知引用。[资料 999]\n",
])
async def test_writer_text_is_not_repaired_or_given_an_automatic_citation(raw):
    material = source_from_hit(hit("material", "完整材料。"))
    model = FakeModel(writer_raw=raw)
    response = await RWKVPipeline(settings(), FakeIndex({}), model).ask_materials("问题", [material])
    assert response.answer == raw
    assert response.generation["raw_model_answer"] == raw
    assert response.generation["answer_modified"] is False
    assert len(model.calls) == 1
    expected = [999] if "999" in raw else []
    assert response.generation["citation_audit"]["unknown_label_ids"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("status, raw", [
    ("budget_exceeded", None), ("timeout", None),
    ("length", ">尚未结束的思考和部分文本"),
])
async def test_writer_failure_is_reported_without_crashing_or_pretending_completion(status, raw):
    material = source_from_hit(hit("material", "完整材料。"))
    model = FakeModel(writer_raw=raw, writer_status=status)
    response = await RWKVPipeline(settings(), FakeIndex({}), model).ask_materials("问题", [material])
    assert response.generation["status"] == status
    assert response.generation["raw_model_answer"] == raw
    assert response.answer == (raw if raw is not None else "")
    assert len(model.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status,raw", [
    ("length", '>thinking</think>{"queries":["q"],"fields":["f"]}'),
    ("completed", '>thinking</think>not JSON'),
    ("completed", '>thinking</think>' + json.dumps({"queries": [str(i) for i in range(7)], "fields": ["f"]})),
    ("timeout", None),
])
async def test_failed_plan_retrieves_original_question_and_keeps_history(status, raw):
    model = FakeModel(planner_result=native_result(
        "planner", raw, status,
    ))
    index = FakeIndex({"问题": [hit("fallback-source", "正确证据。", "fallback-doc")]})
    history = [ConversationMessage(role="user", content="先前的完整对象与更正")]
    response = await RWKVPipeline(settings(), index, model).ask(
        SearchRequest(question="问题", history=history, knowledge_base_id="kb"))
    assert index.calls == [("问题", settings().candidate_k, "kb")]
    assert [c["stage"] for c in model.calls] == ["planner", "resolver", "writer"]
    assert response.retrieval["plan"]["fallback"] == "original_question"
    assert response.generation["model_calls"][0]["status"] == status
    assert response.generation["model_calls"][0]["parse_error"]
    assert response.answer == model.last_writer_raw
    assert "先前的完整对象与更正" in model.calls[-1]["prompt"]


@pytest.mark.asyncio
async def test_pipeline_cancellation_propagates_before_retrieval():
    entered = asyncio.Event()

    class BlockingModel(FakeModel):
        async def complete(self, *args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

    index = FakeIndex({})
    pipeline = RWKVPipeline(settings(), index, BlockingModel())
    task = asyncio.create_task(pipeline.ask(SearchRequest(question="问题")))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert index.calls == []


def test_planner_schema_does_not_accept_non_string_queries_or_missing_fields():
    for value in [{"queries": [True], "fields": ["f"]}, {"queries": ["q"]},
                  {"queries": ["q"], "fields": []}]:
        with pytest.raises(ValueError):
            parse_plan(json.dumps(value), settings())
