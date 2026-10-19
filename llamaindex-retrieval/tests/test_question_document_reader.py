"""Application contracts and scheduling, not a model accuracy benchmark."""
import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.model_client import model_client_class, model_client_options
from llamaindex_retrieval.native_rwkv import NativeRWKVClient, NativeRWKVResult
from llamaindex_retrieval.question_document_reader import PROTOCOL
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline
from llamaindex_retrieval.schemas import SourceItem
from llamaindex_retrieval.structured_native import StructuredNativeRWKVClient


def settings(**overrides):
    values = dict(_env_file=None, rag_pipeline="rwkv", generation_output_mode="immutable",
        native_transport="native", native_completion_protocol="g1j_plain",
        native_require_model_identity=True, native_material_reading="question_documents_v1",
        native_planner_prefill="<think></think", native_resolver_prefill="<think></think",
        native_writer_prefill="<think></think")
    return Settings(**{**values, **overrides})


def sources(count=3):
    return [SourceItem(id=f"D{i}", document_id=f"doc{i}", source="provided", title=f"题名{i}",
        score=0, snippet=f"  原文{i}\n| 对象 | 条件 |\n| 甲 | A{i} |\n", metadata={"version": i})
        for i in range(count)]


class Model:
    def __init__(self, overrides=None, delay=.001):
        self.overrides = overrides or {}
        self.delay = delay
        self.calls = []
        self.active = 0
        self.peak = 0
        self.finished = []

    async def complete(self, messages, **kwargs):
        data = json.loads(messages[0]["content"].split("\n", 1)[1])
        reader = kwargs["stage"] == "reader"
        key = data["source"]["id"] if reader else "final"
        self.calls.append((key, data, kwargs))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay * (4 - int(key[1:]) % 3) if reader else self.delay)
            if key in self.overrides:
                replacement = self.overrides[key]
                if isinstance(replacement, Exception):
                    raise replacement
                status, raw = replacement
            elif reader:
                status, raw = "completed", json.dumps({"findings": [
                    {"source_id": key, "statement": "MODEL NOTE, NOT ORIGINAL"}]})
            else:
                # Whitespace must survive application response projection.
                status, raw = "completed", ' {"answer":"模型正文", "source_ids":[]} \n'
            self.finished.append(key)
            return NativeRWKVResult(status, raw, "stop" if status == "completed" else status,
                                    kwargs.get("trace", {}))
        finally:
            self.active -= 1

    async def aclose(self):
        pass


async def test_all_documents_queue_and_selected_originals_keep_input_order():
    model = Model()
    docs = sources(40)
    before = [d.model_dump() for d in docs]
    pipeline = RWKVPipeline(settings(native_document_read_concurrency=4), None, model=model)
    result = await pipeline.ask_materials("完整问题", docs)
    assert model.peak == 4 and model.active == 0
    assert len(model.calls) == 41 and model.finished[-1] == "final"
    assert model.finished[:4] != ["D0", "D1", "D2", "D3"]
    final = model.calls[-1][1]
    assert final == {"question": "完整问题", "sources": [
        {"id": d.id, "title": d.title, "text": d.snippet} for d in docs]}
    assert "MODEL NOTE" not in json.dumps(final)
    assert [d.model_dump() for d in docs] == before
    assert [d.id for d in result.sources] == [d.id for d in docs]
    assert result.answer == ' {"answer":"模型正文", "source_ids":[]} \n'
    assert result.generation["raw_model_answer"] == result.answer
    assert result.generation["status"] == "completed"
    assert result.generation["protocol"] == PROTOCOL
    assert len({e["call_id"] for e in result.generation["model_calls"]}) == 41


@pytest.mark.parametrize("bad", [
    ("completed", '{"findings":[{"source_id":"D2","statement":"wrong source"}]}'),
    ("completed", '{"findings":[],"findings":[{"source_id":"D0","statement":"x"}]}'),
    ("completed", '{"findings":[{"source_id":"D0","statement":4}]}'),
    ("length", '{"findings":[{"source_id":"D0","statement":"unfinished"}]}'),
    ("completed", 'FAILED_READER_BODY'),
    RuntimeError("connection ended"),
])
async def test_rejected_reader_never_adds_original_or_failed_text_to_writer(bad):
    model = Model({"D0": bad, "D1": ("completed", '{"findings":[]}')})
    result = await RWKVPipeline(settings(), None, model=model).ask_materials("问题", sources())
    assert result.retrieval["selected_ids"] == ["D2"]
    assert result.generation["status"] == "partial"
    final = model.calls[-1][1]
    assert len(final["sources"]) == 1 and final["sources"][0]["id"] == "D2"
    assert final["sources"][0]["text"] == sources()[2].snippet
    assert "FAILED_READER_BODY" not in json.dumps(final)
    event = result.generation["model_calls"][0]
    assert event["parsed"] is None
    if not isinstance(bad, Exception):
        assert event["raw_text"] == bad[1]


@pytest.mark.parametrize("status,raw,expected", [
    ("length", '{"answer":"unfinished', "length"),
    ("completed", '{"answer":"x","source_ids":["outside"]}', "invalid_output"),
    ("completed", 'unstructured model answer', "invalid_output"),
])
async def test_final_failure_is_retained_verbatim_without_retry(status, raw, expected):
    model = Model({"final": (status, raw)})
    result = await RWKVPipeline(settings(), None, model=model).ask_materials("问题", sources(1))
    assert result.answer == raw and result.generation["raw_model_answer"] == raw
    assert result.generation["status"] == expected and len(model.calls) == 2
    assert result.generation["parsed_answer"] is None


async def test_empty_selection_still_gets_independent_writer_without_source_fallback():
    model = Model({"D0": ("completed", '{"findings":[]}')})
    result = await RWKVPipeline(settings(), None, model=model).ask_materials("问题", sources(1))
    assert model.calls[-1][1]["sources"] == [] and result.sources == []
    assert model.calls[-1][2]["structured_schema"]["properties"]["source_ids"]["maxItems"] == 0


async def test_explicit_empty_evidence_failure_skips_writer():
    model = Model({"D0": ("completed", '{"findings":[]}')})
    result = await RWKVPipeline(settings(native_empty_evidence_policy="fail"), None,
                               model=model).ask_materials("问题", sources(1))
    assert result.generation["status"] == "no_evidence" and result.answer == ""
    assert len(model.calls) == 1


async def test_timeout_cancels_all_readers_and_never_starts_writer():
    model = Model(delay=10)
    pipeline = RWKVPipeline(settings(native_request_timeout_seconds=.02,
        native_document_read_concurrency=4), None, model=model)
    result = await pipeline.ask_materials("问题", sources(12))
    assert result.generation["status"] == "request_timeout"
    assert model.active == 0 and len(model.calls) == 4
    assert all(key != "final" for key, _, _ in model.calls)
    assert result.answer == ""


async def test_call_budget_rejects_whole_request_instead_of_dropping_documents():
    model = Model()
    result = await RWKVPipeline(settings(native_request_max_calls=3), None,
                               model=model).ask_materials("问题", sources(3))
    assert result.generation["status"] == "call_budget_exceeded"
    assert model.calls == []


async def test_history_and_duplicate_ids_are_explicitly_rejected_before_dispatch():
    model = Model()
    pipeline = RWKVPipeline(settings(), None, model=model)
    history = await pipeline.ask_materials("问题", sources(), history=[{"role": "user", "content": "以前"}])
    duplicate = await pipeline.ask_materials("问题", [sources()[0], sources()[0]])
    assert history.generation["status"] == "unsupported_history"
    assert duplicate.generation["status"] == "invalid_materials"
    assert model.calls == []


async def test_overlapping_requests_do_not_share_selections_or_trace_records():
    model = Model()
    pipeline = RWKVPipeline(settings(native_request_max_calls=4), None, model=model)
    a, b = await asyncio.gather(pipeline.ask_materials("甲", sources(3)),
                                pipeline.ask_materials("乙", sources(2)))
    assert a.retrieval["selected_ids"] == ["D0", "D1", "D2"]
    assert b.retrieval["selected_ids"] == ["D0", "D1"]
    assert {c["call_id"] for c in a.generation["model_calls"]}.isdisjoint(
        c["call_id"] for c in b.generation["model_calls"])
    assert a.generation["request_budget"]["admitted_calls"] == 4
    assert b.generation["request_budget"]["admitted_calls"] == 3


def test_default_client_and_protocol_are_unchanged():
    default = Settings(_env_file=None)
    assert default.native_material_reading == "direct"
    assert model_client_class(default) is NativeRWKVClient
    assert "transport" not in model_client_options(default)
    assert model_client_class(settings()) is StructuredNativeRWKVClient


def test_documented_environment_profile_loads_with_integer_concurrency(monkeypatch):
    from pathlib import Path
    doc = Path(__file__).parents[1] / "QUESTION_DOCUMENT_READING.md"
    block = doc.read_text().split("```dotenv\n", 1)[1].split("```", 1)[0]
    for line in block.strip().splitlines():
        name, value = line.split("=", 1)
        monkeypatch.setenv(name, value)
    profile = Settings(_env_file=None)
    assert profile.native_material_reading == "question_documents_v1"
    assert profile.native_document_read_concurrency == 32


@pytest.mark.parametrize("changes", [
    {"native_transport": "rwkvos_batch"}, {"native_completion_protocol": "native"},
    {"native_require_model_identity": False}, {"native_writer_pipeline": "funnel_v1"},
    {"native_task_contract": "anchored_v1"}, {"native_answer_quality_policy": "fail_citation"},
    {"native_writer_budget_policy": "whole_sources"},
    {"native_document_read_concurrency": 0}, {"native_document_read_concurrency": True},
    {"native_document_read_concurrency": 33},
])
def test_incompatible_configurations_fail_at_startup(changes):
    with pytest.raises(ValidationError):
        settings(**changes)


async def test_native_http_boundary_uses_reader_and_writer_schemas_and_preserves_raw():
    cfg = settings()
    completions = []
    original = sources(1)[0]
    final_raw = '  {"answer":"边界回答[D0]","source_ids":["D0"]}\n'

    def handle(request):
        payload = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 1, "tokens": [11], "max_model_len": 16384})
        completions.append(payload)
        raw = (json.dumps({"findings": [{"source_id": "D0", "statement": "派生记录"}]})
               if len(completions) == 1 else final_raw)
        return httpx.Response(200, json={"model": cfg.native_model, "choices": [{
            "text": raw, "finish_reason": "stop", "prompt_token_ids": [11], "token_ids": [12],
        }], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    client = model_client_class(cfg)(**model_client_options(cfg, transport=httpx.MockTransport(handle)))
    pipeline = RWKVPipeline(cfg, None, model=client)
    try:
        result = await pipeline.ask_materials("完整问题", [original])
    finally:
        await pipeline.aclose()
    assert result.generation["status"] == "completed" and result.answer == final_raw
    assert len(completions) == 2
    assert [list(c["structured_outputs"]["json"]["properties"]) for c in completions] == [
        ["findings"], ["answer", "source_ids"]]
    assert [c["max_tokens"] for c in completions] == [768, 2048]
    assert all(c["temperature"] == 0 and c["seed"] == 11 for c in completions)
    assert "派生记录" not in completions[-1]["prompt"]
    assert [e["evidence_ids"] for e in result.generation["model_calls"]] == [["D0"], ["D0"]]
