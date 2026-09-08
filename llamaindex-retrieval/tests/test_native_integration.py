from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.dependencies import repository, search_service
from llamaindex_retrieval.lexical_index import LexicalIndex
from llamaindex_retrieval.native_rwkv import NativeRWKVResult
from llamaindex_retrieval.routers.public import router
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline, structured_body
from llamaindex_retrieval.schemas import AskResponse, SearchRequest, SourceItem
from llamaindex_retrieval.service import SearchService


def source(identity="a", text="原文", title="文档"):
    return SourceItem(id=identity, document_id="doc", source="fixture", title=title,
                      score=1, snippet=text)


def test_bm25_keeps_full_text_and_six_chunks_from_same_document():
    class Client:
        def search(self, *, index, body):
            assert index == "isolated-wiki"
            assert body["size"] == 80
            assert body["query"]["bool"]["filter"] == [
                {"term": {"knowledge_base_id": "tenant"}}]
            assert "collapse" not in body
            return {"hits": {"hits": [{"_score": 10 - i, "_source": {
                "node_id": str(i), "document_id": "doc", "text": "原文 " * 1000,
                "metadata": {"title": "原始标题"}}} for i in range(6)]}}

    index = object.__new__(LexicalIndex)
    index.client, index.index_name = Client(), "isolated-wiki"
    results = index.search_chunks("文档", candidate_k=80, knowledge_base_id="tenant")
    assert len(results) == 6
    assert all(item.text == "原文 " * 1000 for item in results)


async def test_material_conflicting_identity_is_rejected_before_model():
    model = AsyncMock()
    pipeline = RWKVPipeline(Settings(), None, model)
    response = await pipeline.ask_materials("问题", [source(), source(title="另一页")])
    assert response.generation["status"] == "invalid_materials"
    assert response.answer == ""
    model.complete.assert_not_awaited()


async def test_citation_audit_does_not_promote_thinking_labels():
    raw = ">考虑[资料 9]</think>原样回答[资料 1]"
    model = AsyncMock()
    model.complete.return_value = NativeRWKVResult("completed", raw, "stop", {})
    response = await RWKVPipeline(Settings(), None, model).ask_materials("问题", [source()])
    assert response.answer == raw
    assert response.generation["citation_audit"]["label_ids"] == [1]
    assert response.generation["citation_audit"]["unknown_label_ids"] == []


async def test_index_error_preserves_completed_planner_trace():
    class Index:
        def search_chunks(self, *args, **kwargs):
            raise RuntimeError("index unavailable")

    model = AsyncMock()
    raw = '>思考</think>{"queries":["关键词"],"fields":["字段"]}'
    model.complete.return_value = NativeRWKVResult("completed", raw, "stop", {"raw": raw})
    response = await RWKVPipeline(Settings(), Index(), model).ask(SearchRequest(question="问题"))
    assert response.generation["status"] == "retrieval_failed"
    assert response.generation["model_calls"] == [{"raw": raw}]
    assert model.complete.await_count == 1


async def test_service_dispatch_and_client_shutdown():
    service = SearchService(Settings(rag_pipeline="rwkv"), None)
    original = service.native_pipeline
    await original.aclose()
    native = AsyncMock()
    native.ask.return_value = AskResponse(answer="raw", sources=[], retrieval={}, generation={})
    service.native_pipeline = native
    request = SearchRequest(question="问题", history=[{"role": "user", "content": "历史"}])
    assert (await service.ask(request)).answer == "raw"
    native.ask.assert_awaited_once_with(request)
    await service.aclose()
    native.aclose.assert_awaited_once()


@pytest.mark.parametrize("endpoint,extra", [
    ("/v1/ask", {}), ("/v1/material-ask", {"materials": [source().model_dump()]}),
])
def test_api_history_and_raw_trace_are_persisted(endpoint, extra):
    app = FastAPI()
    app.include_router(router)
    service, repo = AsyncMock(), AsyncMock()
    response = AskResponse(answer=">思考</think>回答", sources=[source()], retrieval={},
                           generation={"output_mode": "immutable"})
    service.ask.return_value = response
    service.ask_materials.return_value = response
    app.dependency_overrides[search_service] = lambda: service
    app.dependency_overrides[repository] = lambda: repo
    payload = {"question": "现在只问这一项", "history": [
        {"role": "user", "content": "先比较两项"}], **extra}
    result = TestClient(app).post(endpoint, json=payload)
    assert result.status_code == 200
    assert result.json()["answer"] == response.answer
    args = repo.record_search_test_run.await_args.args
    assert args[0]["history"] == payload["history"]
    assert args[1]["answer"] == response.answer


@pytest.mark.parametrize("prefill,raw", [
    ("<think", '>思考</think>{"queries":["问题"],"fields":["字段"]}'),
    ("<think></think", '>{"queries":["问题"],"fields":["字段"]}'),
])
def test_structured_body_uses_actual_prefill_without_changing_raw(prefill, raw):
    result = NativeRWKVResult("completed", raw, "stop", {"prefill": prefill})
    assert structured_body(result) == '{"queries":["问题"],"fields":["字段"]}'
    assert result.raw_text == raw
    with pytest.raises(ValueError, match="did not complete"):
        structured_body(NativeRWKVResult("length", raw, "length", {"prefill": prefill}))


def test_structured_body_rejects_closed_answer_with_missing_or_wrong_prefill():
    raw = '>{"queries":["问题"],"fields":["字段"]}'
    for trace in ({}, {"prefill": "<think"}, {"prefill": "invalid"}):
        with pytest.raises(ValueError, match="thinking envelope"):
            structured_body(NativeRWKVResult("completed", raw, "stop", trace))


def test_planner_prefill_default_and_strict_values():
    from pydantic import ValidationError

    assert Settings().native_planner_prefill == "<think"
    assert Settings(native_planner_prefill="<think></think").native_planner_prefill == "<think></think"
    with pytest.raises(ValidationError):
        Settings(native_planner_prefill="<think></think>")


async def test_only_planner_prefill_changes_actual_native_wire():
    import json
    import httpx
    from llamaindex_retrieval.native_rwkv import NativeRWKVClient

    recorded = []

    def respond(request):
        payload = json.loads(request.content)
        recorded.append((request.url.path, payload))
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 1, "tokens": [0], "max_model_len": 16384})
        raw = '>完成' if payload["prompt"].endswith("<think></think") else '>思考</think>完成'
        return httpx.Response(200, json={"choices": [{"text": raw, "finish_reason": "stop"}],
                                       "usage": {"prompt_tokens": 1, "completion_tokens": 2}})

    for prefill in ("<think", "<think></think"):
        client = NativeRWKVClient(base_url="http://fixture/v1", model="fixed-model",
                                  transport=httpx.MockTransport(respond))
        pipeline = RWKVPipeline(Settings(native_planner_prefill=prefill), None, model=client)
        for stage in ("planner", "resolver", "writer"):
            result = await pipeline._call("原始固定任务\n完整历史和原文", stage=stage,
                                          max_tokens=2048 if stage == "writer" else 1024)
            assert result.status == "completed"
            assert result.trace["prefill"] == (prefill if stage == "planner" else "<think")
        await client.aclose()
    assert len(recorded) == 12
    before, after = recorded[:6], recorded[6:]
    for i, ((path1, payload1), (path2, payload2)) in enumerate(zip(before, after)):
        assert path1 == path2
        if i < 2:  # planner tokenize + completion have the same one-variable prompt change.
            assert payload2["prompt"] == payload1["prompt"] + "></think"
            assert {k: v for k, v in payload1.items() if k != "prompt"} == {
                k: v for k, v in payload2.items() if k != "prompt"}
        else:
            assert payload1 == payload2
