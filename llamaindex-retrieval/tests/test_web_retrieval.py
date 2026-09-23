import asyncio
from dataclasses import replace
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import SecretStr

from llamaindex_retrieval import direct_web
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.repository import search_answer_status, search_failure_category
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline
from llamaindex_retrieval.schemas import SearchRequest
from llamaindex_retrieval.web_retrieval import SearchReaderAdapter, deduplicate_web_groups
from test_rwkv_pipeline import FakeIndex, FakeModel, hit, settings


def web_hit():
    source = hit("web:1", "Public release date is 2024-10-07", "web:document")
    source.metadata.update(source="web", retrieved_at="2026-09-19T00:00:00Z")
    return source


@pytest.mark.asyncio
async def test_default_web_provider_uses_this_repository_and_requires_credential():
    config = Settings(_env_file=None)
    assert config.web_search_provider == "tavily"
    with pytest.raises(direct_web.WebProviderError,
                       match="web_tavily_key_not_configured"):
        await SearchReaderAdapter(config).search("query")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,local_calls,web_calls", [
    ("knowledge_base", 1, 0), ("hybrid", 1, 1), ("web", 0, 1),
])
async def test_scope_evidence_and_immutable_answer(mode, local_calls, web_calls):
    model = FakeModel(plan={"queries": ["release"], "fields": ["release", "date"]})
    index = FakeIndex({"release": [hit("private:1", "Internal upgrade is next Monday")]})
    pipeline = RWKVPipeline(settings(), index, model)
    pipeline.web.search = AsyncMock(return_value=([web_hit()], {"status": "completed", "snapshots": []}))
    response = await pipeline.ask(SearchRequest(question="release", retrieval_mode=mode, knowledge_base_id="team-a"))
    assert len(index.calls) == local_calls
    assert pipeline.web.search.await_count == web_calls
    assert response.answer == model.last_writer_raw
    assert response.generation["status"] == "completed"
    origins = {s.metadata["retrieval_origin"] for s in response.sources}
    assert origins == ({"knowledge_base", "web"} if mode == "hybrid" else {mode})
    if local_calls:
        assert index.calls[0][2] == "team-a"
    if web_calls:
        pipeline.web.search.assert_awaited_once_with("release")
    assert len(response.generation["citation_map"]) == len(response.sources)


@pytest.mark.asyncio
async def test_web_timeout_retains_kb_and_marks_partial_not_completed():
    pipeline = RWKVPipeline(settings(), FakeIndex({"q": [hit("kb", "Local evidence")]}),
        FakeModel(plan={"queries": ["q"], "fields": ["q", "f"]}))
    pipeline.web.search = AsyncMock(side_effect=TimeoutError("secret-key-must-not-be-exposed"))
    result = await pipeline.ask(SearchRequest(question="q", retrieval_mode="hybrid"))
    assert result.generation["status"] == "retrieval_partial_failure"
    assert search_answer_status(result.model_dump()) == "partial"
    assert search_failure_category(result.model_dump()) == "retrieval_failed"
    assert result.sources and result.answer == pipeline.model.last_writer_raw
    assert "secret-key-must-not-be-exposed" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_both_failed_never_calls_reader_or_writer():
    model = FakeModel(plan={"queries": ["q"], "fields": ["q", "f"]})
    pipeline = RWKVPipeline(settings(), FakeIndex({}), model)
    pipeline.web.search = AsyncMock(side_effect=RuntimeError("offline"))
    result = await pipeline.ask(SearchRequest(question="q", retrieval_mode="hybrid"))
    assert result.generation["status"] == "retrieval_failed"
    assert [c["stage"] for c in model.calls] == ["planner"]
    assert result.retrieval["all_providers_failed"]


@pytest.mark.asyncio
async def test_knowledge_failure_still_uses_web():
    pipeline = RWKVPipeline(settings(), FakeIndex({}), FakeModel(plan={"queries": ["q"], "fields": ["q", "f"]}))
    pipeline.web.search = AsyncMock(return_value=([web_hit()], {"status": "completed"}))
    result = await pipeline.ask(SearchRequest(question="q", retrieval_mode="hybrid"))
    assert result.generation["status"] == "retrieval_partial_failure"
    assert {s.metadata["retrieval_origin"] for s in result.sources} == {"web"}


@pytest.mark.asyncio
async def test_per_query_reader_budget_keeps_both_providers():
    pipeline = RWKVPipeline(settings(native_resolver_budget_scope="per_query", native_task_source="queries",
        native_resolver_task_grouping="individual", native_resolver_protocol="task_units"),
        FakeIndex({"q": [hit(str(i), "Local material") for i in range(30)]}),
        FakeModel(plan={"queries": ["q"], "fields": ["q", "f"]}))
    pipeline.web.search = AsyncMock(return_value=([web_hit()], {"status": "completed"}))
    pipeline._resolve = AsyncMock(return_value=([], []))
    response = await pipeline.ask(SearchRequest(question="q", retrieval_mode="hybrid"))
    assert "web:1" in response.retrieval["resolver_source_ids"]
    assert response.retrieval["resolver_source_tasks"]["web:1"] == [["q"]]


def project(tmp_path, search_code):
    package = tmp_path / "src/rwkv_search_reader"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "config.py").write_text("from dataclasses import dataclass\n@dataclass\nclass Settings:\n max_evidence_chars:int=100\ndef get_settings(): return Settings()\n")
    (package / "retrieval.py").write_text("def normalize_url(url): return url\ndef fetch_page(hit, settings): return hit\n")
    (package / "models.py").write_text("")
    (package / "search.py").write_text(search_code)
    bindir = tmp_path / ".venv/bin"
    bindir.mkdir(parents=True)
    (bindir / "python").symlink_to(sys.executable)
    return tmp_path


@pytest.mark.asyncio
async def test_real_subprocess_protocol_preserves_snapshots(tmp_path):
    root = project(tmp_path, '''
from dataclasses import dataclass
@dataclass
class Hit:
 title:str="A"; url:str="https://example.org/a"; content:str="exact text"
 snippet:str="short"; content_markdown:str="exact text plus full snapshot"
 error:str=""; retrieved_at:str="2026-09-19T00:00:00Z"; published_date:str="2024-10-07"
 content_status:str="fetched"; score:float=0.5
class Provider:
 name="fixture"; results_are_material=True
 def __init__(self): self.session=self
 def close(self): pass
 def search(self, query, max_results): return [Hit()]
def create_search_provider(settings): return Provider()
''')
    adapter = SearchReaderAdapter(settings(web_search_provider="searchreader", searchreader_project_dir=root))
    hits, trace = await adapter.search("query")
    assert hits[0].text == "exact text"
    assert hits[0].metadata["material_limited"]
    assert trace["snapshots"][0]["text"] == "exact text plus full snapshot"
    assert len(trace["source_hashes"]) == 4
    assert hits[0].metadata["snapshot_sha256"] == trace["snapshots"][0]["sha256"]


@pytest.mark.asyncio
async def test_timeout_kills_bridge_and_releases_slot(tmp_path):
    root = project(tmp_path, "import time\ntime.sleep(20)\n")
    adapter = SearchReaderAdapter(settings(web_search_provider="searchreader", searchreader_project_dir=root, web_search_timeout=1, web_search_concurrency=1))
    with pytest.raises(TimeoutError):
        await adapter.search("q")
    # Semaphore is released even when process termination is needed.
    async with asyncio.timeout(1):
        async with adapter.slots:
            pass


@pytest.mark.asyncio
async def test_provider_error_body_is_not_exposed(tmp_path):
    root = project(tmp_path, 'raise RuntimeError("private API credential")\n')
    adapter = SearchReaderAdapter(settings(web_search_provider="searchreader", searchreader_project_dir=root))
    with pytest.raises(RuntimeError, match="^searchreader_provider_failed$"):
        await adapter.search("q")


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["tavily", "searxng"])
async def test_single_repository_web_provider_preserves_source_snapshot(monkeypatch, provider):
    def handler(request):
        if provider == "tavily":
            assert request.headers["Authorization"] == "Bearer test-secret"
            assert request.url.path == "/search"
            return httpx.Response(200, json={"results": [
                {"url": "https://example.org/a", "title": "A", "content": "summary",
                 "raw_content": "verbatim original page", "score": 0.8},
                {"url": "file:///private", "content": "discard"}]})
        assert request.url.params["format"] == "json"
        return httpx.Response(200, json={"results": [
            {"url": "https://example.org/a", "title": "A", "content": "verbatim snippet"}]})

    monkeypatch.setattr(direct_web, "_client", lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler), trust_env=False))
    config = SimpleNamespace(web_search_provider=provider, web_search_concurrency=1,
        web_search_timeout=3, web_search_results=2, web_search_material_characters=8,
        web_tavily_api_key=SecretStr("test-secret"),
        web_searxng_base_url="http://127.0.0.1:8888",
        searchreader_router_base_url=None, searchreader_router_model=None,
        searchreader_router_state_sha256=None)
    hits, trace = await SearchReaderAdapter(config).search("natural words")
    assert len(hits) == 1
    expected = "verbatim original page" if provider == "tavily" else "verbatim snippet"
    assert hits[0].text == expected[:8]
    assert trace["snapshots"][0]["text"] == expected
    assert hits[0].metadata["content_status"] == ("fetched" if provider == "tavily" else "snippet_only")
    assert "test-secret" not in str(trace)


@pytest.mark.asyncio
async def test_single_repository_router_uses_model_output_without_keyword_fallback(monkeypatch):
    def handler(request):
        assert request.url.path == "/v1/completions"
        return httpx.Response(200, json={"choices": [{"text": "true"}]})

    monkeypatch.setattr(direct_web, "_client", lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler), trust_env=False))
    config = SimpleNamespace(web_search_provider="tavily", web_search_concurrency=1,
        searchreader_router_base_url="http://127.0.0.1:1234/v1",
        searchreader_router_model="state-router", searchreader_router_state_sha256="frozen",
        searchreader_router_timeout=2, web_router_protocol="completions",
        web_router_api_key=SecretStr("router-secret"))
    decision = await SearchReaderAdapter(config).decide([
        {"role": "user", "content": "比一下最新版本"}])
    assert decision["needs_search"] is True
    assert decision["configured_state_sha256"] == "frozen"
    assert decision["evidence_ids"] == []
    assert decision["elapsed_ms"] >= 0
    assert "router-secret" not in str(decision)


@pytest.mark.asyncio
async def test_single_repository_upstream_error_does_not_expose_credential(monkeypatch):
    monkeypatch.setattr(direct_web, "_client", lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(
            401, text="test-secret is invalid")), trust_env=False))
    config = SimpleNamespace(web_search_provider="tavily", web_search_concurrency=1,
        web_search_timeout=3, web_search_results=2, web_search_material_characters=100,
        web_tavily_api_key=SecretStr("test-secret"))
    with pytest.raises(RuntimeError, match="^web_upstream_http_401$") as caught:
        await SearchReaderAdapter(config).search("q")
    assert "test-secret" not in str(caught.value)
    assert caught.value.safe_code == "web_upstream_http_401"


@pytest.mark.asyncio
async def test_shared_guard_stops_auth_retry_across_adapter_instances(monkeypatch, tmp_path):
    upstream_calls = []
    def handler(request):
        upstream_calls.append(request.headers["Authorization"])
        if request.headers["Authorization"] == "Bearer expired-key":
            return httpx.Response(401, text="private provider account response")
        return httpx.Response(200, json={"results": [{
            "url": "https://example.org/a", "content": "new key works"}]})
    monkeypatch.setattr(direct_web, "_client", lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler), trust_env=False))
    guard_path = tmp_path / "guard.sqlite3"
    options = dict(web_search_provider="tavily", web_search_concurrency=2,
        web_search_timeout=3, web_search_results=1, web_guard_path=guard_path,
        web_min_interval_seconds=0, web_search_material_characters=256)
    expired = settings(**options, web_tavily_api_key="expired-key")
    with pytest.raises(direct_web.WebProviderError, match="web_upstream_http_401"):
        await SearchReaderAdapter(expired).search("first")
    with pytest.raises(direct_web.WebProviderError, match="web_upstream_circuit_open_401"):
        await SearchReaderAdapter(expired).search("second")
    assert upstream_calls == ["Bearer expired-key"]
    hits, _ = await SearchReaderAdapter(settings(**options,
        web_tavily_api_key="restored-key")).search("third")
    assert hits[0].text == "new key works"
    assert upstream_calls == ["Bearer expired-key", "Bearer restored-key"]
    assert guard_path.stat().st_mode & 0o077 == 0


@pytest.mark.asyncio
async def test_concurrent_workers_wait_for_first_auth_result(monkeypatch, tmp_path):
    calls = []
    async def handler(request):
        calls.append(request.url.path)
        await asyncio.sleep(0.05)
        return httpx.Response(401, text="private provider account response")
    monkeypatch.setattr(direct_web, "_client", lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler), trust_env=False))
    config = settings(web_search_provider="tavily", web_tavily_api_key="expired-key",
        web_guard_path=tmp_path / "guard.sqlite3", web_min_interval_seconds=0)
    outcomes = await asyncio.gather(
        SearchReaderAdapter(config).search("one"),
        SearchReaderAdapter(config).search("two"), return_exceptions=True)
    assert sorted(str(item) for item in outcomes) == [
        "web_upstream_circuit_open_401", "web_upstream_http_401"]
    assert calls == ["/search"]


@pytest.mark.asyncio
async def test_rate_limit_retry_after_blocks_next_request_without_retry(monkeypatch, tmp_path):
    calls = []
    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(429, headers={"Retry-After": "120"},
                              text="secret provider response")
    monkeypatch.setattr(direct_web, "_client", lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler), trust_env=False))
    config = settings(web_search_provider="tavily", web_tavily_api_key="some-key",
        web_guard_path=tmp_path / "guard.sqlite3", web_min_interval_seconds=0)
    with pytest.raises(direct_web.WebProviderError, match="web_upstream_http_429"):
        await SearchReaderAdapter(config).search("one")
    with pytest.raises(direct_web.WebProviderError, match="web_upstream_circuit_open_429"):
        await SearchReaderAdapter(config).search("two")
    assert calls == ["/search"]


@pytest.mark.asyncio
async def test_private_key_file_is_read_without_exposing_it(monkeypatch, tmp_path):
    key_file = tmp_path / "tavily.key"
    key_file.write_text("file-secret\n")
    key_file.chmod(0o600)
    def handler(request):
        assert request.headers["Authorization"] == "Bearer file-secret"
        return httpx.Response(200, json={"results": []})
    monkeypatch.setattr(direct_web, "_client", lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler), trust_env=False))
    config = settings(web_search_provider="tavily", web_tavily_api_key_file=key_file,
                      web_guard_path=tmp_path / "guard.sqlite3")
    assert (await SearchReaderAdapter(config).search("q"))[0] == []
    key_file.chmod(0o644)
    with pytest.raises(direct_web.WebProviderError,
                       match="web_tavily_key_file_permissions"):
        await SearchReaderAdapter(config).search("q")


@pytest.mark.asyncio
async def test_single_repository_provider_reaches_hybrid_pipeline_without_second_checkout(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={"results": [{
            "url": "https://example.org/release", "title": "Official release",
            "content": "Web release note", "raw_content": "Web release note with full context",
            "score": 0.9}]})

    monkeypatch.setattr(direct_web, "_client", lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler), trust_env=False))
    model = FakeModel(plan={"queries": ["release"], "fields": ["release"]})
    pipeline = RWKVPipeline(settings(web_search_provider="tavily",
        web_tavily_api_key="test-secret", web_search_results=1, web_search_max_queries=1),
        FakeIndex({"release": [hit("private:1", "Private release note")]}), model)
    response = await pipeline.ask(SearchRequest(question="release",
                                                retrieval_mode="hybrid", knowledge_base_id="team-a"))
    assert response.generation["status"] == "completed"
    assert {source.metadata["retrieval_origin"] for source in response.sources} == {
        "web", "knowledge_base"}
    assert response.retrieval["web_search"][0]["snapshots"][0]["text"] == \
        "Web release note with full context"
    assert response.answer == model.last_writer_raw


@pytest.mark.asyncio
async def test_single_repository_provider_failure_exposes_safe_code_in_trace(monkeypatch):
    monkeypatch.setattr(direct_web, "_client", lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(
            401, text="secret-in-provider-body")), trust_env=False))
    pipeline = RWKVPipeline(settings(web_search_provider="tavily",
        web_tavily_api_key="test-secret", web_search_results=1, web_search_max_queries=1),
        FakeIndex({"q": [hit("private:1", "Private evidence")]}),
        FakeModel(plan={"queries": ["q"], "fields": ["q"]}))
    response = await pipeline.ask(SearchRequest(question="q", retrieval_mode="hybrid"))
    assert response.generation["status"] == "retrieval_partial_failure"
    assert response.retrieval["provider_failures"][0]["error_code"] == "web_upstream_http_401"
    assert "secret-in-provider-body" not in response.model_dump_json()


@pytest.mark.asyncio
async def test_searxng_empty_results_with_engine_failures_are_not_no_matches(monkeypatch):
    monkeypatch.setattr(direct_web, "_client", lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
            "results": [], "unresponsive_engines": [["brave", "timeout"]]})),
        trust_env=False))
    config = settings(web_search_provider="searxng",
        web_searxng_base_url="http://127.0.0.1:8888")
    with pytest.raises(direct_web.WebProviderError,
                       match="web_searxng_empty_with_engine_failures"):
        await SearchReaderAdapter(config).search("q")


def test_invalid_scope_rejected():
    with pytest.raises(ValueError):
        SearchRequest(question="q", retrieval_mode="automatic-guessed")


@pytest.mark.asyncio
@pytest.mark.parametrize("needs_search", [True, False])
async def test_auto_routes_with_history_and_records_raw_decision(needs_search):
    pipeline = RWKVPipeline(settings(), FakeIndex({"q": [hit("kb", "Private material")]}),
        FakeModel(plan={"queries": ["q"], "fields": ["q", "f"]}))
    pipeline.web.decide = AsyncMock(return_value={"stage": "routing", "status": "completed",
        "needs_search": needs_search, "raw_text": "true" if needs_search else "false"})
    pipeline.web.search = AsyncMock(return_value=([web_hit()], {"status": "completed"}))
    response = await pipeline.ask(SearchRequest(question="q", retrieval_mode="auto",
        history=[{"role": "user", "content": "Previous question"}]))
    assert pipeline.web.search.await_count == int(needs_search)
    assert response.retrieval["routing"]["selected_mode"] == ("hybrid" if needs_search else "knowledge_base")
    assert response.generation["model_calls"][0]["raw_text"] == ("true" if needs_search else "false")
    pipeline.web.decide.assert_awaited_once_with([
        {"role": "user", "content": "Previous question"}, {"role": "user", "content": "q"}])
    assert "Private material" not in str(pipeline.web.decide.call_args)


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", [
    {"status": "invalid_decision", "needs_search": None, "raw_text": "maybe"},
    {"status": "completed", "needs_search": "false", "raw_text": "false"},
    TimeoutError("unavailable"),
])
async def test_router_failure_never_silently_enables_web_or_calls_writer(decision):
    model = FakeModel()
    pipeline = RWKVPipeline(settings(), FakeIndex({}), model)
    pipeline.web.decide = AsyncMock(**({"side_effect": decision} if isinstance(decision, Exception)
                                      else {"return_value": decision}))
    pipeline.web.search = AsyncMock()
    response = await pipeline.ask(SearchRequest(question="q", retrieval_mode="auto"))
    assert response.generation["status"] == "routing_failed"
    assert response.answer == ""
    assert model.calls == []
    pipeline.web.search.assert_not_awaited()
    assert search_failure_category(response.model_dump()) == "retrieval_failed"


@pytest.mark.asyncio
async def test_manual_mode_bypasses_router():
    pipeline = RWKVPipeline(settings(), FakeIndex({"q": []}),
        FakeModel(plan={"queries": ["q"], "fields": ["q", "f"]}))
    pipeline.web.decide = AsyncMock(side_effect=AssertionError("must not call"))
    await pipeline.ask(SearchRequest(question="q", retrieval_mode="knowledge_base"))
    pipeline.web.decide.assert_not_awaited()


def test_identical_web_material_is_shared_but_changed_content_is_not():
    first, second, changed = web_hit(), web_hit(), web_hit()
    first.metadata["retrieval_query"] = "q1"
    second.metadata["retrieval_query"] = "q2"
    changed = replace(changed, text="Different observed page text")
    groups = deduplicate_web_groups([[first], [second, changed]])
    assert groups[0][0] is groups[1][0]
    assert groups[1][1] is not groups[0][0]
    assert [o["retrieval_query"] for o in groups[0][0].metadata["retrieval_observations"]] == ["q1", "q2"]


@pytest.mark.asyncio
async def test_hybrid_retains_whole_question_when_plan_drops_one_part():
    question = "internal date and public date"
    model = FakeModel(plan={"queries": ["public date"], "fields": ["date"]})
    pipeline = RWKVPipeline(settings(web_search_max_queries=1),
        FakeIndex({question: [hit("kb", "internal date")], "public date": []}), model)
    pipeline.web.search = AsyncMock(return_value=([web_hit()], {"status": "completed"}))
    pipeline._resolve = AsyncMock(return_value=([], []))
    response = await pipeline.ask(SearchRequest(question=question, retrieval_mode="hybrid"))
    assert response.retrieval["plan"]["queries"] == ["public date"]
    assert response.retrieval["retrieval_queries"] == [question, "public date"]
    assert pipeline._resolve.call_args.args[1] == [question, "date"]
    pipeline.web.search.assert_awaited_once_with(question)
    assert response.retrieval["web_search"][1]["status"] == "skipped_query_budget"
