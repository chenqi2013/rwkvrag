import asyncio
import base64
import hashlib
import json

import httpx
import pytest

from llamaindex_retrieval import direct_web, interactive_app
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.web_retrieval import WebSearchAdapter


def settings(tmp_path, **kwargs):
    return Settings(_env_file=None, web_search_provider="anysearch",
                    web_anysearch_api_key="single-test-key", web_min_interval_seconds=0,
                    web_guard_path=tmp_path / "guard.sqlite3", **kwargs)


def install(monkeypatch, handler):
    monkeypatch.setattr(direct_web, "_client", lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler), trust_env=False))


def response(data):
    return httpx.Response(200, json={"code": 0, "message": "success", "data": data})


async def test_search_extract_record_and_replay_preserve_original_bytes(tmp_path, monkeypatch):
    search_raw = b'{ "code":0,"data":{"results":[{"url":"https://example.org/a",' \
                 b'"title":"A","snippet":"short summary","content":"decorated summary"}]}}\n'
    page_raw = b'{"code":0,"data":{"url":"https://example.org/a","content":"Exact page.\\nMore text."}}'
    calls = []

    def handler(request):
        calls.append(request.url.path)
        assert request.headers["Authorization"] == "Bearer single-test-key"
        body = json.loads(request.content)
        if request.url.path == "/v1/search":
            assert body == {"query": " exact query ", "max_results": 2, "format": "json"}
            return httpx.Response(200, content=search_raw)
        assert request.url.path == "/v1/extract"
        assert body == {"url": "https://example.org/a"}
        return httpx.Response(200, content=page_raw)

    install(monkeypatch, handler)
    config = settings(tmp_path)
    store = tmp_path / "captures"
    adapter = WebSearchAdapter(config, replay={"mode": "record", "store": store})
    hits, trace = await adapter.search(" exact query ", full_snapshot=True)
    assert hits[0].text == "Exact page.\nMore text."
    assert hits[0].metadata["content_status"] == "fetched"
    assert hits[0].metadata["search_snippet"] == "short summary"
    key = adapter.replay.key("anysearch", " exact query ", None, 2, True,
                             provider_options={"extract_timeout": 8.0, "search_timeout": 45.0})
    payload = adapter.replay.load(key)[0]
    captures = payload["provider_responses"]
    assert [r["operation"] for r in captures] == ["search", "extract"]
    for row, raw in zip(captures, (search_raw, page_raw), strict=True):
        assert base64.b64decode(row["response_base64"]) == raw
        assert row["response_sha256"] == hashlib.sha256(raw).hexdigest()
    assert "single-test-key" not in str(payload) + str(trace)

    def forbidden():
        raise AssertionError("replay attempted network")

    monkeypatch.setattr(direct_web, "_client", forbidden)
    replay = WebSearchAdapter(config, replay={"mode": "replay", "store": store})
    replay_hits, replay_trace = await replay.search(" exact query ", full_snapshot=True)
    assert replay_hits == hits
    assert replay_trace["web_replay"]["replayed"]
    assert (await adapter.search(" exact query ", full_snapshot=True))[0] == hits
    assert calls == ["/v1/search", "/v1/extract"]
    changed = config.model_copy(update={"web_anysearch_extract_timeout": 4})
    with pytest.raises(direct_web.WebProviderError, match="web_replay_miss"):
        await WebSearchAdapter(changed, replay={"mode": "replay", "store": store}).search(
            " exact query ", full_snapshot=True)


async def test_one_slow_page_preserves_previously_fetched_pages(tmp_path, monkeypatch):
    async def handler(request):
        if request.url.path == "/v1/search":
            return response({"results": [{"url": "https://example.org/fast", "snippet": "fast snippet"},
                                          {"url": "https://example.org/slow", "snippet": "slow snippet"}]})
        url = json.loads(request.content)["url"]
        if url.endswith("slow"):
            await asyncio.sleep(5)
        return response({"url": url, "content": "Exact page"})

    install(monkeypatch, handler)
    config = settings(tmp_path, web_anysearch_extract_timeout=0.25)
    hits, trace = await WebSearchAdapter(config).search("q", full_snapshot=True)
    assert [h.text for h in hits] == ["Exact page", "slow snippet"]
    assert hits[0].metadata["content_status"] == "fetched"
    assert hits[1].metadata["fetch_error"] == "web_anysearch_extract_timeout"
    assert trace["provider_options"] == {"extract_timeout": 0.25, "search_timeout": 45}


async def test_total_extraction_budget_returns_all_snippets_without_losing_results(tmp_path, monkeypatch):
    attempts = []

    async def handler(request):
        if request.url.path == "/v1/search":
            return response({"results": [{"url": f"https://example.org/{i}", "snippet": f"snippet{i}"}
                                          for i in range(5)]})
        attempts.append(request.url.path)
        await asyncio.sleep(5)
        return response({})

    install(monkeypatch, handler)
    config = settings(tmp_path, web_search_timeout=1, web_anysearch_extract_timeout=0.4,
                      web_search_results=5)
    hits, _ = await WebSearchAdapter(config).search("q", full_snapshot=True)
    assert [h.text for h in hits] == [f"snippet{i}" for i in range(5)]
    assert 1 <= len(attempts) < 5
    assert all(h.metadata["content_status"] == "snippet_only" for h in hits)
    assert hits[-1].metadata["fetch_error"] == "web_anysearch_extract_budget"


async def test_extract_failure_keeps_snippet_and_safe_failure_status(tmp_path, monkeypatch):
    def handler(request):
        if request.url.path == "/v1/search":
            return response({"results": [{"url": "https://example.org/a", "snippet": "Original snippet",
                                          "content": "Search content is not a verified page"}]})
        return httpx.Response(422, text="private provider details single-test-key")

    install(monkeypatch, handler)
    hits, trace = await WebSearchAdapter(settings(tmp_path)).search("q", full_snapshot=True)
    assert hits[0].text == "Original snippet"
    assert hits[0].metadata["content_status"] == "snippet_only"
    assert hits[0].metadata["fetch_error"] == "web_upstream_http_422"
    assert "single-test-key" not in str(trace)


async def test_extract_502_retains_prior_page_and_honors_circuit_for_remaining_pages(tmp_path, monkeypatch):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == '/v1/search':
            return response({'results': [{'url': f'https://example.org/{i}', 'snippet': f'snippet{i}'}
                                          for i in range(3)]})
        url = json.loads(request.content)['url']
        if url.endswith('/1'):
            return httpx.Response(502, text='private upstream details')
        return response({'url': url, 'content': 'Exact first page'})

    install(monkeypatch, handler)
    hits, _ = await WebSearchAdapter(settings(tmp_path, web_search_results=3)).search('q', full_snapshot=True)
    assert [h.text for h in hits] == ['Exact first page', 'snippet1', 'snippet2']
    assert [h.metadata['fetch_error'] for h in hits] == ['', 'web_upstream_http_502',
                                                      'web_upstream_circuit_open_502']
    assert calls == ['/v1/search', '/v1/extract', '/v1/extract']


@pytest.mark.parametrize("status", [401, 402, 403, 429])
async def test_extract_auth_quota_and_rate_failures_block_shared_key_without_rotation(tmp_path, monkeypatch, status):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/v1/search":
            return response({"results": [{"url": "https://example.org/a", "snippet": "a"},
                                          {"url": "https://example.org/b", "snippet": "b"}]})
        return httpx.Response(status, text="credentials and provider message must stay private")

    install(monkeypatch, handler)
    config = settings(tmp_path)
    with pytest.raises(direct_web.WebProviderError, match=f"^web_upstream_http_{status}$"):
        await WebSearchAdapter(config).search("first")
    with pytest.raises(direct_web.WebProviderError, match=f"^web_upstream_circuit_open_{status}$"):
        await WebSearchAdapter(config).search("second")
    assert calls == ["/v1/search", "/v1/extract"]


async def test_domain_filter_is_url_based_without_rewriting_query_or_provider_fields(tmp_path, monkeypatch):
    pages = []

    def handler(request):
        if request.url.path == "/v1/search":
            body = json.loads(request.content)
            assert body["query"] == "unchanged model query" and "include_domains" not in body
            return response({"results": [
                {"url": "https://evil-example.org/a", "snippet": "discard"},
                {"url": "file:///private", "snippet": "discard"},
                {"url": "https://docs.example.org/a", "snippet": "keep"}]})
        url = json.loads(request.content)["url"]
        pages.append(url)
        return response({"url": url, "content": "Original body"})

    install(monkeypatch, handler)
    hits, _ = await WebSearchAdapter(settings(tmp_path)).search(
        "unchanged model query", full_snapshot=True, include_domains=["example.org"])
    assert [h.metadata["uri"] for h in hits] == pages == ["https://docs.example.org/a"]


@pytest.mark.parametrize("body", [{"code": -1, "message": "single-test-key"},
                                 {"code": False, "data": {}}, {"data": {}},
                                 {"code": 0, "data": []}, {"code": 0, "data": {"results": {}}}])
async def test_http_200_with_invalid_business_envelope_is_not_success(tmp_path, monkeypatch, body):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json=body)

    install(monkeypatch, handler)
    with pytest.raises(direct_web.WebProviderError) as caught:
        await WebSearchAdapter(settings(tmp_path)).search("q")
    assert "single-test-key" not in str(caught.value)
    assert calls == ["/v1/search"]


async def test_anysearch_requires_one_private_credential_source(tmp_path, monkeypatch):
    def forbidden():
        raise AssertionError("invalid credentials must not reach network")

    monkeypatch.setattr(direct_web, "_client", forbidden)
    for options, code in [({}, "key_not_configured"),
                          ({"web_anysearch_api_key": "one,two"}, "key_invalid"),
                          ({"web_anysearch_api_key": "one\ntwo"}, "key_invalid")]:
        config = Settings(_env_file=None, web_search_provider="anysearch", **options)
        with pytest.raises(direct_web.WebProviderError, match="web_anysearch_" + code):
            await WebSearchAdapter(config).search("q")
    keyfile = tmp_path / "active.key"
    keyfile.write_text("file-key\n")
    keyfile.chmod(0o600)
    config = settings(tmp_path, web_anysearch_api_key_file=keyfile)
    with pytest.raises(direct_web.WebProviderError, match="web_anysearch_multiple_key_sources"):
        await WebSearchAdapter(config).search("q")
    config.web_anysearch_api_key = type(config.web_anysearch_api_key)("")
    keyfile.chmod(0o644)
    with pytest.raises(direct_web.WebProviderError, match="web_anysearch_key_file_permissions"):
        await WebSearchAdapter(config).search("q")


async def test_trial_meta_uses_selected_providers_credentials(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    trial = SimpleNamespace(settings=settings(tmp_path), config={"gpu": 3},
                            knowledge_bases=AsyncMock(return_value=[]))
    monkeypatch.setattr(interactive_app.app.state, "trial", trial, raising=False)
    assert (await interactive_app.meta())["web_configured"] is True
    trial.settings.web_anysearch_api_key = type(trial.settings.web_anysearch_api_key)("")
    trial.settings.web_tavily_api_key = type(trial.settings.web_tavily_api_key)("irrelevant")
    assert (await interactive_app.meta())["web_configured"] is False


async def test_parallel_guard_slots_extract_pages_concurrently(tmp_path, monkeypatch):
    active, peak = [0], [0]

    async def handler(request):
        if request.url.path == "/v1/search":
            return response({"results": [{"url": f"https://example.org/{i}", "snippet": f"s{i}"} for i in range(4)]})
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        await asyncio.sleep(0.2)
        active[0] -= 1
        url = json.loads(request.content)["url"]
        return response({"url": url, "content": "page " + url[-1]})

    install(monkeypatch, handler)
    config = settings(tmp_path, web_search_results=4, web_provider_parallel=4)
    hits, _ = await WebSearchAdapter(config).search("q", full_snapshot=True)
    assert [h.text for h in hits] == ["page 0", "page 1", "page 2", "page 3"]
    assert peak[0] > 1


async def test_pool_rotates_past_quota_key_without_logging_keys(tmp_path, monkeypatch):
    seen = []

    def handler(request):
        key = request.headers['Authorization'].split()[-1]
        seen.append(key)
        if key == 'k1':
            return httpx.Response(402, text='quota')
        if request.url.path == '/v1/search':
            return response({'results': [{'url': 'https://example.org/a', 'snippet': 'a'}]})
        return response({'url': 'https://example.org/a', 'content': 'page'})

    install(monkeypatch, handler)
    pool = tmp_path / 'pool.keys'
    pool.write_text('k1\nk2\n')
    pool.chmod(0o600)
    config = Settings(_env_file=None, web_search_provider='anysearch', web_anysearch_api_key_pool_file=pool,
                      web_min_interval_seconds=0, web_guard_path=tmp_path / 'guard.sqlite3')
    hits, trace = await WebSearchAdapter(config).search('q', full_snapshot=True)
    assert [h.text for h in hits] == ['page'] and seen[0] == 'k1' and set(seen[1:]) == {'k2'}
    assert 'k1' not in json.dumps(trace) and 'k2' not in json.dumps(trace)
