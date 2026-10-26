import base64
import hashlib
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from llamaindex_retrieval import direct_web, interactive_app
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.web_replay import WebReplay
from llamaindex_retrieval.web_retrieval import WebSearchAdapter


def settings(tmp_path):
    return Settings(_env_file=None, web_tavily_api_key='test-secret',
                    web_guard_path=tmp_path / 'guard.sqlite3', web_min_interval_seconds=0)


async def test_record_replay_preserves_full_payload_and_never_calls_network(tmp_path, monkeypatch):
    raw = b'{ "results": [{"url":"https://example.org/a","title":"A","content":"summary",' \
          b'"raw_content":"complete original text","score":0.8}], "extra":{"untouched":true} }\n'
    requests = []

    def handler(request):
        requests.append(request)
        assert json.loads(request.content)['include_domains'] == ['example.org']
        return httpx.Response(200, content=raw)

    monkeypatch.setattr(direct_web, '_client', lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    store = tmp_path / 'captures'
    config = settings(tmp_path)
    record = WebSearchAdapter(config, replay={'mode': 'record', 'store': store})
    hits, trace = await record.search(' exact query ', full_snapshot=True, include_domains=['example.org'])
    key = WebReplay.key('tavily', ' exact query ', ['example.org'], config.web_search_results, True)
    payload, capture = record.replay.load(key)
    assert base64.b64decode(payload['provider_response_base64']) == raw
    assert payload['provider_response_sha256'] == hashlib.sha256(raw).hexdigest()
    assert json.loads(raw)['extra'] == {'untouched': True}
    assert 'test-secret' not in next(store.glob('*.json')).read_text()

    def forbidden():
        raise AssertionError('replay attempted a network transport')

    monkeypatch.setattr(direct_web, '_client', forbidden)
    replay = WebSearchAdapter(config, replay={'mode': 'replay', 'store': store})
    replay._execute = AsyncMock(side_effect=AssertionError('replay attempted live execution'))
    replay_hits, replay_trace = await replay.search(' exact query ', full_snapshot=True, include_domains=['example.org'])
    assert replay_hits == hits
    assert {k: v for k, v in replay_trace.items() if k != 'web_replay'} == {
        k: v for k, v in trace.items() if k != 'web_replay'}
    assert replay_trace['web_replay']['payload_sha256'] == capture['payload_sha256']
    assert replay.replay.load(key)[0] == payload
    replay._execute.assert_not_awaited()
    # Record also reuses the first capture of an exact request, without spending quota again.
    assert (await record.search(' exact query ', full_snapshot=True, include_domains=['example.org']))[0] == hits
    assert len(requests) == 1


async def test_replay_miss_is_an_acquisition_failure_with_no_live_fallback(tmp_path, monkeypatch):
    def forbidden():
        raise AssertionError('network must never be used for replay misses')

    monkeypatch.setattr(direct_web, '_client', forbidden)
    trial = interactive_app.Trial({'pipeline': 'flat', 'settings': {},
        'admin_base_url': 'http://localhost:1', 'records_dir': str(tmp_path / 'records'),
        'web_replay': {'mode': 'replay', 'store': str(tmp_path / 'empty')}})
    trial.knowledge_bases = AsyncMock(return_value=[])

    async def answer(model, question, *, search, **kwargs):
        try:
            await search('q')
        except Exception as error:
            return {'answer': {}, 'errors': [{'stage': 'acquire', 'error': getattr(error, 'safe_code', '')}],
                    'resolved_question': 'q', 'relation': '新问题', 'telemetry': {},
                    'acquisition_incomplete': True}
        raise AssertionError('replay miss must fail')

    monkeypatch.setattr(interactive_app.interactive_flat, 'answer', answer)
    trial.web._execute = AsyncMock(side_effect=AssertionError('live search called'))
    job = {'id': 'a' * 32, 'question': 'q', 'status': 'queued'}
    try:
        await trial.run(job, interactive_app.Question(question='q', retrieval_mode='web'))
        saved = json.loads((tmp_path / 'records' / ('a' * 32 + '.json')).read_text())
        assert saved['result']['errors'] == [{'stage': 'acquire', 'error': 'web_replay_miss'}]
        assert saved['result']['acquisition_incomplete'] is True
        trial.web._execute.assert_not_awaited()
    finally:
        await trial.native.aclose()
        await trial.admin.aclose()


async def test_corrupt_capture_is_rejected_without_live_fallback(tmp_path):
    store = WebReplay(mode='record', store=tmp_path)
    key = store.key('tavily', 'q', None, 5, True)
    store.save(key, {'provider': 'tavily', 'hits': []})
    path = next(tmp_path.glob('*.json'))
    envelope = json.loads(path.read_text())
    envelope['payload_sha256'] = '0' * 64
    path.write_text(json.dumps(envelope))
    live = AsyncMock(side_effect=AssertionError('corrupt captures must not reach network'))
    with pytest.raises(direct_web.WebProviderError, match='web_replay_corrupt'):
        await WebReplay(mode='replay', store=tmp_path).execute(key, live)
    live.assert_not_awaited()


def test_every_key_field_is_exact_and_capture_is_immutable(tmp_path):
    store = WebReplay(mode='record', store=tmp_path)
    key = store.key('tavily', ' q ', ['a.org', 'b.org'], 5, True)
    store.save(key, {'provider': 'tavily', 'hits': []})
    for field, value in [('provider', 'searxng'), ('query', 'q'), ('include_domains', ['b.org', 'a.org']),
                         ('include_domains', None), ('max_results', 4), ('full_snapshot', False)]:
        with pytest.raises(direct_web.WebProviderError, match='web_replay_miss'):
            store.load({**key, field: value})
    with pytest.raises(direct_web.WebProviderError, match='web_replay_conflict'):
        store.save(key, {'provider': 'tavily', 'hits': [{'changed': True}]})
    assert store.load(key)[0] == {'provider': 'tavily', 'hits': []}


@pytest.mark.parametrize('config', [{'mode': 'unknown'}, {'mode': 'replay'}, {'mode': 'record'}])
def test_invalid_replay_configuration_fails_before_acquisition(config):
    with pytest.raises(ValueError, match='invalid web_replay'):
        WebReplay(**config)


@pytest.mark.parametrize('code', ['web_upstream_http_502', 'web_upstream_failed', 'web_acquisition_timeout'])
async def test_recorded_transient_failure_is_replayed_as_failure_without_network(tmp_path, code):
    store = WebReplay(mode='record', store=tmp_path)
    key = store.key('anysearch', 'exact query', None, 5, True)
    live = AsyncMock(side_effect=direct_web.WebProviderError(code, operation='search'))
    with pytest.raises(direct_web.WebProviderError, match=code) as first:
        await store.execute(key, live)
    assert first.value.web_replay['replayed'] is False
    assert first.value.operation == 'search'
    before = next(tmp_path.glob('*.json')).read_bytes()
    live.reset_mock()
    for mode in ('record', 'replay'):
        with pytest.raises(direct_web.WebProviderError, match=code) as again:
            await WebReplay(mode=mode, store=tmp_path).execute(key, live)
        assert again.value.web_replay['replayed'] is True
    live.assert_not_awaited()
    assert next(tmp_path.glob('*.json')).read_bytes() == before


@pytest.mark.parametrize('code', ['web_upstream_http_401', 'web_upstream_http_402',
                                 'web_upstream_http_429', 'web_guard_unavailable'])
async def test_credential_rate_and_local_failures_do_not_poison_capture(tmp_path, code):
    store = WebReplay(mode='record', store=tmp_path)
    key = store.key('anysearch', 'q', None, 5, True)
    with pytest.raises(direct_web.WebProviderError, match=code):
        await store.execute(key, AsyncMock(side_effect=direct_web.WebProviderError(code)))
    assert not list(tmp_path.glob('*.json'))


def test_failure_capture_is_immutable_and_idempotent(tmp_path):
    store = WebReplay(mode='record', store=tmp_path)
    key = store.key('anysearch', 'q', None, 5, True)
    payload = {'provider': 'anysearch', 'error': {'code': 'web_upstream_http_502', 'operation': None}}
    store.save(key, payload)
    raw = next(tmp_path.glob('*.json')).read_bytes()
    assert json.loads(raw)['format'] == 'web-replay-v2'
    store.save(key, payload)
    with pytest.raises(direct_web.WebProviderError, match='web_replay_conflict'):
        store.save(key, {'provider': 'anysearch', 'error': {'code': 'web_upstream_http_503', 'operation': None}})
    assert next(tmp_path.glob('*.json')).read_bytes() == raw
