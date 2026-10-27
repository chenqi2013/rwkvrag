"""Request-level official-domain scope never supplies page text or answers."""
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from llamaindex_retrieval import interactive_app


def test_scope_accepts_domains_and_rejects_urls_and_search_operators():
    q = interactive_app.Question(question='q', web_include_domains=['EXAMPLE.org.', 'example.org'])
    assert q.web_include_domains == ['example.org']
    for invalid in ['https://example.org', 'example.org OR other.org', 'localhost', 'example.org/path']:
        with pytest.raises(ValidationError):
            interactive_app.Question(question='q', web_include_domains=[invalid])


@pytest.mark.parametrize('scoped', [False, True])
async def test_scope_is_explicit_and_preserves_model_query_in_receipt(monkeypatch, scoped):
    observed = []
    async def search(query, **kwargs):
        observed.append((query, kwargs))
        return [], {'provider': 'test'}
    async def bases():
        return []
    async def answer(model, question, **kwargs):
        await kwargs['search']('original model query')
        return {'telemetry': {}, 'resolved_question': question, 'relation': '新问题'}
    monkeypatch.setattr(interactive_app.interactive_flat, 'answer', answer)
    trial = SimpleNamespace(config={'flat_web_scope_query': scoped, 'flat_web_include_domains': ['fallback.org']},
        web=SimpleNamespace(replay=SimpleNamespace(mode='off'), search=search), knowledge_bases=bases,
        conversation=lambda payload: ([], []))
    job, receipts = {}, []
    payload = interactive_app.Question(question='q', retrieval_mode='web', web_include_domains=['example.org'])
    await interactive_app.Trial.run_flat(trial, job, payload, SimpleNamespace(events=[]), receipts)
    expected = 'original model query' + (' (site:example.org)' if scoped else '')
    assert observed == [(expected, {'full_snapshot': True, 'include_domains': ['example.org']})]
    assert receipts[0]['query'] == 'original model query'
    assert receipts[0]['provider_query'] == expected
    assert receipts[0]['include_domains'] == ['example.org']


def test_separate_source_urls_are_validated_without_changing_question():
    q = interactive_app.Question(question='比较这两份公告', retrieval_mode='web',
        web_source_urls=['https://example.org/a', 'https://example.org/a'])
    assert q.question == '比较这两份公告'
    assert q.web_source_urls == ['https://example.org/a']
    for url in ['file:///tmp/source', 'javascript:alert(1)', 'https://user:secret@example.org/a']:
        with pytest.raises(ValidationError):
            interactive_app.Question(question='q', web_source_urls=[url])
    with pytest.raises(ValidationError):
        interactive_app.Question(question='q', retrieval_mode='knowledge_base',
            web_source_urls=['https://example.org/a'])
    with pytest.raises(ValidationError):
        interactive_app.Question(question='also https://example.org/extra',
            web_source_urls=[f'https://example.org/{i}' for i in range(8)])


async def test_source_urls_reach_acquisition_separately_from_question(monkeypatch):
    observed = []
    async def fetch_urls(urls, **kwargs):
        observed.append((urls, kwargs))
        return [], {'provider': 'test'}
    async def bases():
        return []
    async def answer(model, question, **kwargs):
        assert question == '请比较 https://example.org/b'
        assert kwargs['requested_urls'] == ['https://example.org/a', 'https://example.org/b']
        await kwargs['fetch_urls'](kwargs['requested_urls'])
        return {'telemetry': {}, 'resolved_question': question, 'relation': '新问题'}
    monkeypatch.setattr(interactive_app.interactive_flat, 'answer', answer)
    trial = SimpleNamespace(config={}, web=SimpleNamespace(replay=SimpleNamespace(mode='off'),
        fetch_urls=fetch_urls), knowledge_bases=bases, conversation=lambda payload: ([], []))
    receipts = []
    payload = interactive_app.Question(question='请比较 https://example.org/b', retrieval_mode='web',
        web_source_urls=['https://example.org/a', 'https://example.org/b'], web_include_domains=['example.org'])
    await interactive_app.Trial.run_flat(trial, {}, payload, SimpleNamespace(events=[]), receipts)
    assert observed == [(['https://example.org/a', 'https://example.org/b'], {'include_domains': ['example.org']})]
    assert receipts[0]['requested_urls'] == observed[0][0]
