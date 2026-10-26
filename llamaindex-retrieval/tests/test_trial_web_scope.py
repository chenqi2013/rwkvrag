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
