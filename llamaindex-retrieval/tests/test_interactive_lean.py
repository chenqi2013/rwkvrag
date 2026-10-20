"""Response contracts, deadline cancellation and dependency/call limits."""
import asyncio
import json
from time import monotonic
from unittest.mock import AsyncMock

import pytest

from llamaindex_retrieval import interactive_lean as flow, kb_navigation as nav, model_profiles
from llamaindex_retrieval.direct_web import WebProviderError
from llamaindex_retrieval.native_rwkv import NativeRWKVResult
from llamaindex_retrieval.schemas import SourceItem


class Model:
    def __init__(self, *, scores=True, invalid=False, missing=False, plan_failure=False, delay=None):
        self.scores, self.invalid, self.missing = scores, invalid, missing
        self.plan_failure, self.delay = plan_failure, delay or {}
        self.calls, self.active, self.peak = [], 0, 0
        self.raw = '  原始模型文字\n保留所有内容  '

    async def complete(self, messages, **kwargs):
        if kwargs.get('check_only'):
            return NativeRWKVResult('completed', None, None, {'completion_attempted': False})
        instruction, data = messages[0]['content'].split('\n', 1)
        data = json.loads(data)
        self.calls.append((instruction, data, kwargs))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay.get(instruction, 0))
            if instruction == flow.PLAN:
                if self.plan_failure:
                    return NativeRWKVResult('invalid_output', 'original plan failure', 'stop', {})
                value = {'question': 'resolved', 'needs': ['need1', 'need2'],
                         'kb_queries': ['kb'] if data['channels'].get('knowledge_bases') else [],
                         'web_queries': ['web'] if data['channels'].get('web') else [], 'sites': ['example.org']}
            elif instruction == flow.SCORE:
                value = {'scores': [{'span_id': data['spans'][0]['id'], 'need_id': data['needs'][0]['id'],
                                     'score': '3'}] if self.scores else []}
            elif instruction == flow.ANSWER:
                if self.invalid:
                    return NativeRWKVResult('completed', self.raw, 'stop', {})
                value = {'answer': 'verbatim answer', 'citations': [data['spans'][0]['id']],
                         'not_found_needs': ['I2'] if self.missing else []}
            elif instruction == flow.PAGES:
                value = {'ids': [p['id'] for p in data['pages']][:3]}
            elif instruction.startswith(nav.TITLE_PICK[:20]):
                value = {'ids': [p['id'] for p in data['documents']][:3]}
            elif instruction == flow.OUTLINE:
                value = {'ids': [data['entries'][-1]['id']]}
            else:
                raise AssertionError(instruction)
            return NativeRWKVResult('completed', json.dumps(value), 'stop', {})
        finally:
            self.active -= 1


def source(i=0, *, channel='web', text='An exact original sentence with reliable evidence.'):
    return SourceItem(id=f'D{i}', document_id=f'f{i}', title=f'Article {i}', source=channel,
                      uri=f'https://example.org/{i}' if channel == 'web' else None,
                      snippet=text, score=0, metadata={'search_snippet': f'Navigation snippet {i}'})


async def run(model, *, catalog_size=0, docs=None, search=None, **options):
    docs = docs if docs is not None else [source()]
    search = search or AsyncMock(return_value=docs)
    document = AsyncMock(side_effect=lambda i: source(i, channel='knowledge_base'))
    result = await flow.answer(model, 'raw question', catalog=[{'id': str(i), 'title': f'Title {i}'}
        for i in range(catalog_size)], document=document, search=search,
        scope={'web': True, 'knowledge_bases': ['test'] if catalog_size else []}, progress=lambda _: None,
        **options)
    return result, search


@pytest.mark.parametrize('missing,expected', [(False, 'answered'), (True, 'partial')])
async def test_answer_types_with_original_text_and_bound_citations(missing, expected):
    result, search = await run(Model(missing=missing))
    assert result['response_type'] == expected
    assert result['display_text'] == 'verbatim answer'
    assert result['answer']['generation']['citation_bindings']
    assert search.await_args.kwargs['include_domains'] == ['example.org']
    assert result['not_found_needs'] == ([{'id': 'I2', 'text': 'need2'}] if missing else [])


async def test_no_evidence_never_calls_answer_and_shows_search_and_sources():
    model = Model(scores=False)
    result, _ = await run(model)
    assert result['response_type'] == 'not_found'
    assert result['display_text'] == '资料中未找到可靠依据'
    assert not any(i == flow.ANSWER for i, _, _ in model.calls)
    assert result['acquisition'][0]['query'] == 'web'
    assert result['answer']['sources'][0]['uri'] == 'https://example.org/0'


async def test_invalid_answer_is_unverified_and_preserves_every_raw_character():
    model = Model(invalid=True)
    result, _ = await run(model)
    assert result['response_type'] == 'unverified'
    assert result['display_text'] == model.raw == result['answer']['answer']
    assert result['answer']['generation']['parsed_answer'] is None


async def test_server_down_is_error_and_attempt_is_recorded():
    model = Model()
    model.complete = AsyncMock(side_effect=ConnectionError('down'))
    result, _ = await run(model)
    assert result['response_type'] == 'error'
    assert result['errors'][0]['error_type'] == 'ConnectionError'


async def test_plan_failure_uses_raw_question_in_only_permitted_channels():
    model = Model(plan_failure=True)
    result, search = await run(model, catalog_size=1)
    assert search.await_args.args == ('raw question',)
    assert search.await_args.kwargs['include_domains'] is None
    assert result['graph']['nodes'][0]['raw_text'] == 'original plan failure'
    assert result['graph']['nodes'][0]['fallback'] == 'raw_question_permitted_channels'


@pytest.mark.parametrize('catalog_size', [30, 5000])
async def test_parallel_navigation_and_reading_have_bounded_critical_path(catalog_size):
    model = Model()
    result, _ = await run(model, catalog_size=catalog_size, docs=[source(i) for i in range(5)])
    telemetry = result['telemetry']
    assert telemetry['critical_path_calls'] <= 15
    assert telemetry['reading_calls'] <= 9
    assert telemetry['total_calls'] == len(model.calls)
    assert model.peak > 1
    titles = [d for i, d, _ in model.calls if i.startswith(nav.TITLE_PICK[:20])]
    assert max(len(d['documents']) for d in titles) <= 250
    assert len(titles) == (1 if catalog_size <= 100 else 21)
    assert len([d for i, d, _ in model.calls if i == flow.PAGES]) == 1


async def test_deadline_cancels_reading_and_still_returns_with_attempts():
    model = Model(delay={flow.SCORE: .2})
    start = monotonic()
    result, _ = await run(model, deadline_seconds=.05, answer_reserve_seconds=.01)
    assert monotonic()-start < .15
    assert result['response_type'] == 'not_found'
    assert result['telemetry']['deadline_expired']
    assert model.active == 0


async def test_answer_timeout_is_unverified_and_does_not_invent_raw_text():
    model = Model(delay={flow.ANSWER: .2})
    result, _ = await run(model, deadline_seconds=.05, answer_reserve_seconds=.02)
    assert result['response_type'] == 'unverified'
    assert result['display_text'] == ''
    assert result['answer']['generation']['parsed_answer'] is None
    assert model.active == 0


async def test_domain_rejection_retries_once_without_sites_but_quota_never_retries():
    search = AsyncMock(side_effect=[WebProviderError('web_upstream_http_400'), [source()]])
    result, _ = await run(Model(), search=search)
    assert result['response_type'] == 'answered'
    assert search.await_count == 2
    assert search.await_args.kwargs['include_domains'] is None
    search = AsyncMock(side_effect=WebProviderError('web_upstream_http_432'))
    result, _ = await run(Model(), search=search)
    assert search.await_count == 1
    assert result['response_type'] == 'error'
    assert result['errors'][0]['error'] == 'web_upstream_http_432'


async def test_outline_and_first_window_modes_both_fit_nine_reads():
    docs = [source(i, text='# First\n'+('evidence text. '*300)+'\n# Last\n'+('late text. '*300)) for i in range(6)]
    model = Model()
    result, _ = await run(model, docs=docs, reading_mode='outline')
    assert result['telemetry']['reading_calls'] <= 9
    assert any(i == flow.OUTLINE for i, _, _ in model.calls)
    assert result['response_type'] == 'answered'


async def test_final_evidence_character_budget_and_score_order_are_enforced():
    model = Model()
    result, _ = await run(model, docs=[source(i, text='a sentence. '*400) for i in range(5)], final_chars=640)
    writer = next(data for i, data, _ in model.calls if i == flow.ANSWER)
    assert sum(len(s['text']) for s in writer['spans']) <= 640
    assert result['response_type'] == 'answered'


class BudgetModel(Model):
    async def check_budget(self, messages, **kwargs):
        instruction, text = messages[0]['content'].split('\n', 1)
        data = json.loads(text)
        fits = (len(data.get('spans', [])) <= 1 if instruction in {flow.SCORE, flow.ANSWER}
                else all(len(p['snippet']) <= 128 for p in data['pages']))
        return NativeRWKVResult('completed' if fits else 'budget_exceeded', None, None,
                                {'completion_attempted': False, 'budget': {'fits': fits}})


async def test_exact_budget_packing_preserves_whole_original_spans_and_all_pages():
    model = BudgetModel()
    docs = [source(i, text='Original evidence sentence. '*200) for i in range(5)]
    for doc in docs:
        doc.metadata['search_snippet'] = 'navigation '*200
    result, _ = await run(model, docs=docs, answer_tokens=384)
    assert result['response_type'] == 'answered'
    pages = [d for i, d, _ in model.calls if i == flow.PAGES]
    assert len(pages) == 1 and len(pages[0]['pages']) == 5
    assert all(len(p['snippet']) <= 128 for p in pages[0]['pages'])
    reads = [d for i, d, _ in model.calls if i == flow.SCORE]
    assert reads and all(len(d['spans']) == 1 for d in reads)
    writer = next((d, k) for i, d, k in model.calls if i == flow.ANSWER)
    assert len(writer[0]['spans']) == 1 and writer[1]['max_tokens'] == 384
    cited = result['graph']['spans'][0]
    assert cited['text'] in docs[0].snippet
    assert writer[0]['spans'][0]['text'] == cited['text']
    assert result['telemetry']['total_calls'] == len(model.calls)
    assert result['telemetry']['budget_checks'] > len(reads)
    assert all(not b['trace']['completion_attempted'] for b in result['graph']['budget_checks'])


async def test_no_fitting_answer_evidence_skips_generation_without_inventing_an_answer():
    model = BudgetModel()
    original = model.check_budget

    async def check(messages, **kwargs):
        if messages[0]['content'].startswith(flow.ANSWER):
            return NativeRWKVResult('budget_exceeded', None, None, {'completion_attempted': False})
        return await original(messages, **kwargs)

    model.check_budget = check
    result, _ = await run(model)
    assert result['response_type'] == 'not_found'
    assert not any(i == flow.ANSWER for i, _, _ in model.calls)
    assert result['graph']['spans'] == []
    assert result['answer']['generation']['raw_text'] is None


async def test_observed_budget_check_never_creates_a_generation_event():
    from llamaindex_retrieval.interactive_app import ObservedModel
    native = AsyncMock()
    native.complete.return_value = NativeRWKVResult('completed', None, None, {'completion_attempted': False})
    observed = ObservedModel(native, {})
    await observed.check_budget([{'role': 'user', 'content': 'exact input'}], stage='reader')
    assert native.complete.await_args.kwargs['check_only'] is True
    assert observed.events == []
    native.complete.return_value = NativeRWKVResult('completed', 'unsafe', 'stop', {'completion_attempted': True})
    with pytest.raises(RuntimeError):
        await observed.check_budget([], stage='reader')


async def test_deadline_also_cancels_budget_tokenization():
    model = BudgetModel()

    async def blocked(*args, **kwargs):
        await asyncio.sleep(1)

    model.check_budget = blocked
    start = monotonic()
    result, _ = await run(model, deadline_seconds=.05, answer_reserve_seconds=.01)
    assert monotonic()-start < .15
    assert result['telemetry']['deadline_expired']
    assert not any(i in {flow.SCORE, flow.ANSWER} for i, _, _ in model.calls)


def test_model_profiles_resolve_all_four_fields_without_changing_input():
    config = {'active_model': 'larger', 'settings': {'native_api_key': 'private'}, 'model_profiles': {
        'larger': {'model': 'rwkv-larger', 'base_url': 'http://localhost:123/v1',
                   'context_window_tokens': 4096, 'concurrency': 16}}}
    resolved = model_profiles.resolve(config)
    assert resolved['settings']['native_model'] == 'rwkv-larger'
    assert resolved['settings']['native_context_window_tokens'] == 4096
    assert resolved['settings']['native_api_key'] == 'private'
    assert config['settings'] == {'native_api_key': 'private'}
    with pytest.raises(ValueError):
        model_profiles.resolve({**config, 'active_model': 'unknown'})


async def test_app_honors_lean_deadline_and_preserves_result_without_context_call(tmp_path, monkeypatch):
    from llamaindex_retrieval import interactive_app as app
    trial = app.Trial({'pipeline': 'lean', 'settings': {}, 'gpu': 3, 'admin_base_url': 'http://localhost:1',
                       'records_dir': str(tmp_path),
                       'lean_options': {'deadline_seconds': .05, 'answer_reserve_seconds': .02}})
    trial.knowledge_bases = AsyncMock(return_value=[])
    original_native = trial.native
    trial.native = Model(delay={flow.ANSWER: .2})
    # ObservedModel still records the original model call before cancellation.
    async def search(*args, **kwargs):
        from llamaindex_retrieval.lexical_index import LexicalResult
        return [LexicalResult(node_id='x', document_id='x', text='source text', score=0,
            metadata={'uri': 'https://example.org/x', 'title': 'X', 'search_snippet': 'hint'})], {}
    trial.web.search = search
    contextualize = AsyncMock(side_effect=AssertionError('lean must merge context and planning'))
    monkeypatch.setattr(app.interactive_funnel, 'contextualize', contextualize)
    job = {'id': 'b'*32, 'question': 'q', 'status': 'queued'}
    try:
        await trial.run(job, app.Question(question='q', retrieval_mode='web'))
        assert job['status'] == 'completed'
        assert job['result']['response_type'] == 'unverified'
        assert json.loads((tmp_path/('b'*32+'.json')).read_text())['model_calls'][-1]['status'] == 'cancelled'
        contextualize.assert_not_awaited()
    finally:
        await original_native.aclose()
        await trial.admin.aclose()
