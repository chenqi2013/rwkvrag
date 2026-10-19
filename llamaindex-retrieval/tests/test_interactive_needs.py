"""Per-need scoring, bounded evidence, need answers, composition, card navigation."""
import asyncio
import json

from llamaindex_retrieval import interactive_needs as flow
from llamaindex_retrieval import kb_navigation as nav
from llamaindex_retrieval.native_rwkv import NativeRWKVResult
from llamaindex_retrieval.schemas import SourceItem


def source(i, text, channel='web'):
    return SourceItem(id=f'D{i}', document_id=f'doc{i}', title=f'资料{i}', source=channel,
                      uri=f'https://example.org/{i}' if channel == 'web' else None, score=0, snippet=text)


def reply(value):
    return NativeRWKVResult('completed', json.dumps(value, ensure_ascii=False), 'stop', {})


class Model:
    """Spans containing '答案X' score 3 for need IX; '线索X' scores 2. Answers cite all offered spans."""

    def __init__(self, needs=('A',), plan=None, requery=(), answer_status=None, pick=None):
        self.calls, self.requery, self.events = [], list(requery), []
        self.plan = plan or {'needs': list(needs), 'web_queries': ['q']}
        self.answer_status = answer_status or {}
        self.pick = pick

    async def complete(self, messages, **kwargs):
        instruction, data = messages[0]['content'].split('\n', 1)
        data = json.loads(data)
        self.calls.append((instruction, data, kwargs))
        self.events.append({'messages': messages, 'status': 'completed', 'trace': {}})
        await asyncio.sleep(0)
        props = kwargs['structured_schema']['properties']
        if instruction == flow.PLAN:
            return reply({k: self.plan.get(k, []) for k in props})
        if instruction == flow.REQUERY:
            queries = self.requery.pop(0) if self.requery else []
            return reply({k: queries for k in props})
        if instruction == flow.SCORE:
            rows = []
            for span in data['spans']:
                for n, need in enumerate(data['needs'], 1):
                    if f'答案{n}' in span['text']:
                        rows.append({'span_id': span['id'], 'need_id': need['id'], 'score': '3'})
                    elif f'线索{n}' in span['text']:
                        rows.append({'span_id': span['id'], 'need_id': need['id'], 'score': '2'})
            return reply({'scores': rows})
        if instruction == flow.PICK:
            return reply({'span_ids': self.pick(data) if self.pick else [data['spans'][-1]['id']]})
        if instruction == flow.ANSWER:
            status = self.answer_status.get(data['need']['id'], 'found')
            ids = [s['id'] for s in data['spans']] if status == 'found' else []
            return reply({'status': status, 'answer': '短答案', 'span_ids': ids})
        if instruction == flow.COMPOSE:
            ids = [i for a in data['answers'] for i in a['span_ids']]
            self.composed = data
            return reply({'answer': '最终回答', 'source_ids': ids})
        raise AssertionError(instruction)


async def run(model, table, **kwargs):
    calls = []

    async def acquire(query, channels=None):
        calls.append(query)
        return table.get(query, [])
    kwargs.setdefault('page_gate', False)
    result = await flow.answer(model, '问题', acquire=acquire, scope={'web': True},
                               progress=lambda _: None, **kwargs)
    return result, calls


async def test_each_need_answers_from_its_own_top_spans_and_composition_cites_them():
    docs = [source(0, '答案1在这里。'), source(1, '答案2在这里。'), source(2, '无关内容。')]
    model = Model(needs=['A', 'B'])
    result, _ = await run(model, {'q': docs}, supplementary=False)
    answers = {d['need']['id']: [s['text'] for s in d['spans']] for i, d, _ in model.calls if i == flow.ANSWER}
    assert answers == {'I1': ['答案1在这里。'], 'I2': ['答案2在这里。']}
    gen = result['answer']['generation']
    assert gen['need_answers'] == {'I1': 'found', 'I2': 'found'}
    assert [s['id'] for s in result['answer']['sources']] == ['D0', 'D1']
    assert set(result['telemetry']) >= {'plan', 'score', 'answer', 'compose'}


async def test_need_evidence_is_capped_and_comparison_only_when_strong_spans_exceed_cap():
    strong = [source(i, f'答案1第{i}处。') for i in range(5)]
    model = Model()
    await run(model, {'q': strong}, supplementary=False)
    picks = [d for i, d, _ in model.calls if i == flow.PICK]
    assert len(picks) == 1 and len(picks[0]['spans']) == 5
    answer = next(d for i, d, _ in model.calls if i == flow.ANSWER)
    assert len(answer['spans']) == 1  # The comparison chose one span.
    model = Model()
    await run(model, {'q': [source(0, '答案1。'), source(1, '线索1。'), source(2, '线索1又一处。'),
                            source(3, '线索1第三处。')]}, supplementary=False)
    assert not any(i == flow.PICK for i, _, _ in model.calls)
    answer = next(d for i, d, _ in model.calls if i == flow.ANSWER)
    assert [s['text'] for s in answer['spans']] == ['答案1。', '线索1。', '线索1又一处。']


async def test_need_without_evidence_skips_answer_call_and_triggers_requery_reading_only_new():
    model = Model(needs=['A', 'B'], requery=[['q2'], []])
    result, calls = await run(model, {'q': [source(0, '答案1。')], 'q2': [source(0, '答案1。'), source(1, '答案2。')]})
    assert calls == ['q', 'q2']
    scored = [d['source']['title'] for i, d, _ in model.calls if i == flow.SCORE]
    assert scored == ['资料0', '资料1']
    answered = [d['need']['id'] for i, d, _ in model.calls if i == flow.ANSWER]
    assert answered == ['I1', 'I2']  # I2 had no evidence first; answered after the re-query only.
    assert result['answer']['generation']['need_answers'] == {'I1': 'found', 'I2': 'found'}


async def test_found_without_citation_is_invalid_and_raw_kept():
    class NoCite(Model):
        async def complete(self, messages, **kwargs):
            if messages[0]['content'].startswith(flow.ANSWER):
                self.calls.append((flow.ANSWER, None, kwargs))
                return reply({'status': 'found', 'answer': '答', 'span_ids': []})
            return await super().complete(messages, **kwargs)
    model = NoCite()
    result, _ = await run(model, {'q': [source(0, '答案1。')]}, supplementary=False)
    node = next(n for n in result['graph']['nodes'] if n['kind'] == 'answer')
    assert node['answer_status'] == 'failed' and node['raw_text'].startswith('{"status": "found"')
    assert not any(i == flow.COMPOSE for i, _, _ in model.calls)  # No evidence: nothing composed.
    gen = result['answer']['generation']
    assert gen['status'] == 'no_evidence' and result['answer']['answer'] == ''


async def test_every_call_input_stays_small():
    docs = [source(i, ('答案1' if i % 3 == 0 else '线索1') + '内容' * 1500) for i in range(12)]
    model = Model()
    await run(model, {'q': docs}, supplementary=False)
    for instruction, data, _ in model.calls:
        size = len(json.dumps(data, ensure_ascii=False))
        assert size < 9000, (instruction[:6], size)


def card(i, category, summary=''):
    return {'id': f'F{i}', 'title': f'文档{i}', 'summary': summary, 'entities': [], 'category': category}


class NavModel:
    def __init__(self, categories, wanted):
        self.categories, self.wanted, self.calls = categories, wanted, []

    async def complete(self, messages, **kwargs):
        instruction, data = messages[0]['content'].split('\n', 1)
        data = json.loads(data)
        self.calls.append((instruction, data, kwargs))
        limit = kwargs['structured_schema']['properties']['ids']['maxItems'] if 'ids' in kwargs['structured_schema']['properties'] else 0
        if instruction == nav.CATEGORY_PICK:
            return reply({'ids': self.categories})
        if instruction in {nav.CARD_PICK} or instruction.startswith(nav.FINAL_PICK[:20]):
            ids = [d['id'] for d in data['documents'] if d['title'] in self.wanted][:limit]
            return reply({'ids': ids})
        if instruction == nav.CARD:
            return reply({'summary': '摘要', 'entities': ['甲'], 'category': '历史'})
        raise AssertionError(instruction)


async def test_navigation_tournament_uses_short_ids_small_batches_and_caps_files():
    cards = [card(i, '历史' if i % 10 == 0 else '体育') for i in range(500)]
    wanted = {f'文档{i}' for i in range(0, 500, 10)}
    model = NavModel(['历史'], wanted)
    selected, receipt = await nav.choose(model, '问题', cards, use_categories=True)
    picks = [(i, d, k) for i, d, k in model.calls if i != nav.CATEGORY_PICK]
    assert all(len(d['documents']) <= nav.CARD_BATCH for _, d, _ in picks)
    assert all(doc['id'].startswith('D') for _, d, _ in picks for doc in d['documents'])
    first = [d for i, d, _ in picks if i == nav.CARD_PICK]
    assert {doc['title'] for d in first for doc in d['documents']} == wanted  # Only chosen category.
    assert len(selected) == nav.MAX_FILES and all(s in {c['id'] for c in cards if c['title'] in wanted} for s in selected)
    assert receipt['categories'] == ['历史']


async def test_small_catalog_skips_category_step_and_failed_card_stays_reachable():
    model = NavModel([], {'文档1'})
    selected, receipt = await nav.choose(model, '问题', [card(i, '其他') for i in range(5)])
    assert selected == ['F1'] and 'categories' not in receipt

    class Fail:
        async def complete(self, messages, **kwargs):
            return NativeRWKVResult('length', '{"summ', 'length', {})
    built = await nav.build_card(Fail(), 'F9', '标题', '正文')
    assert built['card_failed'] and built['category'] == '其他' and built['raw_text'] == '{"summ'


async def test_composition_sees_only_found_answers_and_missing_needs_by_name():
    model = Model(needs=['A', 'B'], answer_status={'I2': 'not_found'})
    await run(model, {'q': [source(0, '答案1。'), source(1, '答案2。')]}, supplementary=False)
    data = next(d for i, d, _ in model.calls if i == flow.COMPOSE)
    assert [a['need_id'] for a in data['answers']] == ['I1'] and data['not_found'] == ['B']


async def test_failed_reading_call_keeps_raw_and_failure_status():
    class Failed(Model):
        async def complete(self, messages, **kwargs):
            if messages[0]['content'].startswith(flow.SCORE):
                return NativeRWKVResult('length', '{"scores":[', 'length', {})
            return await super().complete(messages, **kwargs)
    result, _ = await run(Failed(), {'q': [source(0, '答案1。')]}, supplementary=False)
    leaf = next(n for n in result['graph']['nodes'] if n['kind'] == 'leaf')
    assert leaf['raw_text'] == '{"scores":[' and leaf['status'] == 'length'
    gen = result['answer']['generation']
    assert leaf['id'] in gen['failed_nodes']
    assert gen['status'] == 'no_evidence' and gen['raw_model_answer'] is None


async def test_comparison_cannot_overflow_three_span_answer_budget():
    model = Model(pick=lambda data: [s['id'] for s in data['spans']])
    result, _ = await run(model, {'q': [source(i, '答案1。') for i in range(5)]}, supplementary=False)
    pick = next(n for n in result['graph']['nodes'] if n['kind'] == 'pick')
    assert pick['status'] == 'invalid_output' and pick['parsed'] is None
    assert len(json.loads(pick['raw_text'])['span_ids']) == 5
    answer = next(d for i, d, _ in model.calls if i == flow.ANSWER)
    assert len(answer['spans']) == 3


async def test_trial_needs_tokenizer_budget_blocks_generation_and_preserves_trace(tmp_path, monkeypatch):
    import httpx
    from llamaindex_retrieval import interactive_app as app
    requests = []

    def handler(request):
        requests.append(request.url.path)
        assert request.url.path == '/tokenize'
        return httpx.Response(200, json={'count': 6001, 'tokens': [1] * 6001, 'max_model_len': 25600})

    monkeypatch.setattr(app.httpx, 'AsyncHTTPTransport', lambda **_: httpx.MockTransport(handler))
    trial = app.Trial({'pipeline': 'needs', 'settings': {'native_model': 'test-model',
        'native_completion_protocol': 'g1j_plain', 'native_context_window_tokens': 25600,
        'native_transport': 'native', 'native_planner_prefill': '<think></think',
        'native_resolver_prefill': '<think></think', 'native_writer_prefill': '<think></think'},
        'admin_base_url': 'http://localhost:1', 'records_dir': str(tmp_path)})
    observed = app.ObservedModel(trial.native, {})
    try:
        assert trial.native.context_window_tokens == 6000
        event = await flow.generate(observed, 'instruction', {}, {'answer': {'type': 'string'}}, limit=16)
        assert event['status'] == 'budget_exceeded' and event['raw_text'] is None
        assert event['trace']['budget']['input_tokens'] == 6001
        assert not event['trace']['completion_attempted']
        assert requests == ['/tokenize']
        stats = flow.telemetry(observed.events)['other']
        assert stats['max_generation_tokens'] is None and stats['max_not_sent_tokens'] == 6001
    finally:
        await trial.native.aclose()
        await trial.admin.aclose()


async def test_component_payload_builders_are_byte_identical_to_runtime_calls():
    class Capture(Model):
        async def complete(self, messages, **kwargs):
            self.wires = getattr(self, 'wires', []) + [(messages[0]['content'], kwargs)]
            return await super().complete(messages, **kwargs)
    model = Capture()
    doc = source(0, '答案1。')
    await run(model, {'q': [doc]}, supplementary=False)
    score_text, score_kwargs = next(x for x in model.wires if x[0].startswith(flow.SCORE))
    payload = flow.score_input('问题', [{'id': 'I1', 'text': 'A'}], doc,
        json.loads(score_text.split('\n', 1)[1])['source']['parent_context'], [{'id': 'S1', 'text': '答案1。'}])
    assert score_text == flow.SCORE + '\n' + json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    assert score_kwargs['structured_schema']['properties'] == flow.score_schema(['S1'], ['I1'])
    answer_text, answer_kwargs = next(x for x in model.wires if x[0].startswith(flow.ANSWER))
    payload = flow.answer_input('问题', {'id': 'I1', 'text': 'A'}, [
        {'id': 'S1', 'source': doc.title, 'channel': doc.source, 'uri': doc.uri, 'text': '答案1。'}])
    assert answer_text == flow.ANSWER + '\n' + json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    assert answer_kwargs['structured_schema']['properties'] == flow.answer_schema(['S1'])
