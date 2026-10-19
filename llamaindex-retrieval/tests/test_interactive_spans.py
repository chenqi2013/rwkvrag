"""Planned queries, span-only lineage, merge bounds, need-driven supplementary rounds."""
import asyncio
import json

import pytest

from llamaindex_retrieval import interactive_spans as flow
from llamaindex_retrieval.native_rwkv import NativeRWKVResult
from llamaindex_retrieval.schemas import SourceItem


def source(i, text):
    return SourceItem(id=f'D{i}', document_id=f'doc{i}', title=f'资料{i}', source='web',
                      uri=f'https://example.org/{i}', score=0, snippet=text)


def paragraphs(tag, count):
    return '\n\n'.join(f'{tag}第{j}段：' + '内容' * 200 for j in range(count))


class Model:
    """Leaf selects spans containing '答案'; merges keep everything and report no missing."""

    def __init__(self, plan=None, requery=(), missing=None):
        self.calls = []
        self.plan = plan or {'needs': ['信息一'], 'queries': ['查询一']}
        self.requery, self.missing = list(requery), missing

    async def complete(self, messages, **kwargs):
        instruction, data = messages[0]['content'].split('\n', 1)
        data = json.loads(data)
        self.calls.append((instruction, data, kwargs))
        await asyncio.sleep(0)
        fields = [k for k in kwargs['structured_schema']['properties'] if k.endswith('_queries')]
        if instruction == flow.PLAN:
            value = dict(self.plan)
            if 'queries' in value:  # Shorthand: plain queries go to every offered channel.
                queries = value.pop('queries')
                value.update({f: queries for f in fields})
        elif instruction == flow.REQUERY:
            queries = self.requery.pop(0) if self.requery else []
            value = {f: queries for f in fields}
        elif instruction == flow.LEAF:
            ids = [s['id'] for s in data['spans'] if '答案' in s['text']]
            value = {'span_ids': ids, 'need_ids': [n['id'] for n in data['needs']] if ids else []}
        elif instruction == flow.MERGE:
            value = {'span_ids': [s['id'] for s in data['spans']],
                     'missing': self.missing if self.missing is not None else []}
        else:
            value = {'answer': '回答', 'source_ids': [s['id'] for s in data['spans']][:1]}
        return NativeRWKVResult('completed', json.dumps(value, ensure_ascii=False), 'stop', {})


def acquirer(table):
    calls = []

    async def acquire(query, channels=None):
        calls.append(query)
        return table.get(query, [])
    return acquire, calls


async def run(model, table, **kwargs):
    acquire, calls = acquirer(table)
    result = await flow.answer(model, '问题', acquire=acquire, scope={'web': True},
                               progress=lambda _: None, **kwargs)
    return result, calls


async def test_model_queries_drive_acquisition_and_only_selected_spans_reach_final():
    docs = [source(0, paragraphs('无关', 3)), source(1, '这里有答案。')]
    model = Model(plan={'needs': ['A'], 'queries': ['q1', 'q2']})
    result, calls = await run(model, {'q1': docs[:1], 'q2': docs[1:]}, supplementary=False)
    assert calls == ['q1', 'q2']
    final = next(d for i, d, _ in model.calls if i == flow.FINAL)
    assert [s['text'] for s in final['spans']] == ['这里有答案。']
    answer = result['answer']
    assert [s['id'] for s in answer['sources']] == ['D1']
    cited = answer['generation']['parsed_answer']['source_ids']
    assert answer['generation']['citation_bindings'] == {cited[0]: ['D1']}
    assert answer['generation']['cited_spans'][cited[0]]['text'] == '这里有答案。'
    assert answer['answer'] == result['graph']['nodes'][-1]['raw_text']


async def test_spans_are_verbatim_and_cover_every_window():
    text = paragraphs('甲', 12)
    model = Model()
    result, _ = await run(model, {'查询一': [source(0, text)]}, supplementary=False)
    spans = result['graph']['spans']
    covered = set()
    for span in spans.values():
        covered.update(range(span['span']['start'], span['span']['end']))
    assert covered >= {i for i, c in enumerate(text) if not c.isspace()}
    leaves = [n for n in result['graph']['nodes'] if n['kind'] == 'leaf']
    assert len(leaves) > 1 and all(len(n['offered_span_ids']) > 1 for n in leaves)


async def test_failed_plan_falls_back_to_resolved_question_and_is_recorded():
    class BadPlan(Model):
        async def complete(self, messages, **kwargs):
            if messages[0]['content'].startswith(flow.PLAN):
                return NativeRWKVResult('length', '{"needs":[', 'length', {})
            return await super().complete(messages, **kwargs)
    result, calls = await run(BadPlan(), {}, supplementary=False)
    assert calls == ['问题']
    assert result['plan']['status'] == 'length'
    assert result['needs'] == {'I1': '问题'}


@pytest.mark.parametrize('count', [4, 10, 31])
async def test_merges_have_at_most_three_children_and_skip_empty_leaves(count):
    docs = [source(i, f'第{i}份答案。') for i in range(count)] + [source(99, '无关材料。')]
    model = Model()
    result, _ = await run(model, {'查询一': docs}, supplementary=False)
    nodes = result['graph']['nodes']
    assert all(len(n['parents']) <= 3 for n in nodes)
    empty = next(n['id'] for n in nodes if n['kind'] == 'leaf' and not n['span_ids'])
    assert all(empty not in n['parents'] for n in nodes)
    merges = [d for i, d, _ in model.calls if i == flow.MERGE]
    assert all('无关' not in s['text'] for d in merges for s in d['spans'])


async def test_missing_need_triggers_requery_and_only_new_sources_are_read():
    first, second = source(0, '旧答案。'), source(1, '新答案。')
    model = Model(plan={'needs': ['A', 'B'], 'queries': ['q1']}, requery=[['q2'], []])

    # Leaves cover only I1 so the second need is missing after the first round.
    original = model.complete

    async def complete(messages, **kwargs):
        instruction, data = messages[0]['content'].split('\n', 1)
        if instruction == flow.LEAF:
            data = json.loads(data)
            model.calls.append((instruction, data, kwargs))
            ids = [s['id'] for s in data['spans']]
            return NativeRWKVResult('completed', json.dumps({'span_ids': ids, 'need_ids': ['I1']}), 'stop', {})
        return await original(messages, **kwargs)
    model.complete = complete
    result, calls = await run(model, {'q1': [first], 'q2': [first, second]})
    assert calls == ['q1', 'q2']
    leaves = [d for i, d, _ in model.calls if i == flow.LEAF]
    assert [d['source']['title'] for d in leaves] == ['资料0', '资料1']
    assert result['rounds'][0]['missing'] == ['I2']
    assert result['rounds'][0]['new_source_ids'] == ['D1']
    assert result['stop_reason'] == 'model_stopped'
    assert result['executed_queries'] == [[('web', 'q1')], [('web', 'q2')]]


async def test_round_limit_and_unparsed_raw_is_kept():
    model = Model(plan={'needs': ['A'], 'queries': ['q1']}, requery=[['q2'], ['q3'], ['q4']])

    async def nothing(messages, **kwargs):
        instruction = messages[0]['content'].split('\n', 1)[0]
        if instruction == flow.LEAF:
            return NativeRWKVResult('completed', 'not json', 'stop', {})
        return await Model.complete(model, messages, **kwargs)
    model.complete = nothing
    result, calls = await run(model, {'q1': [source(0, '材料。')], 'q2': [source(1, '材料二。')],
                                      'q3': [source(2, '材料三。')]}, max_rounds=2)
    assert calls == ['q1', 'q2', 'q3']
    assert result['stop_reason'] == 'round_limit'
    leaf = next(n for n in result['graph']['nodes'] if n['kind'] == 'leaf')
    assert leaf['raw_text'] == 'not json' and leaf['status'] == 'invalid_output'
    final = next(d for i, d, _ in model.calls if i == flow.FINAL)
    assert final['spans'] == []
    assert result['answer']['generation']['missing_need_ids'] == ['I1']


async def test_every_stage_has_a_native_state_role():
    from llamaindex_retrieval.native_state import NativeStateRouter
    router = NativeStateRouter({'model': 'm', 'roles': {'plan': 'p', 'reader': 'r', 'writer': 'w'}},
                               model='m', prompt_protocol='g1j_plain')
    model = Model(requery=[['q2']], missing=['I1'])
    await run(model, {'查询一': [source(0, '答案。')], 'q2': [source(1, '答案二。')]})
    for stage in {kwargs['stage'] for _, _, kwargs in model.calls}:
        router.select(stage, None)  # Raises for stages the native client would reject.


async def test_exact_duplicate_needs_and_queries_are_dropped():
    model = Model(plan={'needs': ['A', 'A', 'B'], 'queries': ['q', 'q']})
    result, calls = await run(model, {'q': [source(0, '答案。')]}, supplementary=False)
    assert calls == ['q']
    assert result['needs'] == {'I1': 'A', 'I2': 'B'}
    assert result['plan']['parsed']['web_queries'] == ['q', 'q']  # Raw plan retained as produced.


async def test_long_page_outline_reads_only_selected_sections():
    page = '# 总标题\n\n' + '\n\n'.join(f'## 章节{i}\n\n' + ('答案' if i == 3 else '无关') + '内容' * 2000
                                         for i in range(6))
    class Pick(Model):
        async def complete(self, messages, **kwargs):
            instruction, data = messages[0]['content'].split('\n', 1)
            if instruction == flow.OUTLINE:
                data = json.loads(data)
                self.calls.append((instruction, data, kwargs))
                ids = [e['id'] for e in data['entries'] if '章节3' in e['heading']]
                return NativeRWKVResult('completed', json.dumps({'entry_ids': ids}), 'stop', {})
            return await super().complete(messages, **kwargs)
    model = Pick()
    result, _ = await run(model, {'查询一': [source(0, page)]}, supplementary=False, outline_chars=12000)
    outline = result['outlines'][0]
    assert {i for r in outline['entries'] for i in range(r['start'], r['end'])} >= {
        i for i, c in enumerate(page) if not c.isspace()}
    leaves = [d for i, d, _ in model.calls if i == flow.LEAF]
    assert leaves and all('无关' not in s['text'] for d in leaves for s in d['spans'])


async def test_outline_off_by_default():
    model = Model()
    result, _ = await run(model, {'查询一': [source(0, '答案' * 20000)]}, supplementary=False)
    assert result['outlines'] == []
    assert not any(i == flow.OUTLINE for i, _, _ in model.calls)


async def test_page_gate_skips_irrelevant_pages_and_failed_gate_keeps_page():
    class Gate(Model):
        async def complete(self, messages, **kwargs):
            instruction, data = messages[0]['content'].split('\n', 1)
            if instruction == flow.PAGE:
                data = json.loads(data)
                self.calls.append((instruction, data, kwargs))
                title = data['source']['title']
                if title == '资料2':
                    return NativeRWKVResult('length', '{"verd', 'length', {})
                score = '0' if title == '资料1' else '1'
                return NativeRWKVResult('completed', json.dumps({'score': score}), 'stop', {})
            return await super().complete(messages, **kwargs)
    model = Gate()
    docs = [source(0, '答案零。'), source(1, '答案一。'), source(2, '答案二。')]
    result, _ = await run(model, {'查询一': docs}, supplementary=False, page_gate=True)
    read = [d['source']['title'] for i, d, _ in model.calls if i == flow.LEAF]
    assert sorted(read) == ['资料0', '资料2']
    assert [g['keep'] for g in result['page_gates']] == [True, False, True]
    assert result['page_gates'][2]['parsed'] is None


async def test_model_assigns_queries_to_channels_and_only_permitted_fields_exist():
    seen = []

    async def acquire(query, channels=None):
        seen.append((query, channels))
        return []
    model = Model(plan={'needs': ['A'], 'kb_queries': ['kb问'], 'web_queries': ['web问', 'web问']})
    result = await flow.answer(model, '问题', acquire=acquire, scope={'knowledge_bases': ['库'], 'web': True},
                               progress=lambda _: None, supplementary=False)
    assert seen == [('kb问', ['knowledge_base']), ('web问', ['web'])]
    assert result['executed_queries'] == [[('knowledge_base', 'kb问'), ('web', 'web问')]]
    model = Model(plan={'needs': ['A'], 'kb_queries': []})
    seen.clear()
    await flow.answer(model, '问题', acquire=acquire, scope={'knowledge_bases': ['库'], 'web': False},
                      progress=lambda _: None, supplementary=False)
    plan_schema = next(k for i, _, k in model.calls if i == flow.PLAN)['structured_schema']
    assert set(plan_schema['properties']) == {'needs', 'kb_queries'}
    assert seen == [('问题', ['knowledge_base'])]  # Empty plan falls back to the question.


async def test_final_citation_array_is_bounded_by_offered_spans():
    model = Model()
    await run(model, {'查询一': [source(0, '答案一。'), source(1, '答案二。')]}, supplementary=False)
    kwargs = next(k for i, _, k in model.calls if i == flow.FINAL)
    ids = kwargs['structured_schema']['properties']['source_ids']
    assert ids['maxItems'] == len(ids['items']['enum']) == 2
