"""Bounded prompts, complete scheduling, raw/lineage and context isolation."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from llamaindex_retrieval import interactive_app as api
from llamaindex_retrieval import interactive_funnel as flow
from llamaindex_retrieval import interactive_sources as sources_flow
from llamaindex_retrieval.native_rwkv import NativeRWKVResult
from llamaindex_retrieval.schemas import SourceItem


def sources(count, size=1000):
    return [SourceItem(id=f'D{i}', document_id=f'doc{i}', title=f'资料{i}',
        source='provided', score=0, snippet=f'原文{i}：'+('甲乙丙丁' * size)) for i in range(count)]


class Model:
    def __init__(self, queries=(), fail_leaf=False):
        self.calls, self.queries = [], list(queries)
        self.checks = 0
        self.fail_leaf = fail_leaf

    async def complete(self, messages, **kwargs):
        instruction, data = messages[0]['content'].split('\n', 1)
        data = json.loads(data)
        self.calls.append((instruction, data, kwargs))
        await asyncio.sleep(0)
        if instruction == flow.CONTEXT:
            value = {'question': data['latest_question']}
        elif instruction == sources_flow.CATALOG:
            value = {'source_ids': [d['id'] for d in data['documents']]}
        elif instruction == flow.CHECK:
            self.checks += 1
            value = {'questions': self.queries if self.checks == 1 else []}
        elif instruction == flow.LEAF:
            if self.fail_leaf:
                return NativeRWKVResult('length', 'FAILED RAW MUST NOT ENTER PARENT', 'length', {})
            value = {'summary': '模型总结', 'source_ids': [data['original']['id']]}
        else:
            assert len(data['children']) <= 3
            if instruction == flow.MERGE:
                assert '原文' not in json.dumps(data, ensure_ascii=False)
            final = instruction in {flow.FINAL, flow.FINAL_ORIGINALS}
            value = {'answer' if final else 'summary': '模型归并',
                'source_ids': [c['id'] for c in data['children'] if c['status'] == 'completed']}
        raw = ' ' + json.dumps(value, ensure_ascii=False) + '\n'
        return NativeRWKVResult('completed', raw, 'stop', kwargs.get('trace', {}))


@pytest.mark.parametrize('count', [1, 4, 33, 65])
async def test_all_documents_summarized_and_every_merge_has_at_most_three_children(count):
    model = Model()
    docs = sources(count)
    result = await flow.answer(model, '总结多份资料', docs, acquire=None, scope={},
                               progress=lambda _: None, supplementary=False)
    graph = result['funnel']['nodes']
    leaves = [n for n in graph if n['kind'] == 'leaf']
    assert {sid for n in leaves for sid in n['original_text_input_ids']} == {s.id for s in docs}
    for doc in docs:
        spans = [n['source_span'] for n in leaves if doc.id in n['original_text_input_ids']]
        coverage = set(i for s in spans for i in range(s['start'], s['end']))
        assert coverage == set(range(len(doc.snippet)))
    assert all(len(n['parents']) <= 3 for n in graph)
    assert graph[-1]['kind'] == 'final'  # n=1 also gets independent final generation.
    assert all(n['original_text_input_ids'] == [] for n in graph if n['kind'] != 'leaf')
    assert result['answer']['answer'] == graph[-1]['raw_text']
    assert {s['id'] for s in result['answer']['sources']} == {s.id for s in docs}
    assert result['subquestion_count_limit'] is None


async def test_failed_leaf_raw_is_retained_but_only_status_reaches_parent():
    model = Model(fail_leaf=True)
    result = await flow.answer(model, '问题', sources(1, 5), acquire=None, scope={},
                               progress=lambda _: None, supplementary=False)
    assert result['funnel']['nodes'][0]['raw_text'] == 'FAILED RAW MUST NOT ENTER PARENT'
    assert 'FAILED RAW' not in json.dumps(model.calls[-1][1])
    assert result['answer']['generation']['status'] == 'partial'
    assert result['answer']['sources'] == []


async def test_followup_has_no_task_cap_and_uses_funnel_again():
    model = Model([f'子问题{i}' for i in range(65)])
    queries = []
    async def acquire(q):
        queries.append(q)
        return []
    result = await flow.answer(model, '问题', sources(1, 5), acquire=acquire, scope={}, progress=lambda _: None)
    assert queries == model.queries
    assert len([c for c in model.calls if c[0] == flow.LEAF]) == 66
    assert result['stop_reason'] == 'model_stopped'
    assert len([n for n in result['funnel']['nodes'] if n['kind'] == 'final']) == 1
    for _, _, kwargs in (c for c in model.calls if c[0] == flow.CHECK):
        assert 'maxItems' not in kwargs['structured_schema']['properties']['questions']


async def test_context_only_receives_explicit_parent_and_never_looks_up_old_trials():
    model = Model()
    _, event = await flow.contextualize(model, '新的话题')
    assert model.calls[-1][1]['previous_context'] is None
    await flow.contextualize(model, '它呢', {'context': '当前对象', 'answer': '上一轮回答'})
    assert model.calls[-1][1]['previous_context'] == '当前对象'
    assert model.calls[-1][1]['previous_answer_for_reference_only'] == '上一轮回答'
    assert event['raw_text'].startswith(' ')


async def test_every_catalog_title_is_offered_and_source_ids_cannot_be_rebound():
    model = Model()
    catalog = [{'id': str(i), 'title': f'资料{i}'} for i in range(300)]
    ids, _ = await sources_flow.choose_catalog(model, '问题', catalog)
    assert ids == [c['id'] for c in catalog]
    pool = sources(1, 1)
    with pytest.raises(ValueError, match='reused'):
        sources_flow.add_sources(pool, [pool[0].model_copy(update={'snippet': 'different'})])


async def test_cancel_before_start_persists_without_model_calls(monkeypatch, tmp_path):
    async def never(job, payload):
        pytest.fail('cancelled worker ran')
    trial = SimpleNamespace(jobs={}, tasks={}, folder=tmp_path, run=never)
    monkeypatch.setattr(api.app.state, 'trial', trial, raising=False)
    job = await api.ask(api.Question(question='问题'))
    assert (await api.cancel(job['id']))['status'] == 'cancelled'
    assert json.loads((tmp_path / (job['id']+'.json')).read_text())['model_calls'] == []


def test_removed_direct_and_comparison_options_are_rejected():
    from pydantic import ValidationError
    for old in [{'mode': 'compare'}, {'pipeline': 'pr'}]:
        with pytest.raises(ValidationError):
            api.Question(question='问题', **old)


async def test_final_receives_selected_verbatim_originals_within_budget():
    model = Model()
    docs = sources(6, 500)  # one 2003-char leaf per document
    result = await flow.answer(model, '问题', docs, acquire=None, scope={},
                               progress=lambda _: None, supplementary=False, original_chars=7000)
    graph = result['funnel']['nodes']
    leaves = [n for n in graph if n['kind'] == 'leaf']
    final = graph[-1]
    final_data = next(d for i, d, _ in model.calls if i == flow.FINAL_ORIGINALS)
    assert final['original_leaf_input_ids'] == [n['id'] for n in leaves[:3]]
    assert final['omitted_original_leaf_ids'] == [n['id'] for n in leaves[3:]]
    assert [o['text'] for o in final_data['originals']] == [d.snippet for d in docs[:3]]
    assert final['original_text_input_ids'] == [d.id for d in docs[:3]]
    generation = result['answer']['generation']
    assert generation['omitted_original_leaf_ids'] == final['omitted_original_leaf_ids']
    assert generation['protocol'] == flow.ORIGINALS_PROTOCOL


async def test_final_may_cite_original_ids_and_binds_them_to_sources():
    class CitingModel(Model):
        async def complete(self, messages, **kwargs):
            instruction, data = messages[0]['content'].split('\n', 1)
            if instruction != flow.FINAL_ORIGINALS:
                return await super().complete(messages, **kwargs)
            data = json.loads(data)
            self.calls.append((instruction, data, kwargs))
            assert set(kwargs['structured_schema']['properties']['source_ids']['items']['enum']) >= {
                o['id'] for o in data['originals']}
            value = {'answer': '答案', 'source_ids': [data['originals'][1]['id']]}
            return NativeRWKVResult('completed', json.dumps(value, ensure_ascii=False), 'stop', {})
    docs = sources(2, 5)
    result = await flow.answer(CitingModel(), '问题', docs, acquire=None, scope={},
                               progress=lambda _: None, supplementary=False, original_chars=16000)
    answer = result['answer']
    assert [s['id'] for s in answer['sources']] == [docs[1].id]
    cited = answer['generation']['parsed_answer']['source_ids'][0]
    assert answer['generation']['citation_bindings'] == {cited: [docs[1].id]}


async def test_unselected_leaf_original_never_reaches_final():
    class SkippingModel(Model):
        async def complete(self, messages, **kwargs):
            instruction, data = messages[0]['content'].split('\n', 1)
            data = json.loads(data)
            if instruction == flow.LEAF and data['original']['title'] == '资料0':
                self.calls.append((instruction, data, kwargs))
                return NativeRWKVResult('completed', '{"summary":"","source_ids":[]}', 'stop', {})
            return await super().complete(messages, **kwargs)
    model = SkippingModel()
    await flow.answer(model, '问题', sources(2, 5), acquire=None, scope={},
                      progress=lambda _: None, supplementary=False, original_chars=16000)
    final_data = next(d for i, d, _ in model.calls if i == flow.FINAL_ORIGINALS)
    assert [o['title'] for o in final_data['originals']] == ['资料1']


async def test_final_originals_are_off_by_default():
    model = Model()
    result = await flow.answer(model, '问题', sources(2, 5), acquire=None, scope={},
                               progress=lambda _: None, supplementary=False)
    final_call = next(d for i, d, _ in model.calls if i == flow.FINAL)
    assert 'originals' not in final_call
    assert not any(i == flow.FINAL_ORIGINALS for i, _, _ in model.calls)
    assert result['answer']['generation']['protocol'] == flow.PROTOCOL
