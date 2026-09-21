import asyncio
import json

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.native_rwkv import NativeRWKVResult
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline
from llamaindex_retrieval.schemas import SourceItem


def source(i, length=100, document=None):
    return SourceItem(id=str(i), document_id=document or str(i), source='test', title=str(i),
                      snippet=str(i) * length, score=1, metadata={})


class Model:
    def __init__(self, capacity, failure=None):
        self.capacity = capacity
        self.calls = []
        self.failure = failure

    async def complete(self, messages, **kwargs):
        evidence = json.loads(messages[0]['content'].split('逐字证据：')[1])
        self.calls.append((evidence, kwargs))
        status = self.failure or ('completed' if sum(len(s['text']) for s in evidence) <= self.capacity else 'budget_exceeded')
        raw = '原文不改。[资料 2]' if not kwargs.get('check_only') else None
        return NativeRWKVResult(status, raw, 'stop' if raw else None,
            {'stage': kwargs['stage'], 'status': status, 'raw_text': raw,
             'evidence_ids': list(kwargs['evidence_ids']), 'completion_attempted': raw is not None})


def run(sources, capacity=250, failure=None):
    model = Model(capacity, failure)
    pipeline = RWKVPipeline(Settings(native_writer_budget_policy='whole_sources'), None, model=model)
    result = asyncio.run(pipeline.ask_materials('比较全部对象', sources))
    json.dumps(result.model_dump())  # No cyclic trace, including terminal budget failure.
    return result, model


def test_whole_sources_diversity_and_actual_citation_map():
    sources = [source(1, document='a'), source(2, document='a'), source(3, document='b')]
    result, model = run(sources)
    assert [s.id for s in result.sources] == ['1', '3']
    assert [s.snippet for s in result.sources] == [sources[0].snippet, sources[2].snippet]
    assert result.generation['citation_map'] == {'1': '1', '2': '3'}
    assert result.answer == result.generation['raw_model_answer'] == '原文不改。[资料 2]'
    assert result.retrieval['writer_evidence_budget']['omitted_source_ids'] == ['2']
    assert sum(not kw.get('check_only', False) for _, kw in model.calls) == 1
    assert model.calls[-1][0][1]['id'] == '3'
    assert model.calls[-1][0][1]['label'] == '资料 2'


def test_fitting_input_is_not_reordered_or_changed():
    result, model = run([source(1), source(2)], 1000)
    assert [s.id for s in result.sources] == ['1', '2']
    assert len(model.calls) == 2
    assert model.calls[0][0] == model.calls[1][0]
    assert result.retrieval['writer_evidence_budget']['coverage_complete']


def test_oversized_unit_does_not_hide_small_later_unit():
    result, _ = run([source(1, 1000), source(2, 100)], 150)
    assert [s.id for s in result.sources] == ['2']


def test_no_whole_source_fits_is_not_empty_evidence_refusal():
    result, model = run([source(1, 1000)], 150)
    assert result.generation['status'] == 'budget_exceeded'
    assert result.answer == ''
    assert all(kw.get('check_only') for _, kw in model.calls)
    assert result.retrieval['writer_evidence_budget']['status'] == 'no_whole_source_fits'


def test_tokenizer_failure_does_not_generate_or_drop_into_fallback():
    result, model = run([source(1)], failure='timeout')
    assert result.generation['status'] == 'timeout'
    assert len(model.calls) == 1
    assert result.retrieval['writer_evidence_budget']['status'] == 'tokenizer_failed'
