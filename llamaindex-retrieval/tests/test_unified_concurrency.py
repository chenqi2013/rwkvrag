"""Scheduler overlap/budget/provenance tests, not model quality evidence."""
import asyncio
from collections import Counter

import pytest
from pydantic import ValidationError

from llamaindex_retrieval.unified_retrieval import UnifiedRetrievalPipeline
from llamaindex_retrieval.unified_contracts import UnifiedPolicy
from test_unified_retrieval import material
from test_unified_span_pipeline import SpanModel, settings


class DelayedModel(SpanModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active = Counter()
        self.peak = Counter()
        self.finished = []

    async def complete(self, messages, **options):
        stage = options['stage'].removeprefix('unified_')
        node = options['trace'].get('node_id')
        self.active[stage] += 1
        self.peak[stage] = max(self.peak[stage], self.active[stage])
        try:
            # Reverse finish order within each wave to exercise grouping/binding.
            await asyncio.sleep(.003 * (5 - int(node[1:]) % 4) if node else .001)
            result = await super().complete(messages, **options)
            self.finished.append(node)
            return result
        finally:
            self.active[stage] -= 1


async def test_independent_readers_and_same_level_merges_overlap_in_stable_tree_order():
    model = DelayedModel()
    result = await UnifiedRetrievalPipeline(settings(stage_concurrency=4), None, model=model).ask_materials(
        '目标', [material(i) for i in range(12)])
    nodes = result.generation['nodes']
    assert model.peak['leaf'] == 4 and model.peak['merge'] == 4
    assert model.peak['final'] == 1
    assert not any(model.active.values())
    assert model.finished[:4] != ['N1', 'N2', 'N3', 'N4']
    assert [n['original_evidence_id'] for n in nodes[:12]] == [f'E{i}' for i in range(1, 13)]
    assert [n['parents'] for n in nodes[12:16]] == [[f'N{i}' for i in range(s, s + 3)] for s in [1, 4, 7, 10]]
    assert set(result.generation['span_bindings']) == {f'E{i}.S1' for i in range(1, 13)}
    for call in model.calls:
        if call['purpose'] in {'merge', 'final'}:
            assert all(nodes[int(c['node_id'][1:]) - 1]['status'] == 'completed' for c in call['data']['children'])
            for key, binding in call['data']['inherited_span_bindings'].items():
                leaf = nodes[int(binding['leaf_node_id'][1:]) - 1]
                assert key in leaf['span_ids'] and binding['leaf_call_id'] == leaf['call_id']
    assert result.answer == nodes[-1]['raw_text']
    assert result.generation['request_budget']['stage_concurrency'] == 4


@pytest.mark.parametrize('limit', [2, 3, 5, 9, 17])
async def test_concurrency_retains_serial_read_population_budget_and_final_reservation(limit):
    responses = []
    for width in [1, 4]:
        model = DelayedModel()
        result = await UnifiedRetrievalPipeline(settings(stage_concurrency=width, max_calls=limit), None,
            model=model).ask_materials('目标', [material(i) for i in range(10)])
        assert result.generation['request_budget']['admitted_calls'] <= limit
        assert model.counts['final'] == 1
        responses.append(result)
    for field in ['admitted_calls']:
        assert responses[0].generation['request_budget'][field] == responses[1].generation['request_budget'][field]
    assert responses[0].generation['execution_scope']['unread_units'] == responses[1].generation['execution_scope']['unread_units']
    assert [n['parents'] for n in responses[0].generation['nodes']] == [n['parents'] for n in responses[1].generation['nodes']]


async def test_transport_failure_drains_current_wave_and_stops_new_waves_without_fallback():
    model = DelayedModel(overrides={('leaf', 1): ('http_error', None)})
    result = await UnifiedRetrievalPipeline(settings(stage_concurrency=4), None, model=model).ask_materials(
        '目标', [material(i) for i in range(12)])
    assert model.counts['leaf'] == 4
    assert model.counts['merge'] == model.counts['final'] == 0
    assert not any(model.active.values())
    assert result.generation['status'] == 'model_failed' and result.answer == ''
    assert all(n['status'] != 'pending' for n in result.generation['nodes'])


async def test_timeout_cancels_and_awaits_every_inflight_reader():
    entered = asyncio.Event()
    active = 0
    cancelled = 0
    async def slow(_):
        nonlocal active, cancelled
        active += 1
        if active == 4:
            entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled += 1
            raise
        finally:
            active -= 1
    model = SpanModel(overrides={'leaf': slow})
    result = await UnifiedRetrievalPipeline(settings(stage_concurrency=4, timeout_seconds=.05), None,
        model=model).ask_materials('目标', [material(i) for i in range(12)])
    assert entered.is_set() and cancelled == 4 and active == 0
    assert result.answer == '' and result.generation['status'] == 'request_timeout'
    assert model.counts['final'] == 0
    assert len(result.generation['nodes']) == 4
    assert all(n['status'] == 'cancelled' for n in result.generation['nodes'])


async def test_failed_sibling_raw_is_not_added_to_other_merge_inputs():
    model = DelayedModel(overrides={('leaf', 2): 'FAILED_PARALLEL_RAW'})
    result = await UnifiedRetrievalPipeline(settings(stage_concurrency=4), None, model=model).ask_materials(
        '目标', [material(i) for i in range(6)])
    assert any(n['raw_text'] == 'FAILED_PARALLEL_RAW' for n in result.generation['nodes'])
    assert all('FAILED_PARALLEL_RAW' not in c['options']['trace']['messages'][0]['content']
               for c in model.calls if c['purpose'] in {'merge', 'final'})
    assert result.generation['status'] == 'partial'


def test_concurrency_is_explicit_span_resource_control():
    assert UnifiedPolicy().stage_concurrency == 1
    for value in [0, 33, True]:
        with pytest.raises(ValidationError):
            UnifiedPolicy(reading_protocol='span_ids_v1', stage_concurrency=value)
    with pytest.raises(ValidationError, match='concurrent_stages_require_span_ids_v1'):
        UnifiedPolicy(stage_concurrency=4)


async def test_timeout_during_parallel_merges_keeps_completed_leaf_receipts_and_cancels_siblings():
    active = 0
    cancelled = 0
    async def slow(_):
        nonlocal active, cancelled
        active += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled += 1
            raise
        finally:
            active -= 1
    model = SpanModel(overrides={'merge': slow})
    result = await UnifiedRetrievalPipeline(settings(stage_concurrency=4, timeout_seconds=.05), None,
        model=model).ask_materials('目标', [material(i) for i in range(6)])
    assert cancelled == 2 and active == 0
    assert result.generation['status'] == 'request_timeout' and result.answer == ''
    assert result.generation['root_node_id'] is None
    assert all(n['status'] == 'completed' for n in result.generation['nodes'][:6])
    assert all(n['status'] == 'cancelled' for n in result.generation['nodes'][6:])


async def test_overlapping_requests_keep_their_own_nodes_and_source_bindings():
    model = DelayedModel()
    pipeline = UnifiedRetrievalPipeline(settings(stage_concurrency=4), None, model=model)
    first, second = await asyncio.gather(
        pipeline.ask_materials('甲', [material(i, text=f'甲的原文{i}。') for i in range(6)]),
        pipeline.ask_materials('乙', [material(i+20, text=f'乙的原文{i}。') for i in range(6)]))
    first_ids = {c['call_id'] for c in first.generation['model_calls']}
    second_ids = {c['call_id'] for c in second.generation['model_calls']}
    assert first_ids.isdisjoint(second_ids)
    for result, prefix in [(first, '甲'), (second, '乙')]:
        assert result.generation['request_budget']['admitted_calls'] == 9
        for binding in result.generation['span_bindings'].values():
            assert binding['selected_original_text'].startswith(prefix)
