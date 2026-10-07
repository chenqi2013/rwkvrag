"""Request bounds use fake models/HTTP only; no live inference or database writes."""
import asyncio

import httpx
import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.native_rwkv import NativeRWKVClient
from llamaindex_retrieval.repository import search_failure_category
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline, source_from_hit
from llamaindex_retrieval.schemas import SearchRequest
from test_rwkv_pipeline import FakeIndex, FakeModel, hit, native_result, settings
from test_task_matrix import MatrixModel, settings as matrix_settings


class TracedFake(FakeModel):
    async def complete(self, messages, *, trace=None, **kwargs):
        if kwargs.pop('check_only', False):
            self.calls.append({'stage': kwargs['stage'], 'check_only': True})
            return native_result(kwargs['stage'], None)
        result = await super().complete(messages, **kwargs)
        if trace is not None:
            trace.update(result.trace)
        return result


def material():
    return source_from_hit(hit('s', '资料原文😀'))


@pytest.mark.asyncio
async def test_budget_admission_is_atomic_across_parallel_readers_and_stops_writer():
    model = TracedFake()
    pipeline = RWKVPipeline(settings(native_request_max_calls=3, native_empty_evidence_policy='fail'),
        FakeIndex({'alpha query': [hit(str(i), '资料') for i in range(10)], 'beta query': []}), model)
    response = await pipeline.ask(SearchRequest(question='q'))
    assert len(model.calls) == 3
    assert [c['stage'] for c in model.calls] == ['planner', 'resolver', 'resolver']
    assert response.generation['status'] == 'call_budget_exceeded'
    assert response.generation['request_budget']['admitted_calls'] == 3
    assert response.generation['request_budget']['rejected_calls'] >= 8
    assert response.answer == ''
    denied = [c for c in response.generation['model_calls'] if c.get('request_call_admitted') is False]
    assert denied and all(c['completion_attempted'] is False and c['raw_text'] is None for c in denied)
    assert pipeline.model is model and not hasattr(pipeline, '_request_budget')


@pytest.mark.asyncio
async def test_exact_limit_completes_and_concurrent_requests_do_not_share_counters():
    model = TracedFake()
    pipe = RWKVPipeline(settings(native_request_max_calls=3),
        FakeIndex({'alpha query': [hit('s', '资料')], 'beta query': []}), model)
    responses = await asyncio.gather(*(pipe.ask(SearchRequest(question=str(i))) for i in range(4)))
    assert len(model.calls) == 12
    for response in responses:
        assert response.generation['status'] == 'completed'
        assert response.generation['request_budget']['admitted_calls'] == 3
        assert response.generation['request_budget']['rejected_calls'] == 0
        assert [e['request_call_number'] for e in response.generation['model_calls']] == [1, 2, 3]
    ids = [{e['call_id'] for e in r.generation['model_calls']} for r in responses]
    assert len(set.union(*ids)) == 12


@pytest.mark.asyncio
async def test_tokenizer_probes_share_request_budget_with_final_writer():
    model = TracedFake()
    pipe = RWKVPipeline(settings(native_request_max_calls=1, native_writer_budget_policy='whole_sources'),
        None, model)
    response = await pipe.ask_materials('q', [material()])
    assert model.calls == [{'stage': 'writer_budget', 'check_only': True}]
    assert response.generation['status'] == 'call_budget_exceeded'
    assert response.generation['request_budget']['admitted_calls'] == 1
    assert response.answer == ''


@pytest.mark.asyncio
async def test_router_is_counted_even_when_it_uses_a_separate_model_client():
    model = TracedFake()
    pipe = RWKVPipeline(settings(native_request_max_calls=1), FakeIndex({'q': []}), model)
    calls = []

    async def decide(messages):
        calls.append(messages)
        return {'stage': 'routing', 'status': 'completed', 'needs_search': False}

    pipe.web.decide = decide
    response = await pipe.ask(SearchRequest(question='q', retrieval_mode='auto'))
    assert len(calls) == 1 and not model.calls
    assert response.generation['status'] == 'call_budget_exceeded'
    assert response.generation['request_budget']['admitted_calls'] == 1


@pytest.mark.asyncio
async def test_deadline_includes_waiting_for_shared_native_transport_slot():
    http_calls = []

    def handle(request):
        http_calls.append(request.url.path)
        if request.url.path == '/tokenize':
            return httpx.Response(200, json={'count': 1, 'tokens': [0], 'max_model_len': 16384})
        return httpx.Response(200, json={'choices': [{'text': '>done</think>原文[资料 1]',
            'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 1}})

    async with NativeRWKVClient(base_url='http://test/v1', model='test', max_concurrency=1,
                                transport=httpx.MockTransport(handle)) as client:
        await client._semaphore.acquire()
        pipe = RWKVPipeline(settings(native_request_timeout_seconds=.03), None, client)
        try:
            response = await asyncio.wait_for(pipe.ask_materials('q', [material()]), 1)
        finally:
            client._semaphore.release()
        assert response.generation['status'] == 'request_timeout'
        assert http_calls == []
        event = response.generation['model_calls'][0]
        assert event['status'] == 'cancelled' and event['completion_attempted'] is False
        assert event['evidence_ids'] == ['s'] and event['prompt_sha256']
        assert event['cancellation_requested'] is True
        assert event['provider_execution_cancelled'] is None
        assert response.generation['request_budget']['elapsed_ms'] >= 20
        # A timed-out waiter neither closes the shared client nor leaks its slot.
        again = await pipe.ask_materials('q', [material()])
        assert again.generation['status'] == 'completed'
        assert http_calls == ['/tokenize', '/v1/completions']


@pytest.mark.asyncio
async def test_deadline_covers_retrieval_and_keeps_completed_planner_trace():
    model = TracedFake()
    pipe = RWKVPipeline(settings(native_request_timeout_seconds=.03), None, model)

    async def retrieve(*args):
        await asyncio.sleep(10)

    pipe.retrieve_groups = retrieve
    response = await asyncio.wait_for(pipe.ask(SearchRequest(question='q')), 1)
    assert response.generation['status'] == 'request_timeout'
    assert len(model.calls) == 1
    assert response.generation['model_calls'][0]['raw_text'].startswith('>规划思考')
    assert response.generation['stage_status']['writer'] == 'not_called'
    assert response.answer == '' and response.generation['raw_model_answer'] is None
    assert search_failure_category(response.model_dump()) == 'retrieval_failed'


@pytest.mark.asyncio
async def test_partial_writer_output_and_input_sources_survive_request_deadline():
    raw = '>思考</think>未完成的原文😀'

    class PartialModel:
        async def complete(self, messages, *, trace, **kwargs):
            trace.update(raw_text=raw, completion_attempted=True, prompt=messages[0]['content'])
            await asyncio.sleep(10)

    pipe = RWKVPipeline(settings(native_request_timeout_seconds=.03), None, PartialModel())
    response = await pipe.ask_materials('q', [material()])
    assert response.generation['status'] == 'request_timeout'
    assert response.answer == response.generation['raw_model_answer'] == raw
    assert response.generation['answer_modified'] is False
    assert [s.id for s in response.sources] == ['s']
    assert response.generation['citation_map'] == {'1': 's'}


@pytest.mark.asyncio
async def test_later_writer_timeout_keeps_earlier_raw_output_bound_to_its_own_sources():
    raw = '>done</think>第一份原始答案[资料 1]'

    class TwoPass(RWKVPipeline):
        async def _write(self, task, sources, fields):
            await self._call('first', stage='writer', max_tokens=32, sources=sources[:1])
            return await self._call('second', stage='writer', max_tokens=32, sources=sources[1:])

    class RepairModel:
        async def complete(self, messages, *, trace, **kwargs):
            if messages[0]['content'] == 'first':
                return native_result('writer', raw)
            await asyncio.sleep(10)

    sources = [material(), source_from_hit(hit('different', '第二次调用的材料'))]
    pipe = TwoPass(settings(native_request_timeout_seconds=.03), None, RepairModel())
    response = await pipe.ask_materials('q', sources)
    assert response.generation['status'] == 'request_timeout'
    assert response.answer == raw and [s.id for s in response.sources] == ['s']
    assert response.generation['citation_map'] == {'1': 's'}
    calls = response.generation['model_calls']
    assert response.generation['answer_call_id'] == calls[0]['call_id']
    assert calls[1]['status'] == 'cancelled' and calls[1]['evidence_ids'] == ['different']


@pytest.mark.asyncio
async def test_matrix_review_budget_cannot_hide_preexisting_writer_output():
    model = MatrixModel()
    pipe = RWKVPipeline(matrix_settings(native_request_max_calls=6),
        FakeIndex({'A capacity': [hit('a', 'A 8L')], 'B capacity': [hit('b', 'B 12L')]}), model)
    response = await pipe.ask(SearchRequest(question='compare'))
    assert len(model.calls) == 6
    assert response.generation['status'] == 'call_budget_exceeded'
    assert response.answer == '>思考</think>完整模型输出[资料 99]'
    assert response.generation['answer_modified'] is False
    assert len(response.sources) == 2


@pytest.mark.asyncio
async def test_router_timeout_is_not_misclassified_as_generation_failure():
    model = TracedFake()
    pipe = RWKVPipeline(settings(native_request_timeout_seconds=.03), None, model)

    async def decide(messages):
        await asyncio.sleep(10)

    pipe.web.decide = decide
    response = await pipe.ask(SearchRequest(question='q', retrieval_mode='auto'))
    assert response.generation['status'] == 'request_timeout'
    assert not model.calls and response.generation['model_calls'][0]['stage'] == 'routing'
    assert search_failure_category(response.model_dump()) == 'retrieval_failed'


@pytest.mark.asyncio
async def test_external_cancellation_propagates_without_fabricating_a_response():
    entered = asyncio.Event()

    class WaitingModel:
        async def complete(self, messages, *, trace, **kwargs):
            entered.set()
            await asyncio.sleep(10)

    pipe = RWKVPipeline(settings(native_request_timeout_seconds=30), None, WaitingModel())
    task = asyncio.create_task(pipe.ask_materials('q', [material()]))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize('value', [0, -1, float('inf'), float('nan')])
def test_invalid_request_deadlines_are_rejected(value):
    with pytest.raises(ValueError):
        Settings(_env_file=None, native_request_timeout_seconds=value)


def test_request_budgets_are_opt_in():
    config = Settings(_env_file=None)
    assert config.native_request_max_calls is None and config.native_request_timeout_seconds is None
