"""Offline role/wire contracts only. Fixtures are not adaptation data."""
import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from llamaindex_retrieval.current_question import (
    correction_prompt, full_history_prompt, g1k_correction_prompt, resolve_current_question,
)
from llamaindex_retrieval.model_client import model_client_options
from llamaindex_retrieval.native_rwkv import NativeRWKVClient
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline, conversation
from llamaindex_retrieval.schemas import ConversationMessage, SearchRequest, SourceItem
from test_native_state_routing import Engine, ROUTING, call, client, settings
from test_rwkv_pipeline import FakeIndex, hit

HISTORY = [ConversationMessage(role='user', content='保留早期条件'),
           ConversationMessage(role='assistant', content='这里不是证据😀')]
QUESTION = '  最新任务\n'
CORRECTED = ' 独立任务\n'
RAW_TASK = json.dumps({'question': CORRECTED}, ensure_ascii=False)
RAW_ANSWER = '  原始答案不改写[资料 1]\n'
MATERIAL = SourceItem(id='selected', document_id='doc', source='kb', title='fixture',
                      score=1, snippet='SELECTED_LITERAL😀')


def config(ref='task-ref', **overrides):
    routing = deepcopy(ROUTING)
    routing['roles']['current_question'] = ref
    return settings(native_state_routing=routing, native_history_protocol='current-question-v2', **overrides)


class TaskEngine(Engine):
    def __init__(self):
        super().__init__()
        self.refs.add('task-ref')
        self.task_raw = RAW_TASK

    async def handle(self, request):
        response = await super().handle(request)
        if request.url.path != '/v1/completions' or response.status_code != 200:
            return response
        payload = json.loads(request.content)  # request-local, including under concurrent calls
        prompt = payload['prompt']
        ref = payload.get('vllm_xargs', {}).get('rwkv_state_read_ref')
        if prompt.startswith(('User: 只输出JSON：', 'User: 更新检索问题')):
            raw = self.task_raw
        elif ref == 'plan-ref':
            raw = '{"queries":["fixture"],"fields":["field"]}'
        elif ref == 'reader-ref':
            raw = 'f1: E1' if MATERIAL.snippet in prompt else 'f1: NONE'
        else:
            raw = RAW_ANSWER
        body = response.json()
        body['choices'][0]['text'] = raw
        return httpx.Response(200, json=body)


def engine_for_tasks():
    return TaskEngine()


def make_client(engine, cfg):
    return NativeRWKVClient(**model_client_options(cfg, transport=httpx.MockTransport(engine.handle)))


def refs(engine):
    return [p.get('vllm_xargs', {}).get('rwkv_state_read_ref') for p in engine.completions]


@pytest.mark.parametrize('protocol,prompt', [
    ('current-question-v1', correction_prompt),
    ('current-question-g1k-v1', g1k_correction_prompt),
    ('current-question-v2', full_history_prompt),
])
@pytest.mark.parametrize('ref', ['task-ref', None])
async def test_opt_in_only_changes_role_not_prompt_decoding_or_result(protocol, prompt, ref):
    cfg = config()
    cfg.native_history_protocol = protocol
    cfg.native_state_routing.roles['current_question'] = ref
    engine = engine_for_tasks()
    legacy = engine_for_tasks()
    async with make_client(engine, cfg) as model:
        result = await resolve_current_question(RWKVPipeline(cfg, None, model), QUESTION, HISTORY)
    async with make_client(legacy, settings(native_history_protocol=protocol)) as model:
        old = await resolve_current_question(
            RWKVPipeline(settings(native_history_protocol=protocol), None, model), QUESTION, HISTORY)
    assert result[:2] == old[:2] == (CORRECTED, conversation(CORRECTED, []))
    trace = result[2][0]
    assert trace['raw_text'] == old[2][0]['raw_text'] == RAW_TASK
    assert trace['purpose'] == 'current_question' and trace['original_task'] == conversation(QUESTION, HISTORY)
    selection = trace['state_selection']
    assert selection['role'] == 'current_question' and selection['state_ref'] == ref
    assert selection['metadata_checked'] and not selection['state_consumption_verified']
    assert not selection['tensor_compatibility_verified']
    actual, baseline = deepcopy(engine.completions[0]), deepcopy(legacy.completions[0])
    actual.pop('vllm_xargs', None)
    baseline.pop('vllm_xargs')
    assert actual == baseline
    assert actual['prompt'] == 'User: ' + prompt(QUESTION, HISTORY) + '\n\nAssistant: <think></think>\n'
    assert actual['max_tokens'] == 512
    assert refs(engine) == [ref] and refs(legacy) == ['plan-ref']
    assert not any(r.url.path.endswith('/plan-ref') for r in engine.requests)


@pytest.mark.parametrize('raw', [False, True])
async def test_no_history_or_raw_does_not_create_a_task_role_call(raw):
    cfg = settings(native_history_protocol='raw') if raw else config()
    engine = engine_for_tasks()
    history = HISTORY if raw else []
    async with make_client(engine, cfg) as model:
        pipe = RWKVPipeline(cfg, None, model)
        current, task, events = await resolve_current_question(pipe, QUESTION, history)
        assert current == QUESTION and task == conversation(QUESTION, history) and events == []
        assert engine.requests == []
        await pipe._call('retrieval plan', stage='planner', max_tokens=64)
    assert refs(engine) == ['plan-ref']


def test_raw_configuration_cannot_silently_ignore_explicit_task_binding():
    for ref in (None, 'task-ref'):
        routing = deepcopy(ROUTING)
        routing['roles']['current_question'] = ref
        with pytest.raises(ValueError, match='enabled history protocol'):
            settings(native_state_routing=routing)


@pytest.mark.parametrize('overrides', [
    {'native_transport': 'rwkvos_batch'}, {'native_task_matrix_enabled': True},
    {'native_completion_protocol': 'native'}, {'native_model': 'other'},
])
def test_task_role_does_not_bypass_existing_transport_or_matrix_restrictions(overrides):
    with pytest.raises(ValueError):
        config(**overrides)


@pytest.mark.parametrize('stage', ['reader', 'resolver', 'writer', 'writer_budget'])
async def test_task_role_cannot_be_used_for_other_stages(stage):
    engine = engine_for_tasks()
    async with make_client(engine, config()) as model:
        result = await call(model, stage=stage, state_role='current_question', check_only=stage == 'writer_budget')
    assert result.status == 'invalid_request' and not engine.requests


async def test_missing_explicit_binding_never_falls_back_when_task_role_is_requested():
    engine = engine_for_tasks()
    async with client(engine) as model:
        result = await call(model, stage='planner', state_role='current_question')
    assert result.status == 'invalid_request' and not engine.requests


@pytest.mark.parametrize('materials', [False, True])
@pytest.mark.parametrize('failure', ['missing_ref', 'malformed', 'metadata'])
async def test_failed_task_extraction_never_retries_plan_or_runs_downstream(materials, failure):
    engine = engine_for_tasks()
    if failure == 'missing_ref':
        engine.refs.remove('task-ref')
    elif failure == 'malformed':
        engine.task_raw = '{"question":null}'
    else:
        def changed(path, body):
            if path.endswith('/task-ref'):
                body['workers'][0]['processed_token_count'] = 1
            return body
        engine.change = changed
    cfg = config()
    index = FakeIndex({})
    async with make_client(engine, cfg) as model:
        pipe = RWKVPipeline(cfg, index, model)
        if materials:
            response = await pipe.ask_materials(QUESTION, [MATERIAL], history=HISTORY)
        else:
            response = await pipe.ask(SearchRequest(question=QUESTION, history=HISTORY))
    assert response.generation['status'] == 'current_question_failed'
    assert response.answer == '' and not index.calls
    assert refs(engine) == (['task-ref'] if failure == 'malformed' else [])
    event = response.generation['model_calls'][0]
    assert event['state_selection']['role'] == 'current_question'
    if failure == 'malformed':
        assert event['raw_text'] == '{"question":null}'
    assert not any(r.url.path.endswith(('/plan-ref', '/writer-ref', '/reader-ref')) for r in engine.requests)


async def test_public_ask_isolates_task_plan_reader_writer_and_keeps_selected_only():
    engine = engine_for_tasks()
    cfg = config(native_preserve_original_question=False, native_resolver_sources=2)
    index = FakeIndex({'fixture': [hit('selected', MATERIAL.snippet), hit('discard', 'NOT_SELECTED')]})
    async with make_client(engine, cfg) as model:
        response = await RWKVPipeline(cfg, index, model).ask(SearchRequest(question=QUESTION, history=HISTORY))
    assert response.generation['status'] == 'completed'
    assert response.answer == response.generation['raw_model_answer'] == RAW_ANSWER
    assert refs(engine) == ['task-ref', 'plan-ref', 'reader-ref', 'reader-ref', 'writer-ref']
    writer = engine.completions[-1]['prompt']
    assert MATERIAL.snippet in writer and 'NOT_SELECTED' not in writer
    assert HISTORY[1].content not in writer and QUESTION not in writer
    assert json.dumps(CORRECTED, ensure_ascii=False) in writer
    assert len(response.sources) == 1


async def test_parallel_material_requests_and_constructor_snapshot_no_sticky_role():
    engine = engine_for_tasks()
    cfg = config(native_request_max_calls=2)
    async with make_client(engine, cfg) as model:
        pipe = RWKVPipeline(cfg, None, model)
        # Neither refs nor the decision to opt in is read from this mutable dict at call time.
        cfg.native_state_routing.roles.pop('current_question')
        cfg.native_state_routing.roles['writer'] = 'plan-ref'
        answers = await asyncio.gather(*(pipe.ask_materials(
            QUESTION, [MATERIAL], history=HISTORY if i % 2 else []) for i in range(8)))
    for i, response in enumerate(answers):
        assert response.answer == RAW_ANSWER and response.generation['status'] == 'completed'
        expected = ['current_question', 'writer'] if i % 2 else ['writer']
        calls = response.generation['model_calls']
        assert [c['state_selection']['role'] for c in calls] == expected
        assert response.generation['request_budget']['admitted_calls'] == len(expected)
        assert [c['state_selection']['state_ref'] for c in calls] == (
            ['task-ref', 'writer-ref'] if i % 2 else ['writer-ref'])
    assert len(engine.completions) == 12 and 'plan-ref' not in refs(engine)


async def test_task_call_consumes_one_request_slot_and_denied_writer_never_reaches_http():
    engine = engine_for_tasks()
    cfg = config(native_request_max_calls=1)
    async with make_client(engine, cfg) as model:
        response = await RWKVPipeline(cfg, None, model).ask_materials(QUESTION, [MATERIAL], history=HISTORY)
    assert response.generation['status'] == 'call_budget_exceeded'
    assert response.generation['request_budget']['admitted_calls'] == 1
    assert refs(engine) == ['task-ref']
    assert len(engine.requests) == 5  # tokenize + three metadata GETs + completion = ONE logical operation


async def test_routing_disabled_preserves_history_wire_without_state_metadata():
    engine = engine_for_tasks()
    cfg = settings(native_state_routing=None, native_history_protocol='current-question-v2')
    async with make_client(engine, cfg) as model:
        result = await resolve_current_question(RWKVPipeline(cfg, None, model), QUESTION, HISTORY)
    assert result[0] == CORRECTED and refs(engine) == [None]
    assert 'state_selection' not in result[2][0]
    assert [r.url.path for r in engine.requests] == ['/tokenize', '/v1/completions']
