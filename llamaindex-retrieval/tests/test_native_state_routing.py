"""Offline State-routing contract tests, not GPU/tensor isolation acceptance."""
import asyncio
import base64
from copy import deepcopy
from hashlib import sha256
import json

import httpx
import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.model_client import model_client_class, model_client_options
from llamaindex_retrieval.native_rwkv import NativeRWKVClient
from llamaindex_retrieval.native_state import NativeStateRouting, PROTOCOL
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline
from llamaindex_retrieval.schemas import SearchRequest, SourceItem
from llamaindex_retrieval.lexical_index import LexicalResult
from llamaindex_retrieval.structured_native import StructuredNativeRWKVClient

MODEL = 'test-model'
ROUTING = {'model': MODEL, 'roles': {'plan': 'plan-ref', 'reader': 'reader-ref', 'writer': 'writer-ref'}}
RAW = '  不修写原答 [资料 99]\n'
MESSAGES = [{'role': 'user', 'content': '完整任务\n逐字证据😀'}]


def settings(**kwargs):
    values = dict(_env_file=None, native_model=MODEL, native_completion_protocol='g1j_plain',
                  native_planner_prefill='<think></think', native_resolver_prefill='<think></think',
                  native_writer_prefill='<think></think', native_state_routing=deepcopy(ROUTING))
    values.update(kwargs)
    return Settings(**values)


class Engine:
    def __init__(self):
        self.requests = []
        self.refs = {'plan-ref', 'reader-ref', 'writer-ref', 'assessment-ref', 'followup-ref', 'review-ref'}
        self.change = lambda path, body: body
        self.fail_at = None
        self.hang_at = None
        self.entered = asyncio.Event()
        self.finish = 'stop'

    async def handle(self, request):
        self.requests.append(request)
        path = request.url.path
        if path == self.hang_at:
            self.entered.set()
            await asyncio.sleep(10)
        if path == self.fail_at:
            return httpx.Response(404, json={'error': 'Unknown RWKV State ref'})
        if path == '/tokenize':
            data = {'count': 3, 'tokens': [5, 6, 7], 'max_model_len': 16384}
        elif path == '/v1/models':
            data = {'data': [{'id': MODEL, 'max_model_len': 16384}]}
        elif path == '/v1/rwkv/state/capabilities':
            data = {'protocol': PROTOCOL, 'action': 'capabilities', 'workers': [{
                'protocol': PROTOCOL, 'supported': True, 'enabled': True, 'import_supported': True,
                'process_local': True, 'durable': False, 'snapshot_nbytes': 100}]}
        elif path.startswith('/v1/rwkv/state/'):
            ref = path.rsplit('/', 1)[1]
            if ref not in self.refs:
                return httpx.Response(404, json={'error': 'Unknown RWKV State ref'})
            data = {'protocol': PROTOCOL, 'action': 'inspect', 'workers': [{
                'protocol': PROTOCOL, 'state_ref': ref, 'nbytes': 100, 'parent_ref': None,
                'processed_token_count': 0, 'pending_tail_token_count': 0,
                'finish_reason': 'initial', 'state_dtype': 'torch.float16',
                'process_local': True, 'durable': False, 'scheduler_slots_reserved': 0}]}
        else:
            assert path == '/v1/completions'
            await asyncio.sleep(0)  # interleave different roles/schemas, never change client defaults
            data = {'model': MODEL, 'choices': [{'text': RAW, 'finish_reason': self.finish,
                     'prompt_token_ids': [5, 6, 7], 'token_ids': [8, 0]}],
                    'usage': {'prompt_tokens': 3, 'completion_tokens': 2}}
        return httpx.Response(200, json=self.change(path, data))

    @property
    def completions(self):
        return [json.loads(r.content) for r in self.requests if r.url.path == '/v1/completions']


def client(engine, **kwargs):
    options = dict(base_url='http://test/v1', model=MODEL, prompt_protocol='g1j_plain',
                   state_routing=deepcopy(ROUTING), transport=httpx.MockTransport(engine.handle))
    options.update(kwargs)
    return NativeRWKVClient(**options)


async def call(model, **kwargs):
    return await model.complete(MESSAGES, assistant_prefill='<think></think', **kwargs)


@pytest.mark.parametrize('stage,role', [('planner', 'plan'), ('resolver', 'reader'), ('reader', 'reader'), ('writer', 'writer')])
async def test_role_ref_on_wire_and_metadata_checks_preserve_raw(stage, role):
    engine = Engine()
    async with client(engine, api_key='not-in-trace') as model:
        result = await call(model, stage=stage, evidence_ids=['chosen-only'])
    assert result.status == 'completed' and result.raw_text == RAW
    assert engine.completions[0]['vllm_xargs'] == {'rwkv_state_read_ref': role+'-ref'}
    assert [r.method for r in engine.requests] == ['POST', 'GET', 'GET', 'GET', 'POST']
    selected = result.trace['state_selection']
    assert selected['role'] == role and selected['metadata_checked'] is True
    assert selected['state_consumption_verified'] is selected['tensor_compatibility_verified'] is False
    assert result.trace['evidence_ids'] == ['chosen-only']
    assert 'not-in-trace' not in json.dumps(result.trace)
    assert result.trace['envelope']['raw_text_modified'] is False
    for entry, request in zip(result.trace['http'], engine.requests, strict=True):
        body = base64.b64decode(entry['request_body_base64'])
        assert body == request.content and sha256(body).hexdigest() == entry['request_body_sha256']
        response = base64.b64decode(entry['response_body_base64'])
        assert sha256(response).hexdigest() == entry['response_body_sha256']


async def test_explicit_null_is_not_role_fallback_and_disabled_path_unchanged():
    engine = Engine()
    routing = deepcopy(ROUTING)
    routing['roles']['reader'] = None
    async with client(engine, state_routing=routing) as model:
        base = await call(model, stage='resolver')
        written = await call(model, stage='writer')
    assert base.trace['state_selection']['mode'] == 'base_no_ref'
    assert 'vllm_xargs' not in engine.completions[0]
    assert engine.completions[1]['vllm_xargs']['rwkv_state_read_ref'] == 'writer-ref'
    assert written.status == 'completed'
    legacy = Engine()
    async with client(legacy, state_routing=None) as model:
        disabled = await call(model, stage='resolver')
    assert disabled.status == 'completed' and 'state_selection' not in disabled.trace
    assert len(legacy.requests) == 2 and legacy.completions[0] == engine.completions[0]


@pytest.mark.parametrize('ref', ['', ' ', ' x', 'x\n', '../escape', 'a/b', 'a?b', 'a#b', 'x'*129, 12, True])
def test_bad_ref_not_coerced_or_silently_ignored(ref):
    route = deepcopy(ROUTING)
    route['roles']['writer'] = ref
    with pytest.raises(ValueError):
        NativeStateRouting.model_validate(route)


@pytest.mark.parametrize('route', [
    {'model': MODEL, 'roles': {'writer': 'writer-ref'}},
    {'model': MODEL, 'roles': {**ROUTING['roles'], 'unknown': None}},
    {**ROUTING, 'model': ' '+MODEL}, {**ROUTING, 'model': 123}, {**ROUTING, 'default_ref': 'writer-ref'},
])
def test_explicit_configuration_no_unknown_roles_or_defaults(route):
    with pytest.raises(ValueError):
        NativeStateRouting.model_validate(route)


@pytest.mark.parametrize('options', [{'native_model': 'other'}, {'native_transport': 'rwkvos_batch'},
    {'native_completion_protocol': 'native'}, {'rag_pipeline': 'existing'}, {'rwkvos_state_id': 'old'}])
def test_settings_reject_incompatible_route(options):
    with pytest.raises(ValueError):
        settings(**options)


async def test_missing_or_invalid_role_and_disabled_override_never_reach_http():
    engine = Engine()
    async with client(engine) as model:
        for stage, role in [('planner', 'followup'), ('writer', 'reader'), ('other', None), ('writer_budget', None)]:
            result = await call(model, stage=stage, state_role=role)
            assert result.status == 'invalid_request' and not result.trace['completion_attempted']
    async with client(engine, state_routing=None) as model:
        result = await call(model, stage='writer', state_role='writer')
        assert result.status == 'invalid_request'
    assert engine.requests == []


async def test_expired_ref_no_completion_reload_delete_or_fallback():
    engine = Engine()
    async with client(engine) as model:
        first = await call(model, stage='writer')
        engine.refs.remove('writer-ref')
        second = await call(model, stage='writer')
    assert first.status == 'completed' and second.status == 'http_error'
    assert len(engine.completions) == 1
    assert not second.trace['completion_attempted'] and second.raw_text is None
    assert not second.trace['state_selection']['metadata_checked']
    assert all(r.method in {'GET', 'POST'} and not r.url.path.endswith(('upload', 'clone')) for r in engine.requests)


@pytest.mark.parametrize('kind', ['model', 'model_missing', 'context', 'protocol', 'workers', 'import',
    'wrong_ref', 'size', 'processed', 'pending', 'boolean_count', 'continuation', 'durable'])
async def test_bad_live_metadata_blocks_generation(kind):
    engine = Engine()
    def changed(path, body):
        if path == '/v1/models':
            if kind == 'model':
                body['data'][0]['id'] = 'other'
            if kind == 'model_missing':
                body['data'] = []
            if kind == 'context':
                body['data'][0]['max_model_len'] = 8192
        if path.endswith('/capabilities'):
            if kind == 'protocol':
                body['protocol'] = 'unknown'
            if kind == 'workers':
                body['workers'] *= 2
            if kind == 'import':
                body['workers'][0]['import_supported'] = False
        if path.endswith('/writer-ref'):
            worker = body['workers'][0]
            if kind == 'wrong_ref':
                worker['state_ref'] = 'reader-ref'
            if kind == 'size':
                worker['nbytes'] += 1
            if kind == 'processed':
                worker['processed_token_count'] = 1
            if kind == 'pending':
                worker['pending_tail_token_count'] = 1
            if kind == 'boolean_count':
                worker['processed_token_count'] = False
            if kind == 'continuation':
                worker['finish_reason'] = 'stop'
            if kind == 'durable':
                worker['durable'] = True
        return body
    engine.change = changed
    async with client(engine) as model:
        result = await call(model, stage='writer')
    assert result.status == 'invalid_response' and result.raw_text is None
    assert not engine.completions and not result.trace['completion_attempted']


async def test_deletion_between_inspection_and_generation_is_not_retried():
    engine = Engine()
    engine.fail_at = '/v1/completions'
    async with client(engine) as model:
        result = await call(model, stage='writer')
    assert result.status == 'http_error' and len(engine.completions) == 1
    assert result.trace['state_selection']['metadata_checked']  # existence was only a point-in-time observation
    assert result.trace['state_selection']['state_consumption_verified'] is False


async def test_completion_identity_and_length_preserve_original_answer():
    for finish, wrong_model in [('length', False), ('stop', True)]:
        engine = Engine()
        engine.finish = finish
        if wrong_model:
            def changed(path, body):
                if path == '/v1/completions':
                    body['model'] = 'other'
                return body
            engine.change = changed
        async with client(engine, require_model_identity=False) as model:
            result = await call(model, stage='writer')
        assert result.raw_text == RAW and len(engine.completions) == 1
        assert result.status == ('invalid_response' if wrong_model else 'length')


async def test_budget_only_counts_without_state_readiness_claim():
    engine = Engine()
    engine.refs.clear()
    async with client(engine) as model:
        result = await call(model, stage='writer_budget', check_only=True)
    assert result.status == 'completed' and result.trace['budget_only']
    assert result.raw_text is None and not result.trace['completion_attempted']
    assert result.trace['state_selection']['role'] == 'writer'
    assert not result.trace['state_selection']['metadata_checked']
    assert len(engine.requests) == 1 and not engine.completions


async def test_timeout_and_cancellation_keep_metadata_receipts_no_completion():
    for cancel in (False, True):
        engine = Engine()
        engine.hang_at = '/v1/rwkv/state/writer-ref'
        async with client(engine, timeout_seconds=.05 if not cancel else 5) as model:
            trace = {}
            task = asyncio.create_task(call(model, stage='writer', trace=trace))
            await engine.entered.wait()
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert trace['status'] == 'cancelled'
            else:
                assert (await task).status == 'timeout'
        assert not engine.completions and not trace['completion_attempted']
        assert trace['http'][-1]['method'] == 'GET' and 'error_type' in trace['http'][-1]


async def test_concurrent_roles_and_schemas_use_constructor_copy_and_no_sticky_state():
    engine = Engine()
    route = deepcopy(ROUTING)
    route['roles']['plan'] = None
    options = dict(base_url='http://test/v1', model=MODEL, prompt_protocol='g1j_plain', state_routing=route,
                   transport=httpx.MockTransport(engine.handle))
    async with StructuredNativeRWKVClient(**options) as model:
        route['roles']['writer'] = 'reader-ref'
        schemas = [{'type': 'object'}, {'type': 'array'}, None]
        stages = ['writer', 'resolver', 'planner']
        results = await asyncio.gather(*(call(model, stage=stages[i % 3], structured_schema=schemas[i % 3]) for i in range(12)))
    assert all(r.status == 'completed' for r in results)
    assert len(engine.completions) == 12
    for i, result in enumerate(results):
        wire = json.loads(base64.b64decode(result.trace['http'][-1]['request_body_base64']))
        wanted = ['writer-ref', 'reader-ref', None][i % 3]
        assert wire.get('vllm_xargs', {}).get('rwkv_state_read_ref') == wanted
        assert wire.get('structured_outputs') == ({'json': schemas[i % 3]} if schemas[i % 3] else None)
        assert result.trace['parameters'].get('vllm_xargs') == wire.get('vllm_xargs')
        assert result.raw_text == RAW


async def test_model_factory_and_pipeline_forward_matrix_roles_and_selected_only_writer():
    routing = deepcopy(ROUTING)
    routing['roles'].update({r: r+'-ref' for r in ('assessment', 'followup', 'review')})
    config = settings(native_state_routing=routing, native_task_matrix_enabled=True,
        native_resolver_protocol='binary_query', native_resolver_task_grouping='individual', native_task_source='queries')
    assert model_client_class(config) is NativeRWKVClient
    engine = Engine()
    async with NativeRWKVClient(**model_client_options(config, transport=httpx.MockTransport(engine.handle))) as model:
        pipeline = RWKVPipeline(config, object(), model=model)
        selected = SourceItem(id='chosen', document_id='doc', source='kb', title='fixture', score=1,
                              snippet='只选这个原文😀')
        for role in ('plan', 'reader', 'assessment', 'followup', 'review', 'writer'):
            stage = 'writer' if role == 'writer' else 'resolver' if role in {'reader', 'review'} else 'planner'
            result = await pipeline._call('完整任务', stage=stage, state_role=role, max_tokens=32, sources=[selected])
            assert result.status == 'completed' and result.trace['state_selection']['role'] == role
        result = await pipeline._write('只按所选原文回答', [selected], [])
    assert result.raw_text == RAW and result.trace['evidence_ids'] == ['chosen']
    assert '只选这个原文😀' in engine.completions[-1]['prompt']
    assert [r['vllm_xargs']['rwkv_state_read_ref'] for r in engine.completions[:6]] == [r+'-ref' for r in ('plan', 'reader', 'assessment', 'followup', 'review', 'writer')]


@pytest.mark.parametrize('body', [b'[]', b'not JSON', b'{"data":[],"data":[]}',
    b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e400}', b'\xff'])
async def test_ambiguous_or_malformed_metadata_never_generates(body):
    engine = Engine()
    async def handle(request):
        if request.url.path == '/v1/models':
            engine.requests.append(request)
            return httpx.Response(200, content=body)
        return await engine.handle(request)
    async with client(engine, transport=httpx.MockTransport(handle)) as model:
        result = await call(model, stage='writer')
    assert result.status == 'invalid_response' and not engine.completions
    assert base64.b64decode(result.trace['http'][-1]['response_body_base64']) == body


async def test_public_pipeline_real_transport_keeps_selected_only_and_original_answer():
    engine = Engine()
    config = settings(native_preserve_original_question=False, native_resolver_protocol='fields',
        native_writer_pipeline='single', native_writer_prompt_protocol='task_first',
        native_task_source='fields', native_plan_protocol='queries_fields',
        native_resolver_sources=2, native_resolver_format_repair=False)
    class Index:
        def search_chunks(self, query, **kwargs):
            return [LexicalResult(node_id='chosen', document_id='doc', text='SELECTED_ORIGINAL😀',
                                  metadata={'title': 'chosen'}, score=1),
                    LexicalResult(node_id='other', document_id='other', text='UNSELECTED_SOURCE_ONLY',
                                  metadata={'title': 'other'}, score=.5)]
    async def handle(request):
        response = await engine.handle(request)
        if request.url.path == '/v1/completions':
            body = response.json()
            payload = json.loads(request.content)
            ref = payload['vllm_xargs']['rwkv_state_read_ref']
            if ref == 'plan-ref':
                raw = '{"queries":["fixture"],"fields":["field"]}'
            elif ref == 'reader-ref':
                raw = 'f1: E1' if 'SELECTED_ORIGINAL😀' in payload['prompt'] else 'f1: NONE'
            else:
                raw = RAW
            body['choices'][0]['text'] = raw
            return httpx.Response(200, json=body)
        return response
    async with NativeRWKVClient(**model_client_options(config, transport=httpx.MockTransport(handle))) as model:
        response = await RWKVPipeline(config, Index(), model).ask(SearchRequest(question='fixture'))
    assert response.answer == response.generation['raw_model_answer'] == RAW
    assert len(response.sources) == 1 and response.sources[0].snippet == 'SELECTED_ORIGINAL😀'
    writers = [p for p in engine.completions if p['vllm_xargs']['rwkv_state_read_ref'] == 'writer-ref']
    assert len(writers) == 1
    assert 'SELECTED_ORIGINAL😀' in writers[0]['prompt'] and 'UNSELECTED_SOURCE_ONLY' not in writers[0]['prompt']


def test_mutated_pydantic_binding_is_revalidated_before_client_creation():
    route = NativeStateRouting.model_validate(deepcopy(ROUTING))
    route.roles['writer'] = '../escape'
    engine = Engine()
    with pytest.raises(ValueError):
        client(engine, state_routing=route)
    assert not engine.requests


def test_json_environment_setting_reaches_native_factory(monkeypatch):
    reference = settings(native_state_routing=None).model_dump()
    reference.pop('native_state_routing')
    monkeypatch.setenv('RWKVRAG_NATIVE_STATE_ROUTING', json.dumps(ROUTING))
    config = Settings(_env_file=None, **reference)
    assert model_client_options(config)['state_routing'] == ROUTING


async def test_metadata_http_uses_recorder_auth_and_shared_concurrency_limit():
    engine = Engine()
    events = []
    async with client(engine, max_concurrency=1, api_key='private',
                      recorder=lambda event, entry: events.append((event, entry))) as model:
        results = await asyncio.gather(*(call(model, stage=s) for s in ('planner', 'resolver', 'writer')))
    assert all(result.status == 'completed' for result in results)
    assert [r.url.path for r in engine.requests][::5] == ['/tokenize'] * 3
    assert all(r.headers['authorization'] == 'Bearer private' for r in engine.requests)
    finished = [data for event, data in events if event == 'native_http_finished']
    assert len(finished) == 15 and sum(row['method'] == 'GET' for row in finished) == 9
    assert all(row['response_body_complete'] for row in finished)
    assert 'private' not in json.dumps(events)


async def test_route_rejects_bad_model_or_protocol_before_allocating_http_client():
    engine = Engine()
    for kwargs in ({'model': 'wrong'}, {'prompt_protocol': 'native'}):
        with pytest.raises(ValueError, match='matching model'):
            client(engine, **kwargs)
    assert not engine.requests


def test_native_matrix_requires_explicit_auxiliary_bindings():
    with pytest.raises(ValueError, match='all six'):
        settings(native_task_matrix_enabled=True, native_resolver_protocol='binary_query',
                 native_resolver_task_grouping='individual', native_task_source='queries')
