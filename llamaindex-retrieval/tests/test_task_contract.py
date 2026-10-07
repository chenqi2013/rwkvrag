"""Task authority and wire contracts with fake HTTP, not G1K semantic acceptance."""
import asyncio
from dataclasses import FrozenInstanceError
import json

import httpx
import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.model_client import model_client_options
from llamaindex_retrieval.native_rwkv import NativeRWKVClient
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline, conversation, source_from_hit
from llamaindex_retrieval.schemas import ConversationMessage, SearchRequest
from llamaindex_retrieval.task_contract import TaskAnchor, digest
from test_native_state_routing import Engine
from test_rwkv_pipeline import FakeIndex, hit

MODEL = 'test-model'
QUESTION = '  查指定修订版，保留范围与单位😀\n'
LOOKUPS = ['仅检索线索甲，不是用户要求', '仅检索线索乙，不是用户要求']
EVIDENCE = 'SELECTED 原文：版本7的参数为8；未记载单位换算。'
UNSELECTED = 'UNSELECTED 不能泄漏给Writer'
RAW = '  保留 Writer 原始回答，包括错误引用[资料 99]\n'


def config(**updates):
    values = dict(_env_file=None, rag_pipeline='rwkv', native_model=MODEL,
                  native_completion_protocol='g1j_plain', native_require_model_identity=True,
                  native_planner_prefill='<think></think', native_resolver_prefill='<think></think',
                  native_writer_prefill='<think></think', native_history_protocol='current-question-v2',
                  native_plan_protocol='fact_queries_v1', native_task_source='queries',
                  native_resolver_protocol='binary_query', native_resolver_task_grouping='individual',
                  native_resolver_budget_scope='per_query', native_resolver_sources=2,
                  native_writer_prompt_protocol='evidence_first', native_task_contract='anchored_v1',
                  native_empty_evidence_policy='fail', native_answer_quality_policy='fail_citation',
                  native_preserve_original_question=False)
    return Settings(**(values | updates))


class TaskEngine(Engine):
    def __init__(self):
        super().__init__()
        self.rewrite = json.dumps({'question': QUESTION}, ensure_ascii=False)
        self.plan = json.dumps({'queries': LOOKUPS}, ensure_ascii=False)
        self.no_evidence = False
        self.too_long = False
        self.writer_hang = False
        self.prompts = []

    async def handle(self, request):
        response = await super().handle(request)
        if request.url.path == '/tokenize' and self.too_long:
            return httpx.Response(200, json={'count': 16384, 'tokens': [1] * 16384, 'max_model_len': 16384})
        if request.url.path != '/v1/completions':
            return response
        payload = json.loads(request.content)
        prompt = payload['prompt']
        if prompt.startswith('User: 只输出JSON：'):
            stage, raw = 'current_question', self.rewrite
        elif prompt.startswith('User: 将用户当前问题'):
            stage, raw = 'planner', self.plan
        elif prompt.startswith('User: 下面是格式示例'):
            stage = 'reader'
            raw = json.dumps({'answer': 'NO' if self.no_evidence or UNSELECTED in prompt else 'YES'})
        else:
            stage, raw = 'writer', RAW
        self.prompts.append((stage, prompt))
        if stage == 'writer' and self.writer_hang:
            await asyncio.sleep(10)
        body = response.json()
        body['choices'][0]['text'] = raw
        return httpx.Response(200, json=body)


def client(engine, settings):
    return NativeRWKVClient(**model_client_options(settings, transport=httpx.MockTransport(engine.handle)))


def index():
    return FakeIndex({q: [hit('selected', EVIDENCE), hit('discarded', UNSELECTED)] for q in LOOKUPS})


def task_from_writer(prompt):
    return json.JSONDecoder().raw_decode(prompt.split('latest_question是最新问题）：', 1)[1])[0]


def anchor_for(question=QUESTION):
    task = conversation(question, [])
    return TaskAnchor.from_resolution(question, task, task, [])


@pytest.mark.parametrize('override', [
    {'rag_pipeline': 'existing'}, {'native_transport': 'rwkvos_batch'},
    {'native_completion_protocol': 'native'}, {'native_require_model_identity': False},
    {'native_task_matrix_enabled': True}, {'native_history_protocol': 'raw'},
    {'native_plan_protocol': 'queries_fields'}, {'native_plan_protocol': 'shared_tasks'},
    {'native_resolver_protocol': 'task_units'}, {'native_writer_pipeline': 'funnel_v1'},
    {'native_writer_pipeline': 'typed_funnel_v10'}, {'native_writer_prompt_protocol': 'task_first'},
    {'native_writer_budget_policy': 'whole_sources'}, {'native_task_contract': 'invented'},
    {'native_empty_evidence_policy': 'write'}, {'native_answer_quality_policy': 'disabled'},
])
def test_incompatible_contract_is_rejected_without_silently_changing_settings(override):
    with pytest.raises(ValueError):
        config(**override)


def test_task_anchor_preserves_bytes_and_is_frozen_but_does_not_claim_semantics():
    anchor = anchor_for()
    assert anchor.question == QUESTION
    assert anchor.trace()['question_sha256'] == digest(QUESTION)
    assert anchor.trace()['task_sha256'] == digest(conversation(QUESTION, []))
    assert anchor.trace()['origin'] == 'user_no_history'
    assert anchor.trace()['rewrite_call_id'] is None
    assert anchor.trace()['semantic_verified'] is False
    with pytest.raises(FrozenInstanceError):
        anchor.question = 'changed'
    trace = anchor.trace()
    trace['origin'] = 'tampered'
    assert anchor.trace()['origin'] == 'user_no_history'


@pytest.mark.parametrize('bad', ['raw_history', 'different_task', 'missing_call', 'uncompleted', 'wrong_question', 'blank'])
def test_anchor_cannot_drop_history_or_bind_to_uncompleted_different_output(bad):
    question = QUESTION
    task = conversation(question, [])
    original = conversation(question, [ConversationMessage(role='user', content='早期条件')])
    events = [{'purpose': 'current_question', 'status': 'completed', 'parsed_question': question, 'call_id': 'test-call'}]
    if bad == 'raw_history':
        events = []
    elif bad == 'different_task':
        task = original
    elif bad == 'missing_call':
        events[0].pop('call_id')
    elif bad == 'uncompleted':
        events[0]['status'] = 'length'
    elif bad == 'wrong_question':
        events[0]['parsed_question'] = 'another'
    else:
        question = ' '
    with pytest.raises(ValueError):
        TaskAnchor.from_resolution(question, task, original, events)


def test_explicit_contract_not_model_name_auto_detection_and_no_implicit_plan_state():
    assert Settings(_env_file=None, native_model='rwkv7-g1k-7.2b-20260930-ctx25600').native_task_contract == 'legacy'
    route = {'model': MODEL, 'roles': {'plan': None, 'reader': None, 'writer': None}}
    with pytest.raises(ValueError, match='explicit current_question'):
        config(native_state_routing=route)
    route['roles']['current_question'] = None
    assert config(native_state_routing=route).native_state_routing.roles['current_question'] is None


@pytest.mark.parametrize('scope,expected_readers', [('global', 4), ('per_query', 4)])
async def test_queries_are_lookup_hints_while_task_and_selected_evidence_reach_writer(scope, expected_readers):
    cfg = config(native_resolver_budget_scope=scope)
    engine = TaskEngine()
    sources = index()
    request = SearchRequest(question=QUESTION)
    async with client(engine, cfg) as model:
        response = await RWKVPipeline(cfg, sources, model).ask(request)
    assert response.answer == response.generation['raw_model_answer'] == RAW
    assert response.retrieval['plan'] == {'queries': LOOKUPS, 'fields': LOOKUPS}
    assert response.retrieval['retrieval_queries'] == LOOKUPS
    assert response.retrieval['active_tasks_usage'] == 'lookup_hints_only'
    assert response.retrieval['writer_tasks'] == [request.question]
    calls = response.generation['model_calls']
    assert [c['stage'] for c in calls].count('resolver') == expected_readers
    readers = [p for stage, p in engine.prompts if stage == 'reader']
    for prompt in readers:
        task = json.JSONDecoder().raw_decode(prompt.split('当前任务：', 1)[1])[0]
        assert task == {'history': [], 'latest_question': request.question}
        lookup = json.JSONDecoder().raw_decode(prompt.split('检索提示：', 1)[1])[0]
        assert lookup in [[q] for q in LOOKUPS]
    writer = next(p for stage, p in engine.prompts if stage == 'writer')
    assert task_from_writer(writer) == {'history': [], 'latest_question': request.question}
    assert all(q not in writer for q in LOOKUPS)
    assert EVIDENCE in writer and UNSELECTED not in writer
    assert len(response.sources) == 1 and response.sources[0].snippet == EVIDENCE
    assert response.sources[0].metadata['selection_scope'] == 'binary-task-v1'
    assert response.generation['task_source'] == 'current_question'
    assert response.generation['selection_protocol'] == 'binary-task-v1'
    assert all(c['task_contract'] == response.generation['task_contract'] for c in calls)
    assert response.generation['citation_audit']['unknown_label_ids'] == [99]
    assert response.generation['status'] == 'answer_quality_failed'  # bad label exposed, never repaired


async def test_history_is_model_authored_once_not_programmatically_resolved_or_repeated_downstream():
    cfg = config(native_request_max_calls=8)
    engine = TaskEngine()
    history = [ConversationMessage(role='user', content='早期范围，不由代码合并'),
               ConversationMessage(role='assistant', content='这个建议不能直接当事实')]
    request = SearchRequest(question='LATEST_CORRECTION_UNIQUE', history=history)
    async with client(engine, cfg) as model:
        response = await RWKVPipeline(cfg, index(), model).ask(request)
    calls = response.generation['model_calls']
    anchor = response.generation['task_contract']
    assert anchor['origin'] == 'model_authored' and anchor['rewrite_call_id'] == calls[0]['call_id']
    assert anchor['original_task_sha256'] == digest(conversation(request.question, history))
    assert calls[0]['raw_text'] == engine.rewrite
    assert anchor['semantic_verified'] is False
    assert response.generation['request_budget']['admitted_calls'] == 7
    for stage, prompt in engine.prompts:
        if stage != 'current_question':
            assert all(message.content not in prompt for message in history)
            assert request.question not in prompt
            assert json.dumps(QUESTION, ensure_ascii=False) in prompt


async def test_materials_control_has_identical_wire_prompt_raw_and_decoding_to_legacy():
    all_wires = []
    for contract in ('legacy', 'anchored_v1'):
        cfg = config(native_task_contract=contract)
        engine = TaskEngine()
        async with client(engine, cfg) as model:
            response = await RWKVPipeline(cfg, None, model).ask_materials(QUESTION, [source_from_hit(hit('s', EVIDENCE))])
        all_wires.append(engine.completions)
        assert response.answer == RAW
        assert ('task_contract' in response.generation) == (contract == 'anchored_v1')
    assert all_wires[0] == all_wires[1]


async def test_disabled_full_path_retains_query_tasks_reader_protocol_and_no_anchor():
    cfg = config(native_task_contract='legacy')
    engine = TaskEngine()
    async with client(engine, cfg) as model:
        response = await RWKVPipeline(cfg, index(), model).ask(SearchRequest(question=QUESTION))
    assert 'task_contract' not in response.generation and 'task_contract' not in response.retrieval
    assert all('task_contract' not in c for c in response.generation['model_calls'])
    reader = next(p for stage, p in engine.prompts if stage == 'reader')
    assert '待查问题：' in reader and '当前任务：' not in reader
    writer = next(p for stage, p in engine.prompts if stage == 'writer')
    assert all(q in writer for q in LOOKUPS)
    assert response.generation['selection_protocol'] == 'binary-query-v1'


@pytest.mark.parametrize('history', [False, True])
async def test_empty_materials_never_calls_any_role(history):
    cfg = config()
    engine = TaskEngine()
    async with client(engine, cfg) as model:
        response = await RWKVPipeline(cfg, None, model).ask_materials(QUESTION, [],
            history=[ConversationMessage(role='user', content='early')] if history else [])
    assert response.generation['status'] == 'no_evidence' and not engine.requests


async def test_malformed_rewrite_stops_without_creating_task_or_searching():
    cfg = config()
    engine = TaskEngine()
    engine.rewrite = '{"question":null}'
    sources = index()
    async with client(engine, cfg) as model:
        response = await RWKVPipeline(cfg, sources, model).ask(SearchRequest(question='correction',
            history=[ConversationMessage(role='user', content='old')]))
    assert response.generation['status'] == 'current_question_failed'
    assert len(engine.completions) == 1 and not sources.calls
    assert 'task_contract' not in response.generation


async def test_existing_planner_failure_remains_visible_not_promoted_by_task_binding():
    cfg = config()
    engine = TaskEngine()
    engine.plan = '{"queries":[]}'
    request = SearchRequest(question=QUESTION)
    sources = FakeIndex({request.question: [hit('s', EVIDENCE)]})
    async with client(engine, cfg) as model:
        response = await RWKVPipeline(cfg, sources, model).ask(request)
    assert response.generation['status'] == 'planner_partial_failure'
    assert response.generation['planner_fallback'] == 'original_question'
    assert response.generation['model_calls'][0]['raw_text'] == engine.plan
    assert response.answer == RAW and not response.generation['task_contract']['semantic_verified']


async def test_no_selected_evidence_does_not_call_writer_or_invent_refusal():
    cfg = config()
    engine = TaskEngine()
    engine.no_evidence = True
    async with client(engine, cfg) as model:
        response = await RWKVPipeline(cfg, index(), model).ask(SearchRequest(question=QUESTION))
    assert response.generation['status'] == 'no_evidence' and response.answer == ''
    assert all(stage != 'writer' for stage, _ in engine.prompts)
    assert not response.generation['task_contract']['semantic_verified']


async def test_call_budget_stops_before_reader_http_and_counts_same_population():
    cfg = config(native_request_max_calls=1)
    engine = TaskEngine()
    async with client(engine, cfg) as model:
        response = await RWKVPipeline(cfg, index(), model).ask(SearchRequest(question=QUESTION))
    assert response.generation['status'] == 'call_budget_exceeded'
    assert response.generation['request_budget']['admitted_calls'] == 1
    assert len(engine.completions) == 1
    assert all(c['task_contract']['question_sha256'] == digest(SearchRequest(question=QUESTION).question)
               for c in response.generation['model_calls'])


async def test_complete_input_plus_output_exceeding_context_is_rejected_not_truncated():
    cfg = config()
    engine = TaskEngine()
    engine.too_long = True
    async with client(engine, cfg) as model:
        response = await RWKVPipeline(cfg, None, model).ask_materials(QUESTION, [source_from_hit(hit('s', EVIDENCE))])
    assert response.generation['status'] == 'budget_exceeded'
    assert not engine.completions and len(engine.requests) == 1
    token_payload = json.loads(engine.requests[0].content)
    assert EVIDENCE in token_payload['prompt'] and json.dumps(QUESTION, ensure_ascii=False) in token_payload['prompt']


async def test_parallel_material_requests_do_not_share_or_mutate_task_anchor():
    cfg = config(native_request_max_calls=1)
    engine = TaskEngine()
    async with client(engine, cfg) as model:
        pipe = RWKVPipeline(cfg, None, model)
        replies = await asyncio.gather(*(pipe.ask_materials(f'问题{i}😀', [source_from_hit(hit('s', EVIDENCE))]) for i in range(8)))
    for i, response in enumerate(replies):
        task = f'问题{i}😀'
        anchor = response.generation['task_contract']
        trace = response.generation['model_calls'][0]
        assert anchor['question_sha256'] == digest(task) and trace['task_contract'] == anchor
        assert task_from_writer(trace['prompt'])['latest_question'] == task
        assert response.generation['request_budget']['admitted_calls'] == 1
    assert len(engine.completions) == 8


async def test_timeout_preserves_task_anchor_and_does_not_fabricate_answer():
    cfg = config(native_request_timeout_seconds=.05)
    engine = TaskEngine()
    engine.writer_hang = True
    async with client(engine, cfg) as model:
        response = await RWKVPipeline(cfg, None, model).ask_materials(QUESTION, [source_from_hit(hit('s', EVIDENCE))])
    assert response.generation['status'] == 'request_timeout'
    assert response.answer == ''
    assert response.generation['task_contract'] == response.generation['model_calls'][0]['task_contract']


async def test_internal_reader_writer_cannot_bypass_anchor_or_use_query_as_answer_task():
    cfg = config()
    engine = TaskEngine()
    anchor = anchor_for()
    async with client(engine, cfg) as model:
        pipe = RWKVPipeline(cfg, None, model)
        with pytest.raises(ValueError, match='must agree'):
            await pipe._write(anchor.task, [], [QUESTION])
        with pytest.raises(ValueError, match='must agree'):
            await pipe._resolve(anchor.task, LOOKUPS, [])
        with pytest.raises(ValueError, match='exact current question'):
            await pipe._write(anchor.task, [], LOOKUPS, task_anchor=anchor)
        with pytest.raises(ValueError, match='differs'):
            await pipe._resolve('other task', LOOKUPS, [], task_anchor=anchor)
    assert not engine.requests


@pytest.mark.parametrize('scope', ['global', 'per_query'])
async def test_source_quotas_rank_order_and_lookup_call_population_do_not_change(scope):
    results = []
    populations = []
    for contract in ('legacy', 'anchored_v1'):
        cfg = config(native_task_contract=contract, native_resolver_budget_scope=scope,
                     native_resolver_sources=1, native_candidate_order='query_round_robin')
        engine = TaskEngine()
        sources = FakeIndex({LOOKUPS[0]: [hit('a', EVIDENCE), hit('b', EVIDENCE)],
                             LOOKUPS[1]: [hit('b', EVIDENCE), hit('c', EVIDENCE)]})
        async with client(engine, cfg) as model:
            response = await RWKVPipeline(cfg, sources, model).ask(SearchRequest(question=QUESTION))
        results.append(response.retrieval)
        populations.append([(c['source_id'], c['task_group']) for c in response.generation['model_calls']
                            if c['stage'] == 'resolver'])
    for key in ('resolver_source_ids', 'resolver_source_tasks', 'omitted_by_reader_budget',
                'query_results', 'retrieval_queries', 'candidate_order'):
        assert results[0][key] == results[1][key]
    assert populations[0] == populations[1]
    assert results[1]['resolver_source_ids'] == (['a'] if scope == 'global' else ['a', 'b'])


async def test_anchored_flow_and_independent_task_state_compose_without_role_or_task_leakage():
    route = {'model': MODEL, 'roles': {'current_question': 'task-ref', 'plan': 'plan-ref',
                                     'reader': 'reader-ref', 'writer': 'writer-ref'}}
    cfg = config(native_state_routing=route)
    engine = TaskEngine()
    engine.refs.add('task-ref')
    async with client(engine, cfg) as model:
        response = await RWKVPipeline(cfg, index(), model).ask(SearchRequest(question='correction',
            history=[ConversationMessage(role='user', content='early')]))
    assert response.answer == RAW
    calls = response.generation['model_calls']
    expected = ['current_question', 'plan', 'reader', 'reader', 'reader', 'reader', 'writer']
    assert [c['state_selection']['role'] for c in calls] == expected
    assert all(c['task_contract'] == response.generation['task_contract'] for c in calls[1:])
    assert [p['vllm_xargs']['rwkv_state_read_ref'] for p in engine.completions] == [
        'task-ref', 'plan-ref', 'reader-ref', 'reader-ref', 'reader-ref', 'reader-ref', 'writer-ref']
    assert not any(c['state_selection']['state_consumption_verified'] for c in calls)


@pytest.mark.parametrize('conflict', [False, True])
async def test_retrieval_failure_retains_task_binding_without_reader_or_writer(conflict):
    cfg = config()
    engine = TaskEngine()
    sources = FakeIndex({LOOKUPS[0]: [hit('same-id', 'first')],
                         LOOKUPS[1]: [hit('same-id', 'different')]} if conflict else {})
    async with client(engine, cfg) as model:
        response = await RWKVPipeline(cfg, sources, model).ask(SearchRequest(question=QUESTION))
    assert response.generation['status'] == 'retrieval_failed'
    assert response.answer == '' and len(engine.completions) == 1
    assert response.generation['task_contract'] == response.generation['model_calls'][0]['task_contract']
