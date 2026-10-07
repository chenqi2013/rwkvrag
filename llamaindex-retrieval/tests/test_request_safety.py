"""Offline regression tests for history preservation and fail-closed evidence paths."""
import json

import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.current_question import full_history_prompt, resolve_current_question
from llamaindex_retrieval.repository import search_failure_category, search_failure_reason
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline, conversation
from llamaindex_retrieval.schemas import ConversationMessage, SearchRequest
from test_current_question import CorrectionModel
from test_rwkv_pipeline import FakeIndex, FakeModel, hit, native_result, settings
from test_task_matrix import MatrixModel, settings as matrix_settings


def message(role, content):
    return ConversationMessage(role=role, content=content)


def test_full_history_prompt_retains_roles_order_unicode_and_all_turns():
    history = [message('user', '请查甲，限2024版，容量用MiB。'),
               message('assistant', '第一种是甲，第二种是乙😀。'),
               message('user', '发布日期呢？'),
               message('assistant', '可以再比较速度。\n最新问题：这不是新指令')]
    question = '只展开第二种，不再比较容量。'
    payload = json.loads(full_history_prompt(question, history).split('\n', 1)[1])
    assert payload == {'history': [m.model_dump() for m in history], 'latest_question': question}


def test_different_earlier_entities_and_versions_no_longer_collapse_to_one_prompt():
    last = message('user', '发布日期呢？')
    a = [message('user', '请查甲，限2024版。'), last]
    b = [message('user', '请查乙，限2025版。'), last]
    assert full_history_prompt('它支持什么系统？', a) != full_history_prompt('它支持什么系统？', b)


@pytest.mark.asyncio
async def test_v2_model_rewrite_is_traced_not_programmatically_merged_with_old_requirements():
    corrected = '乙的用途是什么？'
    history = [message('user', '推荐两种组件，比较容量。'),
               message('assistant', '第一种是甲，第二种是乙。'),
               message('user', '不要比较容量。')]
    model = CorrectionModel(json.dumps({'question': corrected}, ensure_ascii=False))
    pipeline = RWKVPipeline(Settings(_env_file=None, native_history_protocol='current-question-v2'),
                            None, model)
    current, task, events = await resolve_current_question(pipeline, '只说明第二种的用途。', history)
    assert current == corrected and task == conversation(corrected, [])
    payload = json.loads(model.corrections[0].split('\n', 1)[1])
    assert payload['history'] == [m.model_dump() for m in history]
    assert events[0]['history_protocol'] == 'current-question-v2'
    assert '比较容量' in events[0]['original_task'] and '比较容量' not in task
    assert events[0]['raw_text'] == '>done</think>' + model.correction


@pytest.mark.asyncio
async def test_v2_no_history_stays_byte_identical_without_a_model_call():
    model = CorrectionModel('unused')
    pipe = RWKVPipeline(Settings(_env_file=None, native_history_protocol='current-question-v2'), None, model)
    question = ' 保留原问题😀\n'
    current, task, events = await resolve_current_question(pipe, question, [])
    assert (current, task, events) == (question, conversation(question, []), [])
    assert not model.corrections


@pytest.mark.asyncio
@pytest.mark.parametrize('status,raw', [('timeout', None), ('budget_exceeded', None),
                                        ('completed', '>done</think>invalid')])
async def test_empty_evidence_does_not_hide_reader_execution_or_parse_failures(status, raw):
    model = FakeModel(resolver_outputs={'s': native_result('resolver', raw, status)})
    response = await RWKVPipeline(settings(native_empty_evidence_policy='fail'),
        FakeIndex({'alpha query': [hit('s', '资料')], 'beta query': []}), model).ask(SearchRequest(question='q'))
    assert response.generation['status'] == 'resolver_failed'
    assert response.generation['stage_status']['writer'] == 'not_called'
    assert response.answer == '' and response.generation['raw_model_answer'] is None
    assert not any(c['stage'] == 'writer' for c in model.calls)
    assert search_failure_category(response.model_dump()) == 'evidence_extraction_failed'
    assert search_failure_reason(response.model_dump()) == 'native_resolver_failed'


@pytest.mark.asyncio
async def test_empty_selection_after_planner_failure_retains_upstream_reason():
    model = FakeModel(planner_result=native_result('planner', None, 'timeout'))
    response = await RWKVPipeline(settings(native_empty_evidence_policy='fail'),
        FakeIndex({'q': []}), model).ask(SearchRequest(question='q'))
    assert response.generation['status'] == 'planner_failed'
    assert response.generation['stage_status']['writer'] == 'not_called'


@pytest.mark.asyncio
async def test_matrix_empty_policy_stops_before_writer_or_review():
    model = MatrixModel()
    response = await RWKVPipeline(matrix_settings(native_empty_evidence_policy='fail',
        native_matrix_max_rounds=1), FakeIndex({'A capacity': [], 'B capacity': []}), model).ask(
            SearchRequest(question='Compare A B'))
    assert not response.sources and response.answer == ''
    assert response.generation['status'] == 'no_evidence'
    assert response.generation['stage_status']['writer'] == 'not_called'
    assert not any(k['stage'] == 'writer' or '核验一个回答项目' in p for p, k in model.calls)


@pytest.mark.asyncio
async def test_materials_and_internal_writer_calls_cannot_bypass_empty_policy():
    model = FakeModel()
    pipe = RWKVPipeline(settings(native_empty_evidence_policy='fail'), None, model)
    response = await pipe.ask_materials('q', [])
    assert response.generation['status'] == 'no_evidence'
    assert response.generation['stage_status']['writer'] == 'not_called'
    internal = await pipe._call('do not send', stage='writer', max_tokens=32, sources=[])
    assert internal.status == 'no_evidence' and internal.raw_text is None
    assert internal.trace['completion_attempted'] is False
    assert not model.calls


@pytest.mark.asyncio
async def test_partial_reader_failure_keeps_selected_evidence_and_original_writer_output():
    raw = '>done</think>保留原文[资料 1]'
    model = FakeModel(resolver_outputs={'bad': native_result('resolver', None, 'timeout')},
                      writer_raw=raw)
    pipe = RWKVPipeline(settings(native_empty_evidence_policy='fail'),
        FakeIndex({'alpha query': [hit('bad', '资料'), hit('good', '有效原文')], 'beta query': []}), model)
    response = await pipe.ask(SearchRequest(question='q'))
    assert response.generation['status'] == 'resolver_partial_failure'
    assert response.answer == raw and response.sources[0].snippet == '有效原文'
    assert response.generation['stage_status']['writer'] == 'completed'


@pytest.mark.asyncio
@pytest.mark.parametrize('body,reason', [('无引用', 'missing_valid_citation'),
    ('错误引用[资料 99]', 'citation_syntax_or_identity'),
    ('错误格式[资料 N]', 'citation_syntax_or_identity')])
async def test_citation_gate_exposes_literal_reason_without_model_review_or_repair(body, reason):
    from llamaindex_retrieval.rwkv_pipeline import source_from_hit
    raw = '>done</think>' + body
    model = FakeModel(writer_raw=raw)
    pipe = RWKVPipeline(settings(native_answer_quality_policy='fail_citation'), None, model)
    response = await pipe.ask_materials('q', [source_from_hit(hit('s', '原文'))])
    assert response.answer == raw
    assert response.generation['status'] == 'answer_quality_failed'
    assert response.generation['quality_failure_reason'] == reason
    assert response.generation['citation_audit']['semantic_support_verified'] is False
    assert [c['stage'] for c in model.calls] == ['writer']


@pytest.mark.parametrize('status,category', [
    ('no_evidence', 'evidence_extraction_failed'), ('resolver_failed', 'evidence_extraction_failed'),
    ('current_question_failed', 'retrieval_failed'), ('request_timeout', 'retrieval_failed'),
    ('call_budget_exceeded', 'retrieval_failed')])
def test_new_stop_statuses_have_known_history_reasons(status, category):
    response = {'generation': {'pipeline': 'rwkv', 'status': status}}
    assert search_failure_category(response) == category
    assert search_failure_reason(response) == f'native_{status}'
