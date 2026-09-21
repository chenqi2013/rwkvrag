import copy
import importlib.util
from pathlib import Path

import httpx
import pytest

spec = importlib.util.spec_from_file_location('repeatability', Path(__file__).with_name('run.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_only_expired_reference_changes():
    row = {'stage': 'assess', 'payload': {'messages': [{'role': 'user', 'content': '原文'}],
           'vllm_xargs': {'rwkv_state_read_ref': 'FRESH_ZERO_STATE_REF'}, 'top_k': 1}}
    before = copy.deepcopy(row)
    actual = runner.request_for(row, 'new')
    expected = copy.deepcopy(row['payload'])
    expected['vllm_xargs']['rwkv_state_read_ref'] = 'new'
    assert actual == expected and row == before


def test_no_state_added_to_other_stages():
    row = {'stage': 'writer', 'payload': {'messages': [], 'top_k': 1}}
    assert runner.request_for(row, 'new') == row['payload']
    row['payload']['vllm_xargs'] = {}
    with pytest.raises(ValueError):
        runner.request_for(row, 'new')


@pytest.mark.parametrize('mutation', ['tokens', 'reasoning', 'choices', 'content'])
def test_mismatches_stop_instead_of_becoming_success(mutation):
    row = {'prompt_token_ids': [1], 'historical_raw_text': 'x', 'historical_finish_reason': 'stop'}
    body = {'prompt_token_ids': [1], 'choices': [{'message': {'content': 'x'}, 'finish_reason': 'stop'}], 'usage': {}}
    if mutation == 'tokens':
        body['prompt_token_ids'] = [2]
    elif mutation == 'reasoning':
        body['choices'][0]['message']['reasoning'] = 'hidden'
    elif mutation == 'choices':
        body['choices'].append(copy.deepcopy(body['choices'][0]))
    else:
        body['choices'][0]['message']['content'] = None
    with pytest.raises(ValueError):
        runner.response_record(row, httpx.Response(200, json=body, request=httpx.Request('POST', 'http://test')))


def test_raw_response_not_repaired():
    row = {'prompt_token_ids': [1], 'historical_raw_text': 'old', 'historical_finish_reason': 'stop'}
    body = {'prompt_token_ids': [1], 'choices': [{'message': {'content': '>broken{'}, 'finish_reason': 'length', 'token_ids': [2, 3]}], 'usage': {}}
    result = runner.response_record(row, httpx.Response(200, json=body, request=httpx.Request('POST', 'http://test')))
    assert result['raw_text'] == '>broken{' and result['finish_reason'] == 'length'
    assert not result['historical_raw_equal']
