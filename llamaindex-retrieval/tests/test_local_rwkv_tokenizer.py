from hashlib import sha256
import json

import httpx
import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.native_rwkv import NativeRWKVClient
from llamaindex_retrieval.rwkv_world_tokenizer import WorldTokenizer


@pytest.fixture
def vocabulary(tmp_path):
    entries = [(i + 1, bytes([i])) for i in range(256)]
    entries += [(257, b'ab'), (258, b'abc'), (259, '你好'.encode())]
    raw = ''.join(f'{i} {value!r} {len(value)}\n' for i, value in entries).encode()
    p = tmp_path / 'world.txt'
    p.write_bytes(raw)
    return p, sha256(raw).hexdigest()


def test_byte_greedy_longest_match_and_literal_whitespace(vocabulary):
    tokenizer = WorldTokenizer(*vocabulary)
    assert tokenizer.encode('abcab你好\n ') == [258, 257, 259, 11, 33]
    assert tokenizer.encode('') == []
    assert tokenizer.encode('😀') == [byte + 1 for byte in '😀'.encode()]


def test_vocab_hash_and_byte_coverage_are_required(vocabulary, tmp_path):
    with pytest.raises(ValueError, match='hash mismatch'):
        WorldTokenizer(vocabulary[0], '0' * 64)
    p = tmp_path / 'incomplete.txt'
    p.write_bytes(b"1 b'a' 1\n")
    with pytest.raises(ValueError, match='every byte'):
        WorldTokenizer(p, sha256(p.read_bytes()).hexdigest())


def remote_client(vocabulary, handler, **kwargs):
    return NativeRWKVClient(base_url='https://rwkv.test/v1', model='remote-world',
        prompt_protocol='g1j_plain', tokenizer_file=vocabulary[0], tokenizer_sha256=vocabulary[1],
        context_window_tokens=6000, transport=httpx.MockTransport(handler), **kwargs)


@pytest.mark.parametrize('mismatch', [False, True])
async def test_local_preflight_and_server_id_check_preserve_raw_output(vocabulary, mismatch):
    tokenizer = WorldTokenizer(*vocabulary)
    calls = []
    raw = '  {"answer":"原样"}\n'

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == '/v1/models':
            return httpx.Response(200, json={'data': [{'id': 'remote-world', 'max_model_len': 32768}]})
        assert request.url.path == '/v1/completions'
        body = json.loads(request.content)
        ids = tokenizer.encode(body['prompt'])
        assert body['add_special_tokens'] is False and body['return_token_ids'] is True
        return httpx.Response(200, json={'model': 'remote-world', 'choices': [
            {'text': raw, 'finish_reason': 'stop', 'prompt_token_ids': ids + ([42] if mismatch else []),
             'token_ids': [11]}], 'usage': {'prompt_tokens': len(ids), 'completion_tokens': 1}})

    async with remote_client(vocabulary, handler) as model:
        check = await model.complete([{'role': 'user', 'content': '你好'}], max_tokens=32,
                                     assistant_prefill='<think></think', check_only=True)
        assert check.status == 'completed' and check.trace['completion_attempted'] is False
        assert calls == ['/v1/models']
        result = await model.complete([{'role': 'user', 'content': '你好'}], max_tokens=32,
                                      assistant_prefill='<think></think')
    assert result.status == ('invalid_response' if mismatch else 'completed')
    assert result.raw_text == raw
    assert result.trace['local_tokenization']['server_ids_verified'] is (not mismatch)
    assert calls == ['/v1/models', '/v1/completions']
    assert 'no weight or FP32' in result.trace['runtime_observation']['scope']


async def test_over_budget_rejects_before_any_generation(vocabulary):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        assert request.url.path == '/v1/models'
        return httpx.Response(200, json={'data': [{'id': 'remote-world', 'max_model_len': 32768}]})

    async with remote_client(vocabulary, handler) as model:
        result = await model.complete([{'role': 'user', 'content': 'x' * 6000}], max_tokens=32,
                                      assistant_prefill='<think></think')
    assert result.status == 'budget_exceeded'
    assert result.trace['completion_attempted'] is False
    assert calls == ['/v1/models']


def test_local_tokenizer_configuration_cannot_be_silently_ignored(vocabulary):
    with pytest.raises(ValueError, match='both vocabulary'):
        Settings(_env_file=None, native_tokenizer_file=vocabulary[0])
    with pytest.raises(ValueError, match='plain native'):
        Settings(_env_file=None, native_tokenizer_file=vocabulary[0],
                 native_tokenizer_sha256=vocabulary[1])
