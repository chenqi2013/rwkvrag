import json
from types import SimpleNamespace

import pytest

from llamaindex_retrieval.entity_index import EntityIndex
from llamaindex_retrieval.entity_index_build import build, read_inputs


class Model:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    async def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        raw, status, http = next(self.replies)
        return SimpleNamespace(raw_text=raw, status=status,
                               trace={'completion_attempted': True, 'http': http})


async def test_build_keeps_raw_failures_and_ids_cannot_escape_output_directory(tmp_path):
    model = Model([('{"entities":["臺灣"],"aliases":[]}', 'completed', []),
                   ('{broken raw', 'length', [])])
    inputs = [{'id': '../a', 'title': 'a', 'opening': 'original', 'headings': []},
              {'id': 'b', 'title': 'b', 'opening': 'original b', 'headings': []}]
    out = tmp_path/'index'
    summary = await build(model, inputs, out, concurrency=1)
    assert summary['files'] == 2 and summary['valid_files'] == 1
    assert len(model.calls) == 2
    assert json.loads((out/'records/000001.json').read_text())['raw_text'] == '{broken raw'
    assert EntityIndex.from_paths([out/'ENTITIES.jsonl']).lookup(['台湾'])[0] == ['../a']
    assert not (tmp_path/'a.json').exists()
    assert model.calls[0][1]['stage'] == 'index'
    with pytest.raises(FileExistsError):
        await build(model, inputs, out, concurrency=1)


async def test_auth_failure_stops_index_without_retry(tmp_path):
    model = Model([(None, 'http_error', [{'http_status': 401}])])
    inputs = [{'id': str(i), 'title': '', 'opening': '', 'headings': []} for i in range(3)]
    out = tmp_path/'index'
    with pytest.raises(RuntimeError, match='authentication'):
        await build(model, inputs, out, concurrency=1)
    assert len(model.calls) == 1
    receipt = json.loads((out/'INTERRUPTED.json').read_text())
    assert receipt['complete'] is False
    assert receipt['completed_records'] == ['000000.json']
    assert not (out/'ENTITIES.jsonl').exists()


def test_input_ids_are_unique_and_headings_are_strings(tmp_path):
    path = tmp_path/'inputs.jsonl'
    row = {'id': 'x', 'title': 'Original', 'opening': 'Exact source.'}
    path.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='unique IDs'):
        read_inputs(path)
    row['headings'] = [None]
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='headings'):
        read_inputs(path)


@pytest.mark.parametrize('require_fp32', [False, True])
async def test_builder_respects_public_vs_private_profile_capabilities(tmp_path, monkeypatch, require_fp32):
    from unittest.mock import AsyncMock
    from llamaindex_retrieval import interactive_app
    from llamaindex_retrieval.entity_index_build import execute
    native = Model([('{"entities":["Original"],"aliases":[]}', 'completed', [])])
    native.verify_model_identity = AsyncMock()
    native.verify_fp32_runtime = AsyncMock()
    if not require_fp32:
        native.verify_fp32_runtime.side_effect = AssertionError('public gateway has no private State endpoint')
    native.aclose = AsyncMock()
    trial = SimpleNamespace(native=native, settings=SimpleNamespace(native_max_concurrency=1,
        native_require_fp32_runtime=require_fp32, native_model='alias', native_base_url='https://example.org/v1'),
        admin=SimpleNamespace(aclose=AsyncMock()))
    monkeypatch.setattr(interactive_app, 'Trial', lambda _: trial)
    config = tmp_path/'config.json'
    config.write_text('{}')
    inputs = tmp_path/'inputs.jsonl'
    inputs.write_text(json.dumps({'id': 'file', 'title': 'Original', 'opening': 'Exact source.'})+'\n')
    result = await execute(config, inputs, tmp_path/'index')
    assert result['complete'] and result['valid_files'] == 1
    native.verify_model_identity.assert_awaited_once()
    assert native.verify_fp32_runtime.await_count == int(require_fp32)
    native.aclose.assert_awaited_once()
