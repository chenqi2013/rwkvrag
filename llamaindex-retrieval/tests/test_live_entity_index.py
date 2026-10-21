import json
from unittest.mock import AsyncMock

import httpx
import pytest

from llamaindex_retrieval import interactive_app as app
from test_interactive_lean import Model, source


@pytest.mark.parametrize('index_state', ['current', 'pending', 'stale'])
async def test_live_index_refreshes_without_exports_and_pending_file_stays_reachable(tmp_path, index_state):
    original = source('doc', channel='knowledge_base', text='Original fact from parsed document.')
    original.title = 'Original'
    fid = original.document_id
    original.metadata.update(knowledge_base_id='kb', source_sha256='a'*64, parsed_snapshot_sha256='b'*64)
    trial = app.Trial({'pipeline': 'lean', 'settings': {}, 'entity_index_source': 'admin',
                      'admin_base_url': 'http://admin', 'records_dir': str(tmp_path)})
    native, admin = trial.native, trial.admin
    model = Model(entity_terms=['entity'])
    trial.native = model
    trial.knowledge_bases = AsyncMock(return_value=[{'id': 'kb', 'name': 'KB'}])
    trial.catalog = AsyncMock(return_value=([{'id': fid, 'title': 'Original'}], {
        fid: {'id': fid, 'knowledge_base_id': 'kb', 'filename': 'Original', 'sha256': 'a'*64,
              'last_indexed_revision': {'parsed_snapshot_sha256': 'b'*64}}}))
    requested = []
    def route(request):
        requested.append(request.url.path)
        if request.url.path.endswith('/entity-index'):
            return httpx.Response(200, json={'version': 'v1', 'complete': index_state != 'pending',
                'records': [] if index_state == 'pending' else [{'id': fid, 'title': 'Original', 'status': 'completed',
                    'parsed': {'entities': ['entity'], 'aliases': []},
                    'source_sha256': ('c' if index_state == 'stale' else 'a')*64,
                    'parsed_snapshot_sha256': 'b'*64}]})
        if request.url.path.endswith('/parsed-source'):
            return httpx.Response(200, json=original.model_dump(mode='json'))
        raise AssertionError('unexpected request '+str(request.url))
    trial.admin = httpx.AsyncClient(base_url='http://admin', transport=httpx.MockTransport(route))
    job = {'id': 'd'*32, 'question': 'q', 'status': 'queued'}
    try:
        await trial.run(job, app.Question(question='q', knowledge_base_ids=['kb'], retrieval_mode='knowledge_base'))
        result = job['result']
        assert result['response_type'] == 'answered'
        kinds = [n['kind'] for n in result['graph']['nodes']]
        assert ('navigation_entities' in kinds) == (index_state == 'current')
        assert ('navigation_titles' in kinds) == (index_state != 'current')
        assert result['graph']['spans'][0]['text'] == original.snippet
        assert result['answer']['sources'][0]['uri'].startswith('http://admin/v1/admin/files/')
        record = json.loads((tmp_path/('d'*32+'.json')).read_text())
        state = next(r for r in record['acquisition'] if r['channel'] == 'navigation_index')
        assert (state['fallback'] is None) == (index_state == 'current')
        assert requested[-1].endswith('/parsed-source')
    finally:
        await trial.admin.aclose()
        await admin.aclose()
        await native.aclose()


async def test_live_catalog_and_source_revision_must_match_before_reading(tmp_path):
    trial = app.Trial({'pipeline': 'lean', 'settings': {}, 'entity_index_source': 'admin',
        'admin_base_url': 'http://admin', 'records_dir': str(tmp_path)})
    native, admin = trial.native, trial.admin
    trial.native = Model()
    trial.knowledge_bases = AsyncMock(return_value=[{'id': 'kb', 'name': 'KB'}])
    trial.catalog = AsyncMock(return_value=([{'id': 'file', 'title': 'Original'}], {'file': {
        'knowledge_base_id': 'kb', 'filename': 'Original', 'sha256': 'a'*64,
        'last_indexed_revision': {'parsed_snapshot_sha256': 'b'*64}}}))
    changed = source('changed', channel='knowledge_base')
    changed.document_id = 'file'
    changed.metadata.update(knowledge_base_id='kb', source_sha256='c'*64, parsed_snapshot_sha256='b'*64)
    def route(request):
        if request.url.path.endswith('/entity-index'):
            return httpx.Response(200, json={'version': 'v', 'complete': False, 'records': []})
        return httpx.Response(200, json=changed.model_dump(mode='json'))
    trial.admin = httpx.AsyncClient(base_url='http://admin', transport=httpx.MockTransport(route))
    job = {'id': 'e'*32, 'question': 'q', 'status': 'queued'}
    try:
        await trial.run(job, app.Question(question='q', knowledge_base_ids=['kb'], retrieval_mode='knowledge_base'))
        assert job['result']['response_type'] == 'error'
        assert not any(n['kind'] == 'answer' for n in job['result']['graph']['nodes'])
        assert job['result']['graph']['spans'] == []
    finally:
        await trial.admin.aclose()
        await admin.aclose()
        await native.aclose()


async def test_partial_live_index_retains_valid_rows_and_reports_title_supplement(tmp_path):
    trial = app.Trial({'pipeline': 'lean', 'settings': {}, 'entity_index_source': 'admin',
        'admin_base_url': 'http://admin', 'records_dir': str(tmp_path)})
    native, admin = trial.native, trial.admin
    records = [{'id': 'known', 'title': 'Known', 'source_sha256': 'a'*64,
        'parsed_snapshot_sha256': 'b'*64, 'status': 'completed',
        'parsed': {'entities': ['term'], 'aliases': []}}]
    trial.admin = httpx.AsyncClient(base_url='http://admin', transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={'version': 'v', 'complete': False, 'records': records})))
    catalog = [{'id': 'known', 'title': 'Known'}, {'id': 'new', 'title': 'New'}]
    files = {r['id']: {'sha256': 'a'*64, 'last_indexed_revision': {'parsed_snapshot_sha256': 'b'*64}}
             for r in catalog}
    try:
        index, receipt = await trial.live_entity_index(catalog, files, ['kb'])
        assert index.lookup(['term'])[0] == ['known']
        assert receipt['unindexed_files'] == 1 and receipt['fallback'] == 'supplement_titles'
    finally:
        await trial.admin.aclose()
        await admin.aclose()
        await native.aclose()


async def test_pending_file_competes_in_final_model_selection_and_unselected_text_stays_out():
    from llamaindex_retrieval import entity_index, interactive_lean as flow, kb_navigation
    from llamaindex_retrieval.native_rwkv import NativeRWKVResult
    from llamaindex_retrieval.retrieval_eval import score_record
    class ChooseNew(Model):
        async def complete(self, messages, **kwargs):
            result = await super().complete(messages, **kwargs)
            instruction, data = messages[0]['content'].split('\n', 1)
            data = json.loads(data)
            if instruction.startswith(kb_navigation.TITLE_PICK[:20]) and len(data['documents']) == 2:
                return NativeRWKVResult('completed', '{"ids":["D2"]}', 'stop', {})
            return result
    model = ChooseNew(entity_terms=['term'])
    known = source('known', channel='knowledge_base', text='Known document facts.')
    new = source('new', channel='knowledge_base', text='New uploaded original fact.')
    sources = {s.document_id: s for s in [known, new]}
    catalog = [{'id': s.document_id, 'title': s.title} for s in [known, new]]
    index = entity_index.EntityIndex([{'id': known.document_id, 'title': known.title, 'status': 'completed',
                                     'parsed': {'entities': ['term'], 'aliases': []}}])
    pipeline = flow.Lean(model, 'question', catalog=catalog,
        document=AsyncMock(side_effect=lambda fid: sources[fid]), search=AsyncMock(return_value=[]),
        scope={'knowledge_bases': ['kb'], 'web': False}, progress=lambda _: None,
        entity_index=index, cover_unindexed=True)
    result = await pipeline.run()
    assert result['response_type'] == 'answered'
    assert [s['source_id'] for s in result['graph']['spans']] == [new.id]
    writer = next(data for instruction, data, _ in model.calls if instruction == flow.ANSWER)
    assert [s['text'] for s in writer['spans']] == [new.snippet]
    assert result['telemetry']['calls_by_stage']['navigation_titles'] == 2
    assert result['telemetry']['calls_by_stage']['navigation_entities'] == 1
    gold = {'case_id': 'case', 'gold_file_ids': [new.document_id], 'gold_pages': [],
            'needs': [{'id': 'G1', 'answerable': True, 'span_grades': {new.id: {
                'source_admissible': True, 'reviewed_all_spans': True, 'default': 0, 'overrides': {'S1': 3}}}}]}
    scored = score_record({'result': result}, gold, record_sha256='synthetic')
    assert scored['navigation']['picked_file_ids'] == [new.document_id]
    assert scored['navigation']['gold_in_top3'] is True


def test_live_index_configuration_cannot_open_frozen_case_pools(tmp_path, monkeypatch):
    from unittest.mock import Mock
    client = Mock(side_effect=AssertionError('must reject before client construction'))
    monkeypatch.setattr(app, 'StructuredNativeRWKVClient', client)
    with pytest.raises(ValueError, match='frozen case pools'):
        app.Trial({'pipeline': 'lean', 'settings': {}, 'entity_index_source': 'admin',
                   'web_replay': {'mode': 'case_pool', 'store': str(tmp_path)}})
    client.assert_not_called()
