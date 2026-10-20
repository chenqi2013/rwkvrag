import json
from unittest.mock import AsyncMock

import pytest

from llamaindex_retrieval.case_pool import CasePool
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.direct_web import WebProviderError
from llamaindex_retrieval.schemas import SourceItem
from llamaindex_retrieval.web_retrieval import WebSearchAdapter


def freeze(tmp_path):
    pool = CasePool(tmp_path)
    source = SourceItem(id='K1', document_id='f1', title='Original', snippet=' Exact original. ',
                        source='knowledge_base', score=0)
    value = pool.freeze('C1', catalog=[{'id': 'f1', 'title': 'Original', 'knowledge_base_id': 'kb'}],
        documents={'f1': source}, web_payload={'provider': 'tavily', 'hits': [], 'source_hashes': []},
        metadata={'knowledge_bases': [{'id': 'kb', 'name': 'KB'}]})
    return pool, value, source


async def test_case_pool_ignores_query_variation_without_live_transport(tmp_path):
    pool, value, source = freeze(tmp_path/'pool')
    adapter = WebSearchAdapter(Settings(_env_file=None, web_guard_path=tmp_path/'guard.sqlite3'),
        replay={'mode': 'case_pool', 'store': tmp_path/'pool'})
    adapter._execute = AsyncMock(side_effect=AssertionError('case pool must not call live'))
    for query, domains in [('first question', None), ('unseen planner variant', ['other.org'])]:
        hits, trace = await adapter.search(query, full_snapshot=True, include_domains=domains, case_id='C1')
        assert hits == []
        assert trace['web_replay']['pool_sha256'] == value.sha256
        assert trace['web_replay']['query_independent']
    assert pool.load('C1').document('f1') == source
    assert pool.load('C1').document('f1').snippet == ' Exact original. '
    adapter._execute.assert_not_awaited()
    for case_id in (None, 'missing', '../C1'):
        with pytest.raises(WebProviderError) as error:
            await adapter.search('q', case_id=case_id)
        assert error.value.safe_code == 'web_case_pool_unavailable'
    adapter._execute.assert_not_awaited()


def test_case_pool_rejects_conflicts_incomplete_catalog_and_corrupt_blob(tmp_path):
    pool, value, source = freeze(tmp_path)
    with pytest.raises(ValueError, match='full catalog'):
        pool.freeze('bad', catalog=[], documents={'f1': source}, web_payload={})
    with pytest.raises(ValueError, match='conflicts'):
        pool.freeze('C1', catalog=[], documents={}, web_payload={})
    key = value.value['documents']['f1']
    (tmp_path/'blobs'/(key+'.json')).write_text('{}')
    with pytest.raises(ValueError, match='corrupt'):
        pool.load('C1').document('f1')


def test_case_pool_corrupt_manifest_never_changes_case_identity(tmp_path):
    pool, _, _ = freeze(tmp_path)
    path = tmp_path/'cases/C1.json'
    envelope = json.loads(path.read_text())
    envelope['value']['case_id'] = 'C2'
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match='corrupt'):
        pool.load('C1')
