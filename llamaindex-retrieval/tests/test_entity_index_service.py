import asyncio
from copy import deepcopy
from hashlib import sha256
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.entity_index_service import EntityIndexService, load_source
from llamaindex_retrieval.repository import MongoRepository, utc_now
from llamaindex_retrieval.source_revisions import prepare_revision
from llamaindex_retrieval.tasks import TaskManager


class Cursor:
    def __init__(self, rows):
        self.rows = deepcopy(rows)

    def __aiter__(self):
        async def rows():
            for row in self.rows:
                yield row
        return rows()

    async def to_list(self, length=None):
        return self.rows if length is None else self.rows[:length]


class Files:
    def __init__(self, repo):
        self.repo = repo

    def find(self, query, projection=None):
        return Cursor([row for row in self.repo.items.values() if all(
            key == 'last_indexed_revision' and bool(row.get(key)) if isinstance(value, dict)
            else row.get(key) == value for key, value in query.items())])


class Repo:
    """Atomic in-memory lifecycle fixture; real Mongo predicates are reused below."""
    def __init__(self, item):
        self.items = {item['id']: deepcopy(item)}
        self.jobs, self.attempts = {}, {}
        self.files = Files(self)
        self.knowledge_bases = {'kb', 'other'}

    async def get_file(self, fid):
        return deepcopy(self.items.get(fid))

    async def get_knowledge_base(self, kb):
        return {'id': kb} if kb in self.knowledge_bases else None

    async def create_job(self, kind, payload, *, job_id=None):
        job_id = job_id or 'job-'+str(len(self.jobs))
        self.jobs.setdefault(job_id, {'id': job_id, 'kind': kind, 'payload': deepcopy(payload),
            'status': 'pending', 'created_at': utc_now()})
        return deepcopy(self.jobs[job_id])

    async def get_job(self, jid):
        return deepcopy(self.jobs.get(jid))

    async def update_job(self, jid, values):
        if jid in self.jobs:
            self.jobs[jid].update(deepcopy(values))
        return await self.get_job(jid)

    async def update_file(self, fid, values):
        self.items[fid].update(deepcopy(values))
        return await self.get_file(fid)

    async def claim_entity_job(self, jid, owner, lease_seconds):
        job = self.jobs[jid]
        if not (job['status'] == 'pending' or (job['status'] == 'running'
                and job.get('entity_lease_until', utc_now()) < utc_now())):
            return None
        job.update(status='running', entity_owner=owner,
                   entity_lease_until=utc_now()+timedelta(seconds=lease_seconds),
                   entity_fence=job.get('entity_fence', 0)+1)
        return deepcopy(job)

    async def claim_index_job(self, jid):
        if self.jobs[jid]['status'] != 'pending':
            return False
        self.jobs[jid]['status'] = 'running'
        return True

    async def update_entity_job(self, jid, owner, values):
        if jid not in self.jobs or self.jobs[jid].get('entity_owner') != owner:
            return False
        await self.update_job(jid, values)
        return True

    def matching(self, binding):
        row = self.items.get(binding['file_id'])
        if row is None:
            return False
        for key, value in MongoRepository.entity_source_query(binding).items():
            current = row
            for part in key.split('.'):
                current = current.get(part) if isinstance(current, dict) else None
            if current != value:
                return False
        return True

    async def reserve_entity_index(self, binding, state):
        if not self.matching(binding):
            return False
        item = self.items[binding['file_id']]
        if (item.get('entity_index') or {}).get('job_id') == state['job_id']:
            return False
        item['entity_index'] = deepcopy(state)
        return True

    async def publish_entity_index(self, binding, jid, values, *, owner, fence, lease_until=None):
        if not self.matching(binding):
            return False
        item = self.items[binding['file_id']]
        if item['entity_index']['job_id'] != jid:
            return False
        state = item['entity_index']
        if lease_until is not None:
            if state.get('lease_fence', 0) > fence:
                return False
            values = {**values, 'owner_token': owner, 'lease_fence': fence, 'lease_until': lease_until}
        elif (state.get('owner_token') != owner or state.get('lease_fence') != fence
              or state.get('lease_until', utc_now()) <= utc_now()):
            return False
        item['entity_index'].update(deepcopy(values))
        return True

    async def fail_unfinished_entity_index(self, binding, jid):
        if self.matching(binding):
            state = self.items[binding['file_id']]['entity_index']
            if state.get('job_id') == jid and state['status'] in {'pending', 'processing'}:
                state.update(status='failed', error='index_publication_incomplete')

    async def save_entity_attempt(self, attempt):
        self.attempts.setdefault(attempt['id'], deepcopy(attempt))

    async def completed_entity_attempt(self, jid):
        return next((deepcopy(a) for a in self.attempts.values()
                     if a['job_id'] == jid and a['execution_complete']), None)


class Model:
    def __init__(self):
        self.raw = '{"entities":["Original"],"aliases":[]}'
        self.status, self.http = 'completed', []
        self.calls = 0
        self.on_call = None
        self.verify_model_identity = AsyncMock()
        self.verify_fp32_runtime = AsyncMock()
        self.aclose = AsyncMock()

    async def complete(self, messages, **kw):
        trace = {'completion_attempted': not kw.get('check_only'), 'messages': messages, 'http': self.http}
        if kw.get('check_only'):
            return SimpleNamespace(status='completed', raw_text=None, trace=trace)
        self.calls += 1
        kw.get('trace', {}).update(trace)
        if self.on_call:
            await self.on_call()
        return SimpleNamespace(status=self.status, raw_text=self.raw, trace=trace)


def item(settings, text='# Original\nA complete source fact.', fid='file', kb='kb'):
    path = settings.upload_dir.parent/(fid+'-'+sha256(text.encode()).hexdigest()[:8]+'.md')
    path.write_text(text)
    _, revision = prepare_revision(settings, path, fid, kb, sha256(text.encode()).hexdigest())
    return {'id': fid, 'knowledge_base_id': kb, 'filename': 'original.md', 'path': str(path),
            'sha256': revision['source_sha256'], 'status': 'ready', 'last_indexed_revision': revision}


@pytest.fixture
def setup(tmp_path):
    settings = Settings(_env_file=None, upload_dir=tmp_path/'uploads', entity_index_auto_generate=True)
    original = item(settings)
    repo, tasks, model = Repo(original), SimpleNamespace(submit=Mock()), Model()
    service = EntityIndexService(settings, repo, tasks, model)
    service.verified = True  # Isolate lifecycle tests from the separately tested startup probe.
    return settings, repo, tasks, model, service


async def test_automatic_enqueue_is_idempotent_and_preserves_exact_model_output(setup):
    _, repo, tasks, model, service = setup
    one, two = await asyncio.gather(service.enqueue('file'), service.enqueue('file'))
    assert one['id'] == two['id']
    await asyncio.gather(service.generate(one), service.generate(two))
    assert model.calls == 1
    assert repo.items['file']['status'] == 'ready'
    assert repo.items['file']['entity_index']['status'] == 'ready'
    result = await service.current('kb')
    assert result['complete'] and result['indexed_files'] == 1
    assert result['records'][0]['parsed']['entities'] == ['Original']
    record = next(iter(repo.attempts.values()))['record']
    assert record['raw_text'] == model.raw
    assert record['model_calls'][0]['messages'][0]['content'].endswith(
        json.dumps(record['input'], ensure_ascii=False, separators=(',', ':')))
    assert not (await service.current('other'))['records']
    # Unchanged re-ingestion reuses the accepted index and job.
    again = service.pending_state(repo.items['file'])
    assert again['status'] == 'ready'
    assert (await service.enqueue('file'))['id'] == one['id']
    assert model.calls == 1


@pytest.mark.parametrize('deleted', [False, True])
async def test_inflight_result_cannot_overwrite_revision_or_resurrect_deleted_file(setup, deleted):
    settings, repo, _, model, service = setup
    job = await service.enqueue('file')
    async def changed():
        if deleted:
            del repo.items['file']
        else:
            repo.items['file'] = item(settings, '# Original\nA changed source fact.')
    model.on_call = changed
    await service.generate(job)
    assert repo.jobs[job['id']]['stage'] == 'obsolete'
    assert len(repo.attempts) == 1
    if deleted:
        assert 'file' not in repo.items
    else:
        assert 'entity_index' not in repo.items['file']
    assert (await service.current('kb'))['records'] == []


async def test_failed_output_preserved_and_manual_retry_has_a_new_trace(setup):
    _, repo, _, model, service = setup
    model.raw = '{broken output'
    job = await service.enqueue('file')
    await service.generate(job)
    assert repo.items['file']['status'] == 'ready'
    assert repo.items['file']['entity_index']['status'] == 'failed'
    assert (await service.current('kb'))['complete'] is False
    assert next(iter(repo.attempts.values()))['record']['raw_text'] == '{broken output'
    await service.reconcile()
    assert model.calls == 1
    retry = await service.enqueue('file', retry=True)
    assert retry['id'] != job['id']
    model.raw = '{"entities":["Original"],"aliases":[]}'
    await service.generate(retry)
    assert model.calls == 2 and len(repo.attempts) == 2
    assert (await service.current('kb'))['complete']
    assert (await service.enqueue('file'))['id'] == retry['id']


async def test_reconcile_recovers_crash_between_pending_slot_and_job_creation(setup, monkeypatch):
    _, repo, tasks, model, service = setup
    create = repo.create_job
    with monkeypatch.context() as patch:
        patch.setattr(repo, 'create_job', AsyncMock(side_effect=RuntimeError('database unavailable')))
        with pytest.raises(RuntimeError):
            await service.enqueue('file')
    assert not repo.jobs and repo.items['file']['entity_index']['status'] == 'pending'
    await service.reconcile()
    assert len(repo.jobs) == 1
    job = await repo.get_job(tasks.submit.call_args.args[0])
    await service.generate(job)
    assert model.calls == 1 and (await service.current('kb'))['complete']
    assert create is not None


async def test_saved_completion_reused_after_publication_crash_without_regeneration(setup):
    _, repo, _, model, service = setup
    job = await service.enqueue('file')
    await service.generate(job)
    old = deepcopy(repo.attempts)
    repo.jobs[job['id']]['status'] = 'pending'
    repo.items['file']['entity_index']['status'] = 'pending'
    await service.generate(job)
    assert model.calls == 1 and repo.attempts == old
    assert (await service.current('kb'))['complete']


async def test_cancellation_retains_attempt_and_restart_gets_a_separate_attempt(setup):
    _, repo, _, model, service = setup
    entered, unblock = asyncio.Event(), asyncio.Event()
    async def wait():
        entered.set()
        await unblock.wait()
    model.on_call = wait
    job = await service.enqueue('file')
    work = asyncio.create_task(service.generate(job))
    await entered.wait()
    work.cancel()
    with pytest.raises(asyncio.CancelledError):
        await work
    assert repo.jobs[job['id']]['status'] == 'pending'
    assert repo.items['file']['entity_index']['status'] == 'pending'
    interrupted = next(iter(repo.attempts.values()))
    assert interrupted['execution_complete'] is False
    assert interrupted['record']['model_calls'][0]['status'] == 'cancelled'
    model.on_call = None
    await service.generate(await repo.get_job(job['id']))
    assert model.calls == 2 and len(repo.attempts) == 2
    assert (await service.current('kb'))['complete']


async def test_corrupt_or_escaped_parsed_snapshot_stops_before_model(setup):
    _, repo, _, model, service = setup
    revision = repo.items['file']['last_indexed_revision']
    from urllib.parse import urlparse
    Path(urlparse(revision['parsed_snapshot_uri']).path).write_bytes(b'changed')
    job = await service.enqueue('file')
    await service.generate(job)
    assert model.calls == 0
    assert repo.items['file']['status'] == 'ready'
    assert repo.items['file']['entity_index']['status'] == 'failed'
    binding = service.binding(repo.items['file'])
    binding['revision']['parsed_snapshot_uri'] = Path('/etc/passwd').as_uri()
    with pytest.raises(ValueError, match='outside managed'):
        load_source(service.settings, binding)


async def test_auth_failure_blocks_other_queued_model_calls(setup):
    settings, repo, _, model, service = setup
    repo.items['second'] = item(settings, '# Other\nAnother original.', 'second')
    a, b = await service.enqueue('file'), await service.enqueue('second')
    model.status, model.raw, model.http = 'http_error', None, [{'http_status': 401}]
    await service.generate(a)
    await service.generate(b)
    assert model.calls == 1 and service.blocked
    assert repo.jobs[b['id']]['status'] == 'pending'


async def test_file_ingest_atomically_marks_index_pending_and_does_not_wait_for_model(setup, monkeypatch):
    settings, repo, _, model, service = setup
    tasks = TaskManager(settings, repo, Mock())
    tasks.submit = Mock()
    tasks.entities = service
    service.tasks = tasks
    job = await repo.create_job('file_ingest', {'file_id': 'file', 'path': repo.items['file']['path']})
    repo.items['file']['status'] = 'pending'
    monkeypatch.setattr('llamaindex_retrieval.tasks.replace_uploaded_documents',
                        lambda *args: {'nodes': 1, 'documents': 1, 'index_version': 'version'})
    await tasks._run_job(job['id'])
    assert repo.jobs[job['id']]['status'] == 'completed'
    assert repo.items['file']['status'] == 'ready'
    assert repo.items['file']['entity_index']['status'] == 'pending'
    assert model.calls == 0
    child = await repo.get_job(tasks.submit.call_args.args[0])
    assert child['kind'] == 'entity_index'
    await tasks._run_job(child['id'])
    assert model.calls == 1 and (await service.current('kb'))['complete']


async def test_admin_endpoint_returns_current_index_and_parsed_original(setup):
    from fastapi import FastAPI
    from llamaindex_retrieval.routers.admin import router
    _, repo, _, _, service = setup
    app = FastAPI()
    app.include_router(router)
    app.state.entity_index_service = service
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        pending = await client.get('/v1/admin/knowledge-bases/kb/entity-index')
        assert pending.status_code == 200 and pending.json()['complete'] is False
        job = await service.enqueue('file')
        await service.generate(job)
        result = await client.get('/v1/admin/knowledge-bases/kb/entity-index')
        assert result.json()['complete'] is True
        parsed = await client.get('/v1/admin/files/file/parsed-source')
        assert parsed.status_code == 200
        assert parsed.json()['snippet'].endswith('A complete source fact.')
        assert parsed.json()['metadata']['source_sha256'] == repo.items['file']['sha256']


async def test_model_outage_does_not_block_upload_api_startup_or_retry_bad_credentials(setup):
    _, repo, _, model, service = setup
    service.verified = False
    request = httpx.Request('GET', 'https://example.org/v1/models')
    model.verify_model_identity.side_effect = httpx.HTTPStatusError('unauthorized',
        request=request, response=httpx.Response(401, request=request))
    try:
        await service.start()
        await asyncio.sleep(.01)
        assert service.blocked == 'model_auth_quota_or_rate'
        job = await service.enqueue('file')
        await service.generate(job)
        assert repo.items['file']['status'] == 'ready'
        assert model.calls == 0
        model.verify_model_identity.assert_awaited_once()
    finally:
        await service.aclose()


async def test_ready_source_added_to_another_database_is_discovered_without_upload_hook(setup):
    settings, repo, tasks, model, service = setup
    first = await service.enqueue('file')
    await service.generate(first)
    repo.items['added'] = item(settings, '# Added\nAnother original.', fid='added', kb='other')
    await service.reconcile()
    added = await repo.get_job(tasks.submit.call_args.args[0])
    assert added['payload']['knowledge_base_id'] == 'other'
    await service.generate(added)
    assert model.calls == 2
    assert (await service.current('other'))['indexed_files'] == 1
    assert (await service.current('kb'))['indexed_files'] == 1


async def test_binary_upload_uses_parsed_pages_and_retains_original_offsets(tmp_path):
    import fitz
    settings = Settings(_env_file=None, upload_dir=tmp_path/'uploads', ocr_enabled=False,
                        entity_index_auto_generate=True)
    path = tmp_path/'two-pages.pdf'
    pdf = fitz.open()
    for text in ['First page complete fact.', 'Second page complete fact.']:
        page = pdf.new_page()
        page.insert_text((50, 50), text)
    pdf.save(path)
    pdf.close()
    documents, revision = prepare_revision(settings, path, 'pdf', 'kb', sha256(path.read_bytes()).hexdigest())
    original = {'id': 'pdf', 'knowledge_base_id': 'kb', 'filename': path.name, 'status': 'ready',
                'sha256': revision['source_sha256'], 'last_indexed_revision': revision}
    service = EntityIndexService(settings, Repo(original), SimpleNamespace(submit=Mock()), Model())
    parsed = await service.source('pdf')
    segments = parsed.metadata['parsed_segments']
    assert len(segments) == len(documents) == 2
    for segment, document in zip(segments, documents):
        assert parsed.snippet[segment['start']:segment['end']] == document.text
        assert segment['document_id'] == document.id_
    assert parsed.metadata['source_sha256'] == sha256(path.read_bytes()).hexdigest()


async def test_old_model_job_is_not_run_under_new_model_configuration(setup):
    settings, repo, tasks, model, service = setup
    job = await service.enqueue('file')
    service = EntityIndexService(settings.model_copy(update={'native_model': 'new-alias'}), repo, tasks, model)
    service.verified = True
    await service.generate(await repo.get_job(job['id']))
    assert model.calls == 0
    assert repo.jobs[job['id']]['stage'] == 'obsolete'
    await service.reconcile()
    fresh = await repo.get_job(tasks.submit.call_args.args[0])
    assert fresh['id'] != job['id'] and fresh['payload']['profile_sha256'] == service.profile


async def test_live_lease_prevents_second_process_from_regenerating(setup):
    settings, repo, tasks, model, first = setup
    entered, release = asyncio.Event(), asyncio.Event()
    async def wait():
        entered.set()
        await release.wait()
    model.on_call = wait
    job = await first.enqueue('file')
    work = asyncio.create_task(first.generate(job))
    await entered.wait()
    other_model = Model()
    second = EntityIndexService(settings, repo, tasks, other_model)
    second.verified = True
    await second.generate(job)
    assert other_model.calls == 0
    release.set()
    await work
    assert model.calls == 1 and len(repo.attempts) == 1


async def test_expired_lease_takeover_fences_off_old_execution(setup):
    settings, repo, tasks, model, first = setup
    entered, release = asyncio.Event(), asyncio.Event()
    async def wait():
        entered.set()
        await release.wait()
    model.on_call = wait
    model.raw = '{"entities":["Old"],"aliases":[]}'
    job = await first.enqueue('file')
    work = asyncio.create_task(first.generate(job))
    await entered.wait()
    repo.jobs[job['id']]['entity_lease_until'] = utc_now()-timedelta(seconds=1)
    repo.items['file']['entity_index']['lease_until'] = utc_now()-timedelta(seconds=1)
    other_model = Model()
    other_model.raw = '{"entities":["New"],"aliases":[]}'
    second = EntityIndexService(settings, repo, tasks, other_model)
    second.verified = True
    await second.generate(job)
    assert repo.items['file']['entity_index']['parsed']['entities'] == ['New']
    release.set()
    await work
    assert repo.items['file']['entity_index']['parsed']['entities'] == ['New']
    assert repo.jobs[job['id']]['status'] == 'completed'
    assert len(repo.attempts) == 2  # Both raw observations survive; only the current owner publishes.


async def test_startup_does_not_reset_another_workers_running_entity_job(setup):
    settings, repo, _, _, service = setup
    job = await service.enqueue('file')
    claim = await repo.claim_entity_job(job['id'], 'live-owner', 300)
    repo.ensure_revision_jobs = AsyncMock()
    repo.recoverable_jobs = AsyncMock(return_value=[claim])
    manager = TaskManager(settings, repo, Mock())
    manager.submit = Mock()
    await manager.start()
    assert repo.jobs[job['id']]['status'] == 'running'
    assert repo.jobs[job['id']]['entity_owner'] == 'live-owner'
    manager.submit.assert_called_once_with(job['id'])


async def test_whole_index_job_deadline_is_failed_without_failing_original(setup):
    settings, repo, tasks, model, _ = setup
    short = settings.model_copy(update={'entity_index_job_timeout_seconds': .05})
    service = EntityIndexService(short, repo, tasks, model)
    service.verified = True
    model.on_call = lambda: asyncio.sleep(10)
    job = await service.enqueue('file')
    await service.generate(job)
    assert repo.jobs[job['id']]['status'] == 'failed'
    assert repo.jobs[job['id']]['error'] == 'index_job_deadline'
    assert repo.items['file']['status'] == 'ready'
    assert repo.items['file']['entity_index']['status'] == 'failed'


async def test_mongo_claim_and_publish_include_owner_revision_and_expiry_guards():
    repo = MongoRepository.__new__(MongoRepository)
    repo.jobs = SimpleNamespace(find_one_and_update=AsyncMock(return_value={'entity_fence': 2}))
    repo.files = SimpleNamespace(update_one=AsyncMock(return_value=SimpleNamespace(matched_count=0)))
    await repo.claim_entity_job('job', 'owner', 300)
    query, update = repo.jobs.find_one_and_update.call_args.args
    assert query['kind'] == 'entity_index'
    assert query['$or'][0] == {'status': 'pending'}
    assert query['$or'][1]['status'] == 'running'
    assert query['$or'][1]['$or'][0]['entity_lease_until']['$lt'] <= utc_now()
    assert update['$set']['entity_owner'] == 'owner' and update['$inc']['entity_fence'] == 1
    binding = {'file_id': 'f', 'knowledge_base_id': 'kb', 'title': 'title',
               'revision': {'source_sha256': 'a'*64, 'parsed_snapshot_sha256': 'b'*64}}
    assert not await repo.publish_entity_index(binding, 'job', {'status': 'ready'}, owner='old', fence=1)
    query = repo.files.update_one.call_args.args[0]
    assert query['entity_index.owner_token'] == 'old' and query['entity_index.lease_fence'] == 1
    assert query['entity_index.lease_until']['$gt'] <= utc_now()
    assert query['sha256'] == 'a'*64 and query['last_indexed_revision.parsed_snapshot_sha256'] == 'b'*64
    assert repo.files.update_one.call_args.kwargs.get('upsert', False) is False
