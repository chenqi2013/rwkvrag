"""Durable, revision-bound automatic entity indexing of ingested sources."""
import asyncio
from hashlib import sha256
import json
import logging
from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import uuid4

import httpx

from . import entity_index
from .admin_service import AdminConflictError, AdminNotFoundError, AdminValidationError
from .interactive_spans import Reader
from .repository import utc_now
from .schemas import SourceItem

logger = logging.getLogger(__name__)


def fingerprint(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_source(settings, binding):
    """Load only the server's content-addressed parsed original, never generated text."""
    revision = binding['revision']
    uri = urlparse(revision['parsed_snapshot_uri'])
    path = Path(unquote(uri.path)).resolve()
    if (uri.scheme != 'file' or uri.netloc
            or not path.is_relative_to((settings.upload_dir/'.revisions').resolve())):
        raise ValueError('parsed snapshot outside managed source revisions')
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != revision['parsed_snapshot_sha256']:
        raise ValueError('parsed snapshot hash mismatch')
    data = json.loads(raw)
    if (data.get('schema') != 'rwkvrag-parsed-source-v1'
            or data.get('file_id') != binding['file_id']
            or data.get('knowledge_base_id') != binding['knowledge_base_id']
            or data.get('source_sha256') != revision['source_sha256']):
        raise ValueError('parsed snapshot source binding mismatch')
    documents = data['documents']
    if not documents or any(not isinstance(d.get('text'), str) for d in documents):
        raise ValueError('empty or invalid parsed original')
    parts, segments, offset = [], [], 0
    for doc in documents:
        if parts:
            parts.append('\n\n')
            offset += 2
        text = doc['text']
        segments.append({'document_id': doc['id_'], 'start': offset, 'end': offset+len(text),
                         'text_sha256': sha256(text.encode()).hexdigest()})
        parts.append(text)
        offset += len(text)
    text = ''.join(parts)
    return SourceItem(id='K'+fingerprint([binding['file_id'], revision['source_sha256'], text])[:16],
        document_id=binding['file_id'], title=binding['title'], source='knowledge_base', score=0,
        snippet=text, metadata={'file_id': binding['file_id'],
            'knowledge_base_id': binding['knowledge_base_id'], 'source_sha256': revision['source_sha256'],
            'parsed_snapshot_sha256': revision['parsed_snapshot_sha256'], 'parsed_segments': segments,
            'source_kind': 'parsed_original'})


class EntityIndexService:
    def __init__(self, settings, repository, tasks, model=None):
        self.settings, self.repo, self.tasks, self.model = settings, repository, tasks, model
        self.enabled = settings.entity_index_auto_generate and model is not None
        self.semaphore = asyncio.Semaphore(settings.entity_index_workers)
        self.blocked = None
        self.verified = False
        self.watcher = None
        self.profile = fingerprint({'model': settings.native_model, 'base_url': settings.native_base_url,
            'context': settings.native_context_window_tokens, 'protocol': settings.native_completion_protocol,
            'factory': sha256(Path(entity_index.__file__).read_bytes()).hexdigest(),
            'normalization': entity_index.normalization_manifest(), 'input_version': 'parsed-file-opening1200-headings12-v1'})

    def binding(self, item):
        revision = item.get('last_indexed_revision')
        if (item['status'] != 'ready' or not revision
                or item['sha256'] != revision['source_sha256']):
            raise AdminConflictError('来源尚未完成入库或缺少当前原文版本')
        return {'file_id': item['id'], 'knowledge_base_id': item['knowledge_base_id'],
                'title': item['filename'], 'revision': revision, 'profile_sha256': self.profile,
                'model': self.settings.native_model}

    def fresh(self, item):
        state = item.get('entity_index') or {}
        revision = item.get('last_indexed_revision') or {}
        return (item.get('status') == 'ready' and state.get('profile_sha256') == self.profile
                and state.get('source_sha256') == item.get('sha256') == revision.get('source_sha256')
                and state.get('parsed_snapshot_sha256') == revision.get('parsed_snapshot_sha256')
                and state.get('title') == item.get('filename'))

    def pending_state(self, item):
        if self.fresh(item):
            return item['entity_index']
        binding = self.binding(item)
        return {'status': 'pending', 'job_id': 'entity-'+fingerprint(binding),
                'profile_sha256': self.profile, 'source_sha256': item['sha256'],
                'parsed_snapshot_sha256': binding['revision']['parsed_snapshot_sha256'],
                'title': item['filename'], 'model': self.settings.native_model,
                'updated_at': utc_now(), 'parsed': None, 'error': None}

    async def enqueue(self, file_id, *, retry=False):
        if not self.enabled:
            raise AdminValidationError('自动检索索引未启用或模型未配置')
        item = await self.repo.get_file(file_id)
        if item is None:
            raise AdminNotFoundError('来源文件不存在')
        binding = self.binding(item)
        if retry and (item.get('entity_index') or {}).get('status') != 'failed':
            raise AdminConflictError('当前没有失败的检索索引任务')
        existing = item.get('entity_index') or {}
        job_id = ('entity-'+uuid4().hex if retry else existing['job_id']
                  if self.fresh(item) and existing.get('job_id') else 'entity-'+fingerprint(binding))
        state = {'status': 'pending', 'job_id': job_id, 'profile_sha256': self.profile,
                 'source_sha256': item['sha256'], 'parsed_snapshot_sha256': binding['revision']['parsed_snapshot_sha256'],
                 'title': item['filename'], 'model': self.settings.native_model, 'updated_at': utc_now(),
                 'parsed': None, 'error': None}
        await self.repo.reserve_entity_index(binding, state)
        current = await self.repo.get_file(file_id)
        if not current or not self.fresh(current) or (current.get('entity_index') or {}).get('job_id') != job_id:
            raise AdminConflictError('来源版本或索引任务归属已改变')
        job = await self.repo.create_job('entity_index', binding, job_id=job_id)
        if job['status'] == 'pending' and not self.blocked:
            self.tasks.submit(job_id)
        return job

    async def reconcile(self):
        if not self.enabled or self.blocked:
            return
        query = {'status': 'ready', 'last_indexed_revision': {'$exists': True}}
        async for item in self.repo.files.find(query, {'_id': 0}):
            state = item.get('entity_index') or {}
            if self.fresh(item) and state.get('status') in {'ready', 'failed'}:
                continue
            try:
                # Recover the existing pending slot, including explicit retry jobs.
                if self.fresh(item) and state.get('job_id'):
                    job = await self.repo.create_job('entity_index', self.binding(item), job_id=state['job_id'])
                    if job['status'] in {'pending', 'running'}:
                        self.tasks.submit(job['id'])
                    else:
                        await self.repo.fail_unfinished_entity_index(self.binding(item), job['id'])
                else:
                    await self.enqueue(item['id'])
            except (AdminConflictError, AdminNotFoundError):
                continue
            except (KeyError, TypeError, ValueError):
                logger.exception('invalid entity source metadata for file %s', item.get('id'))

    async def start(self):
        if not self.enabled:
            return
        async def watch():
            while True:
                try:
                    if not self.verified and not self.blocked:
                        await self.model.verify_model_identity()
                        if self.settings.native_require_fp32_runtime:
                            await self.model.verify_fp32_runtime()
                        self.verified = True
                    await self.reconcile()
                except asyncio.CancelledError:
                    raise
                except httpx.HTTPStatusError as error:
                    if error.response.status_code in {401, 402, 403, 429}:
                        self.blocked = 'model_auth_quota_or_rate'
                    logger.warning('entity index model verification failed: HTTP%s', error.response.status_code)
                except Exception:
                    logger.exception('entity index reconciliation failed')
                await asyncio.sleep(self.settings.entity_index_reconcile_seconds)
        self.watcher = asyncio.create_task(watch(), name='entity-index-reconcile')

    async def stop(self):
        if self.watcher:
            self.watcher.cancel()
            await asyncio.gather(self.watcher, return_exceptions=True)

    async def aclose(self):
        await self.stop()
        if self.model is not None:
            await self.model.aclose()

    async def generate(self, job):
        if not self.enabled:
            return
        owner = 'entity-owner-'+uuid4().hex
        lease_seconds = self.settings.entity_index_job_timeout_seconds+60
        if (job['payload'].get('profile_sha256') != self.profile
                or job['payload'].get('model') != self.settings.native_model):
            if await self.repo.claim_entity_job(job['id'], owner, lease_seconds):
                await self.repo.update_entity_job(job['id'], owner, {'status': 'completed', 'stage': 'obsolete',
                    'progress': 100, 'message': '索引配置已改变，旧任务已跳过', 'completed_at': utc_now()})
            return
        if not self.verified or self.blocked:
            return
        async with self.semaphore:
            if self.blocked:
                return
            claim = await self.repo.claim_entity_job(job['id'], owner, lease_seconds)
            if not claim:
                return
            try:
                await asyncio.wait_for(self._generate_claimed(job, owner, claim),
                                       timeout=self.settings.entity_index_job_timeout_seconds)
            except TimeoutError:
                await self.repo.publish_entity_index(job['payload'], job['id'],
                    {'status': 'failed', 'error': 'index_job_deadline', 'updated_at': utc_now()},
                    owner=owner, fence=claim['entity_fence'])
                await self.repo.update_entity_job(job['id'], owner, {'status': 'failed', 'stage': 'failed',
                    'progress': 100, 'error': 'index_job_deadline', 'completed_at': utc_now()})

    async def _generate_claimed(self, job, owner, claim):
        binding = job['payload']
        async def publish(values, *, acquire=False):
            return await self.repo.publish_entity_index(binding, job['id'], values, owner=owner,
                fence=claim['entity_fence'], lease_until=claim['entity_lease_until'] if acquire else None)
        async def update(values):
            return await self.repo.update_entity_job(job['id'], owner, values)
        from .interactive_app import ObservedModel
        observed = ObservedModel(self.model, {})
        attempt_id = 'entity-attempt-'+uuid4().hex
        try:
            if not await publish({'status': 'processing', 'updated_at': utc_now()}, acquire=True):
                await update({'status': 'completed', 'stage': 'obsolete',
                    'progress': 100, 'message': '来源已改变，旧索引任务已跳过', 'completed_at': utc_now()})
                return
            previous = await self.repo.completed_entity_attempt(job['id'])
            if previous:
                if previous['binding'] != binding:
                    raise ValueError('saved entity completion belongs to another input binding')
                record = previous['record']
                attempt_id = previous['id']
            else:
                source = await asyncio.to_thread(load_source, self.settings, binding)
                headings = [r['heading'][:128] for r in Reader.entries(source) if r['heading']][:12]
                record = await entity_index.build_record(observed, binding['file_id'], binding['title'],
                                                        source.snippet[:1200], headings)
                record['model_calls'] = observed.events
                await self.repo.save_entity_attempt({'id': attempt_id, 'job_id': job['id'],
                    'file_id': binding['file_id'], 'created_at': utc_now(), 'binding': binding,
                    'execution_complete': True, 'record': record})
            valid = record['status'] == 'completed' and entity_index.valid_terms(record.get('parsed'), 12)
            codes = [h.get('http_status') for e in record.get('model_calls', [])
                     for h in (e.get('trace') or {}).get('http', [])]
            if any(c in {401, 402, 403, 429} for c in codes):
                self.blocked = 'model_auth_quota_or_rate'
            published = await publish(
                {'status': 'ready' if valid else 'failed', 'parsed': record.get('parsed') if valid else None,
                 'attempt_id': attempt_id, 'updated_at': utc_now(),
                 'error': None if valid else record.get('validation_error') or record['status']})
            await update({'status': 'completed' if valid or not published else 'failed',
                'stage': ('completed' if valid else 'failed') if published else 'obsolete',
                'progress': 100, 'message': '检索索引已更新' if valid and published else '检索索引未发布，原文仍可检索',
                'error': None if valid or not published else 'entity_generation_failed', 'completed_at': utc_now()})
        except asyncio.CancelledError:
            if observed.events:
                await self.repo.save_entity_attempt({'id': attempt_id, 'job_id': job['id'],
                    'file_id': binding['file_id'], 'created_at': utc_now(), 'binding': binding,
                    'execution_complete': False, 'record': {'status': 'cancelled', 'model_calls': observed.events}})
            await publish({'status': 'pending', 'updated_at': utc_now()})
            await update({'status': 'pending', 'stage': 'queued'})
            raise
        except Exception as error:
            await self.repo.save_entity_attempt({'id': attempt_id, 'job_id': job['id'],
                'file_id': binding['file_id'], 'created_at': utc_now(), 'binding': binding,
                'execution_complete': False, 'record': {'status': 'failed', 'error_type': type(error).__name__,
                                                      'model_calls': observed.events}})
            await publish({'status': 'failed',
                'error': type(error).__name__, 'attempt_id': attempt_id, 'updated_at': utc_now()})
            await update({'status': 'failed', 'stage': 'failed', 'progress': 100,
                'error': type(error).__name__, 'message': '检索索引生成失败，原文未受影响', 'completed_at': utc_now()})

    async def current(self, knowledge_base_id):
        if await self.repo.get_knowledge_base(knowledge_base_id) is None:
            raise AdminNotFoundError('知识库不存在')
        rows = await self.repo.files.find({'knowledge_base_id': knowledge_base_id, 'status': 'ready'}, {'_id': 0}).to_list(length=None)
        records, states = [], []
        for item in rows:
            state = item.get('entity_index') or {}
            valid = self.fresh(item) and state.get('status') == 'ready' and entity_index.valid_terms(state.get('parsed'), 12)
            states.append({'id': item['id'], 'source_sha256': item['sha256'],
                           'status': state.get('status', 'pending') if self.fresh(item) else 'pending'})
            if valid:
                records.append({'id': item['id'], 'title': item['filename'], 'status': 'completed',
                    'parsed': state['parsed'], 'source_sha256': item['sha256'],
                    'parsed_snapshot_sha256': state['parsed_snapshot_sha256']})
        return {'version': fingerprint([self.profile, records, states]), 'profile_sha256': self.profile,
                'enabled': self.enabled, 'blocked': self.blocked, 'complete': self.enabled and len(records) == len(rows),
                'ready_files': len(rows), 'indexed_files': len(records), 'states': states, 'records': records}

    async def source(self, file_id):
        item = await self.repo.get_file(file_id)
        if item is None:
            raise AdminNotFoundError('来源文件不存在')
        return await asyncio.to_thread(load_source, self.settings, self.binding(item))
