"""Local, separately configured trial server. Existing services remain unchanged."""
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import monotonic
from typing import Literal
from uuid import uuid4
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel, ConfigDict, Field

from .config import Settings
from .interactive_sources import choose_catalog, gather, add_sources
from . import entity_index, interactive_funnel, interactive_lean, interactive_needs, interactive_spans, kb_navigation, model_profiles
from .model_client import model_client_options
from .schemas import SourceItem
from .structured_native import StructuredNativeRWKVClient
from .web_retrieval import WebSearchAdapter


class Question(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: str = Field(min_length=1, max_length=10000)
    supplementary: bool = True
    parent_job_id: str | None = Field(default=None, pattern=r'^[0-9a-f]{32}$')
    retrieval_mode: Literal['knowledge_base', 'web', 'hybrid'] = 'hybrid'
    knowledge_base_ids: list[str] = Field(default_factory=list)
    pipeline: Literal['lean', 'needs', 'spans'] | None = None
    case_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_.-]{1,128}$')


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class ObservedModel:
    def __init__(self, native, job):
        self.native, self.job = native, job
        self.events = []

    async def check_budget(self, messages, **kwargs):
        result = await self.native.complete(messages, check_only=True, **kwargs)
        if result.trace.get('completion_attempted') is not False:
            raise RuntimeError('budget check must not attempt generation')
        return result

    async def complete(self, messages, **kwargs):
        started = monotonic()
        trace = kwargs.setdefault('trace', {})
        event = {'messages': deepcopy(messages), 'stage': kwargs['stage'], 'raw_text': None,
                 'evidence_ids': list(kwargs.get('evidence_ids', [])), 'trace': trace,
                 'status': 'pending'}
        self.events.append(event)
        try:
            result = await self.native.complete(messages, **kwargs)
            event.update(status=result.status, raw_text=result.raw_text, trace=result.trace)
            return result
        except asyncio.CancelledError:
            event.update(status='cancelled', provider_execution_cancelled=None)
            raise
        except Exception as error:
            event.update(status='failed', error_type=type(error).__name__)
            raise
        finally:
            event.update(elapsed_ms=round((monotonic() - started) * 1000),
                         raw_sha256=digest(event['raw_text'] or ''))
            self.job['model_calls_finished'] = sum(e['status'] != 'pending' for e in self.events)


class Trial:
    def __init__(self, config):
        config = model_profiles.resolve(config)
        self.config = config
        source = config.get('entity_index_source')
        if source not in {None, 'admin'}:
            raise ValueError('entity_index_source must be admin or absent')
        self.dynamic_entities = source == 'admin'
        if self.dynamic_entities and (config.get('entity_indexes') or config.get('web_replay', {}).get('mode') == 'case_pool'):
            raise ValueError('live entity indexes cannot mix with static indexes or frozen case pools')
        options = dict(config['settings'])
        if config.get('pipeline') == 'needs':
            # The native client tokenizes the complete wire prompt before every
            # generation. Reserve output within this ceiling too; reject oversize
            # calls explicitly, without truncating model inputs or outputs.
            options['native_context_window_tokens'] = min(options.get('native_context_window_tokens', 6000), 6000)
        self.settings = Settings(_env_file=None, **options)
        transport = httpx.AsyncHTTPTransport(retries=0, limits=httpx.Limits(
            max_connections=self.settings.native_max_concurrency, max_keepalive_connections=0))
        self.native = StructuredNativeRWKVClient(**model_client_options(self.settings, transport=transport))
        self.web = WebSearchAdapter(self.settings, replay=config.get('web_replay'))
        self.admin = httpx.AsyncClient(base_url=config['admin_base_url'], trust_env=False, timeout=60)
        self.lock = asyncio.Lock()
        self.jobs, self.tasks = {}, {}
        # Optional {knowledge_base_id: CARDS.jsonl}; files with cards are navigated hierarchically.
        self.cards = {}
        for path in config.get('kb_cards', {}).values():
            for line in Path(path).read_text().splitlines():
                card = json.loads(line)
                self.cards[card['id']] = card
        self.entity_index = (entity_index.EntityIndex.from_paths(config['entity_indexes'].values(), self.cards.values())
                             if config.get('entity_indexes') else None)
        self.folder = Path(config['records_dir'])
        self.folder.mkdir(parents=True, exist_ok=True)

    async def knowledge_bases(self):
        response = await self.admin.get('/v1/admin/knowledge-bases')
        response.raise_for_status()
        return response.json()

    async def catalog(self, ids):
        catalog, files = [], {}
        for kb in dict.fromkeys(ids):
            offset = 0
            while True:
                response = await self.admin.get('/v1/admin/files', params={
                    'knowledge_base_id': kb, 'limit': 200, 'offset': offset})
                response.raise_for_status()
                batch = response.json()
                for item in batch:
                    if item['status'] == 'ready':
                        files[item['id']] = item
                        catalog.append({'id': item['id'], 'title': item['filename']})
                if len(batch) < 200:
                    break
                offset += len(batch)
        return catalog, files

    async def live_entity_index(self, catalog, files, knowledge_base_ids):
        receipt = {'channel': 'navigation_index', 'mode': 'admin', 'versions': [], 'fallback': 'titles'}
        if not catalog:
            return None, receipt
        try:
            responses = await asyncio.gather(*(self.admin.get(
                '/v1/admin/knowledge-bases/'+quote(kb, safe='')+'/entity-index') for kb in dict.fromkeys(knowledge_base_ids)))
            records = {}
            for response in responses:
                response.raise_for_status()
                data = response.json()
                receipt['versions'].append(data['version'])
                for row in data['records']:
                    if row['id'] in records:
                        raise ValueError('duplicate live entity file')
                    records[row['id']] = row
            valid = []
            for row in catalog:
                record = records.get(row['id'])
                item = files[row['id']]
                revision = item.get('last_indexed_revision') or {}
                if (record and record['source_sha256'] == item['sha256'] and record['title'] == row['title']
                        and record['parsed_snapshot_sha256'] == revision.get('parsed_snapshot_sha256')
                        and record.get('status') == 'completed' and entity_index.valid_terms(record.get('parsed'), 12)):
                    valid.append(record)
            receipt.update(indexed_files=len(valid), unindexed_files=len(catalog)-len(valid),
                           fallback='supplement_titles' if valid and len(valid) < len(catalog) else None if valid else 'titles')
            return (entity_index.EntityIndex(valid, self.cards.values()) if valid else None), receipt
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            receipt.update(reason='index_unavailable', error_type=type(error).__name__)
            return None, receipt

    async def run(self, job, payload):
        model = ObservedModel(self.native, job)
        receipts = []
        def progress(text):
            job['phase'] = text
        try:
            if (payload.pipeline or self.config.get('pipeline')) == 'lean':
                await self.run_lean(job, payload, model, receipts)
                return
            if self.web.replay.mode == 'case_pool':
                raise ValueError('case_pool trial requests require pipeline lean')
            async with asyncio.timeout(900):
                async with self.lock:
                    job['status'] = 'running'
                    bases = {b['id']: b for b in await self.knowledge_bases()}
                    if any(kb not in bases for kb in payload.knowledge_base_ids):
                        raise ValueError('unknown_knowledge_base')
                    question = payload.question
                    progress('整理问题与上下文')
                    parent = None
                    if payload.parent_job_id:
                        previous = self.jobs.get(payload.parent_job_id)
                        if previous is None:
                            previous = json.loads((self.folder / (payload.parent_job_id + '.json')).read_text())
                        if previous['status'] != 'completed':
                            raise ValueError('parent_question_not_completed')
                        result = previous.get('result', {})
                        previous_answer = result.get('answer') or {}
                        parsed = previous_answer.get('generation', {}).get('parsed_answer')
                        parent = {'context': previous.get('context', {}).get('resolved_question') or previous['question'],
                                  'answer': parsed['answer'] if parsed else None}
                    question, event = await interactive_funnel.contextualize(model, question, parent)
                    job['context'] = {'resolved_question': question, 'event': event,
                                      'parent_job_id': payload.parent_job_id}
                    if question is None:
                        job.update(status='failed', phase='上下文整理未完成')
                        return
                    catalog, files = (await self.catalog(payload.knowledge_base_ids)
                        if payload.retrieval_mode != 'web' else ([], {}))
                    cache = {}
                    source_slots = asyncio.Semaphore(8)
                    async def document(file_id):
                        async with source_slots:
                            if file_id not in cache:
                                response = await self.admin.get('/v1/admin/files/' + file_id + '/download')
                                response.raise_for_status()
                                text = response.content.decode('utf-8')
                                item = files[file_id]
                                cache[file_id] = SourceItem(id='K' + digest(file_id + '\0' + text)[:16],
                                    document_id=file_id, title=item['filename'], source='knowledge_base',
                                    score=0, snippet=text, metadata={'knowledge_base_id': item['knowledge_base_id'],
                                        'sha256': digest(text), 'source_kind': 'stored_original'})
                            return cache[file_id].model_copy(deep=True)

                    async def acquire(query, channels=None):
                        # channels=None keeps the funnel behaviour: every permitted channel.
                        acquired = []
                        async def kb():
                            progress(f'从知识库目录选择资料 · {len(catalog)} 份文档')
                            if all(c['id'] in self.cards for c in catalog):
                                ids, receipt = await kb_navigation.choose(model, query,
                                    [self.cards[c['id']] for c in catalog],
                                    **self.config.get('kb_navigation_options', {}))
                                receipts.append({'channel': 'knowledge_base', 'mode': 'navigation_cards',
                                    'catalog_size': len(catalog), 'selected_file_ids': ids, **receipt})
                            else:
                                ids, decisions = await choose_catalog(model, query, catalog)
                                receipts.append({'channel': 'knowledge_base', 'query': query,
                                    'catalog_size': len(catalog), 'selected_file_ids': ids, 'decisions': decisions})
                            return await gather([document(i) for i in ids])
                        async def web():
                            progress('查找联网资料')
                            try:
                                hits, trace = await self.web.search(query, full_snapshot=True)
                                receipts.append({'channel': 'web', 'query': query, 'trace': trace})
                                return [SourceItem(id='W' + digest(h.metadata['uri'] + '\0' + h.text)[:16],
                                    document_id=h.document_id, title=h.metadata['title'], source='web',
                                    uri=h.metadata['uri'], score=0, snippet=h.text, metadata=h.metadata) for h in hits]
                            except Exception as error:
                                receipts.append({'channel': 'web', 'query': query, 'status': 'failed',
                                    'error_type': type(error).__name__, 'error': getattr(error, 'safe_code', 'web_unavailable'),
                                    **({'provider_operation': error.operation} if getattr(error, 'operation', None) else {}),
                                    **({'web_replay': error.web_replay} if getattr(error, 'web_replay', None) else {})})
                                return []
                        jobs = []
                        if catalog and (channels is None or 'knowledge_base' in channels):
                            jobs.append(kb())
                        if payload.retrieval_mode != 'knowledge_base' and (channels is None or 'web' in channels):
                            jobs.append(web())
                        for group in await gather(jobs):
                            add_sources(acquired, group)
                        return acquired

                    scope = {'knowledge_bases': [bases[i]['name'] for i in dict.fromkeys(payload.knowledge_base_ids)],
                             'web': payload.retrieval_mode != 'knowledge_base'}
                    if (payload.pipeline or self.config.get('pipeline')) == 'needs':
                        result = await interactive_needs.answer(model, question, acquire=acquire,
                            scope=scope, progress=progress, supplementary=payload.supplementary,
                            **self.config.get('needs_options', {}))
                    elif (payload.pipeline or self.config.get('pipeline')) == 'spans':
                        result = await interactive_spans.answer(model, question, acquire=acquire,
                            scope=scope, progress=progress, supplementary=payload.supplementary,
                            **self.config.get('spans_options', {}))
                    else:
                        initial = await acquire(question)
                        result = await interactive_funnel.answer(model, question, initial,
                            acquire=acquire, scope=scope, progress=progress, supplementary=payload.supplementary)
                    result['acquisition'] = receipts
                    result['acquisition_incomplete'] = any(r.get('status') == 'failed' or
                        any(d['parsed'] is None for d in r.get('decisions', [])) for r in receipts)
                    job.update(result=result, status='completed', phase='回答完成')
        except asyncio.CancelledError:
            job.update(status='cancelled', phase='已取消', provider_execution_cancelled=None)
        except TimeoutError:
            job.update(status='timeout', phase='请求达到运行时限')
        except Exception as error:
            job.update(status='failed', phase='本次请求未完成', error_type=type(error).__name__)
        finally:
            job['finished_at'] = datetime.now(timezone.utc).isoformat()
            record = {**job, 'model_calls': model.events, 'acquisition': receipts}
            with (self.folder / (job['id'] + '.json')).open('x') as f:
                json.dump(record, f, ensure_ascii=False)

    async def run_lean(self, job, payload, model, receipts):
        """Include metadata acquisition in the deadline; no serial request queue."""
        options = dict(self.config.get('lean_options', {}))
        started = monotonic()
        deadline = started + options.get('deadline_seconds', 45)
        job['status'] = 'running'
        def progress(text):
            job['phase'] = text
        try:
            async with asyncio.timeout_at(deadline):
                pool = (self.web.replay.case_pool.load(payload.case_id)
                        if self.web.replay.mode == 'case_pool' else None)
                bases = {b['id']: b for b in (pool.metadata['knowledge_bases'] if pool else await self.knowledge_bases())}
                if any(i not in bases for i in payload.knowledge_base_ids):
                    raise ValueError('unknown_knowledge_base')
                if pool:
                    catalog = ([c for c in pool.catalog if c['knowledge_base_id'] in payload.knowledge_base_ids]
                               if payload.retrieval_mode != 'web' else [])
                    files = {}
                    receipts.append({'channel': 'case_pool', 'case_id': payload.case_id,
                                     'pool_sha256': pool.sha256})
                else:
                    catalog, files = (await self.catalog(payload.knowledge_base_ids)
                        if payload.retrieval_mode != 'web' else ([], {}))
                navigation_index = self.entity_index
                if self.dynamic_entities:
                    navigation_index, receipt = await self.live_entity_index(catalog, files, payload.knowledge_base_ids)
                    receipts.append(receipt)
                    options.setdefault('title_batch', 150)
                    options['cover_unindexed'] = True
                parent = None
                if payload.parent_job_id:
                    previous = self.jobs.get(payload.parent_job_id)
                    if previous is None:
                        previous = json.loads((self.folder / (payload.parent_job_id + '.json')).read_text())
                    previous_result = previous.get('result', {})
                    parent = {'question': previous['question'],
                              'answer': previous_result.get('display_text') or
                              (previous_result.get('answer', {}).get('generation', {}).get('parsed_answer') or {}).get('answer')}

                async def document(file_id):
                    if pool:
                        if file_id not in {c['id'] for c in catalog}:
                            raise ValueError('document outside permitted frozen catalog')
                        return pool.document(file_id)
                    if self.dynamic_entities:
                        response = await self.admin.get('/v1/admin/files/'+quote(file_id, safe='')+'/parsed-source')
                        response.raise_for_status()
                        source = SourceItem.model_validate(response.json())
                        expected = files[file_id]
                        if (source.document_id != file_id or source.source != 'knowledge_base'
                                or source.title != expected['filename']
                                or source.metadata.get('knowledge_base_id') != expected['knowledge_base_id']
                                or source.metadata.get('source_sha256') != expected['sha256']
                                or source.metadata.get('parsed_snapshot_sha256') != (expected.get('last_indexed_revision') or {}).get('parsed_snapshot_sha256')):
                            raise ValueError('parsed source changed after catalog read')
                        source.uri = self.config['admin_base_url'].rstrip('/')+'/v1/admin/files/'+quote(file_id, safe='')+'/source/'+expected['sha256']
                        return source
                    response = await self.admin.get('/v1/admin/files/' + file_id + '/download')
                    response.raise_for_status()
                    text = response.content.decode('utf-8')
                    item = files[file_id]
                    return SourceItem(id='K' + digest(file_id + '\0' + text)[:16], document_id=file_id,
                        title=item['filename'], source='knowledge_base', score=0, snippet=text,
                        metadata={'knowledge_base_id': item['knowledge_base_id'], 'sha256': digest(text),
                                  'source_kind': 'stored_original'})

                async def search(query, *, include_domains):
                    options = {'case_id': payload.case_id} if pool else {}
                    hits, trace = await self.web.search(query, full_snapshot=True, include_domains=include_domains, **options)
                    receipts.append({'channel': 'web', 'query': query, 'trace': trace})
                    return [SourceItem(id='W' + digest(h.metadata['uri'] + '\0' + h.text)[:16],
                        document_id=h.document_id, title=h.metadata['title'], source='web', uri=h.metadata['uri'],
                        score=0, snippet=h.text, metadata=h.metadata) for h in hits]

            setup_ms = (monotonic()-started)*1000
            result = await interactive_lean.answer(model, payload.question, catalog=catalog, document=document,
                search=search, scope={'knowledge_bases': [bases[i]['name'] for i in payload.knowledge_base_ids],
                                     'web': payload.retrieval_mode != 'knowledge_base'},
                progress=progress, parent=parent, deadline_at=deadline,
                entity_index=navigation_index,
                concurrency=self.settings.native_max_concurrency, **options)
            result['telemetry']['total_ms'] = (monotonic()-started)*1000
            result['telemetry']['stages_ms']['setup'] = setup_ms
            job['context'] = {'resolved_question': result['resolved_question'], 'parent_job_id': payload.parent_job_id}
        except Exception as error:
            # Metadata/transport failures also have a typed response and a persisted attempt.
            pipeline = interactive_lean.Lean(model, payload.question, catalog=[], document=None, search=None,
                scope={}, progress=progress, deadline_at=deadline)
            pipeline.errors.append({'stage': 'setup', 'error_type': type(error).__name__})
            result = pipeline.result()
            result['telemetry']['total_ms'] = (monotonic()-started)*1000
        result['acquisition_incomplete'] = bool(result['errors'])
        job.update(result=result, status='completed', phase='回答完成')


@asynccontextmanager
async def lifespan(app):
    config = json.loads(Path(os.environ['RWKVRAG_TRIAL_CONFIG']).read_text())
    trial = Trial(config)
    if trial.settings.native_require_fp32_runtime:
        await trial.native.verify_fp32_runtime()
    else:
        await trial.native.verify_model_identity()
    app.state.trial = trial
    yield
    for task in trial.tasks.values():
        if not task.done():
            task.cancel()
    await asyncio.gather(*trial.tasks.values(), return_exceptions=True)
    await trial.native.aclose()
    await trial.admin.aclose()


app = FastAPI(title='RWKV 资料问答试用', lifespan=lifespan)


@app.get('/v1/try/meta')
async def meta():
    trial = app.state.trial
    provider = trial.settings.web_search_provider
    key = getattr(trial.settings, f'web_{provider}_api_key', None)
    key_file = getattr(trial.settings, f'web_{provider}_api_key_file', None)
    return {'model': trial.settings.native_model, 'knowledge_bases': await trial.knowledge_bases(),
        'concurrency': trial.settings.native_max_concurrency, 'gpu': trial.config['gpu'],
        'expires_at': trial.config.get('expires_at'), 'history_supported': True,
        'default_pipeline': trial.config.get('pipeline', 'funnel'), 'funnel_fan_in': 3,
        'pipeline': trial.config.get('pipeline', 'funnel'), 'pipelines': ['lean', 'needs', 'spans'],
        'model_profile': trial.config.get('active_model'),
        'context_window_tokens': trial.settings.native_context_window_tokens,
        'web_configured': (provider == 'searxng' or bool(key and key.get_secret_value()) or
            bool(key_file and key_file.is_file())),
        'quality_verified': False}


@app.post('/v1/try/ask')
async def ask(payload: Question):
    if not payload.question.strip():
        raise HTTPException(422, '请输入问题')
    if payload.retrieval_mode == 'knowledge_base' and not payload.knowledge_base_ids:
        raise HTTPException(422, '请选择知识库')
    trial = app.state.trial
    job = {'id': uuid4().hex, 'question': payload.question, 'request': payload.model_dump(),
           'status': 'queued', 'phase': '等待处理', 'model_calls_finished': 0,
           'created_at': datetime.now(timezone.utc).isoformat()}
    trial.jobs[job['id']] = job
    trial.tasks[job['id']] = asyncio.create_task(trial.run(job, payload))
    return job


@app.get('/v1/try/jobs/{job_id}')
async def job(job_id: str):
    trial = app.state.trial
    if job_id in trial.jobs:
        return trial.jobs[job_id]
    if len(job_id) == 32 and all(c in '0123456789abcdef' for c in job_id):
        path = trial.folder / (job_id + '.json')
        if path.is_file():
            value = json.loads(path.read_text())
            value.pop('model_calls', None)
            return value
    raise HTTPException(404, '未找到这次提问')


@app.post('/v1/try/jobs/{job_id}/cancel')
async def cancel(job_id: str):
    trial = app.state.trial
    task = trial.tasks.get(job_id)
    if task is None:
        raise HTTPException(404, '未找到这次提问')
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    result = trial.jobs[job_id]
    # Cancellation can arrive before the coroutine enters its try/finally.
    if result['status'] == 'queued':
        result.update(status='cancelled', phase='已取消',
                      finished_at=datetime.now(timezone.utc).isoformat())
        with (trial.folder / (job_id + '.json')).open('x') as f:
            json.dump({**result, 'model_calls': [], 'acquisition': []}, f, ensure_ascii=False)
    return result


@app.get('/')
async def root():
    return RedirectResponse('/admin/#/try')


app.mount('/admin', StaticFiles(directory=os.environ.get('RWKVRAG_TRIAL_STATIC', '/tmp/rwkvrag-trial-unbuilt'),
                                html=True, check_dir=False), name='admin')
