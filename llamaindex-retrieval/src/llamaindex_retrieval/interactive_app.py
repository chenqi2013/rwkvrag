"""Local, separately configured trial server. Existing services remain unchanged."""
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
from time import monotonic
from typing import Literal
from uuid import uuid4
from urllib.parse import quote, unquote, urlsplit

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel, ConfigDict, Field

from .config import Settings
from .interactive_sources import gather
from . import interactive_flat, model_profiles, trial_defects
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
    pipeline: Literal['flat'] | None = None


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def answer_opening(text, limit=120):
    """First sentence of a previous answer, citation marks removed; for pronoun resolution only."""
    if not text:
        return None
    first = re.split(r'(?<=[。！？!?\n])', text.strip(), maxsplit=1)[0]
    return re.sub(r'\[\d{1,3}\]', '', first).strip()[:limit] or None


def read_record(trial, job_id):
    folders = [trial.folder, *(Path(p) for p in trial.config.get('history_record_dirs', []))]
    for folder in folders:
        path = folder / (job_id + '.json')
        if path.is_file():
            return json.loads(path.read_text())
    raise FileNotFoundError('trial record not found')


async def persist_record(trial, record):
    path = trial.folder / (record['id'] + '.json')
    with path.open('x') as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(record, stream, ensure_ascii=False)
    folder = getattr(trial, 'config', {}).get('defects_dir')
    if not folder:
        return {}
    try:
        await asyncio.to_thread(trial_defects.capture_record, path, folder)
        return {'diagnostic_capture': 'saved'}
    except Exception as error:
        # Never erase a completed answer or mutate its original record.
        status = {'diagnostic_capture': 'failed', 'diagnostic_error_type': type(error).__name__}
        try:
            trial_defects.private_write(path.with_suffix('.capture-error.json'), json.dumps(status).encode())
        except OSError:
            pass
        return status


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
        options = dict(config['settings'])
        self.settings = Settings(_env_file=None, **options)
        transport = httpx.AsyncHTTPTransport(retries=0, limits=httpx.Limits(
            max_connections=self.settings.native_max_concurrency, max_keepalive_connections=0))
        self.native = StructuredNativeRWKVClient(**model_client_options(self.settings, transport=transport))
        self.web = WebSearchAdapter(self.settings, replay=config.get('web_replay'))
        self.admin = httpx.AsyncClient(base_url=config['admin_base_url'], trust_env=False, timeout=60)
        self.jobs, self.tasks = {}, {}
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

    async def run(self, job, payload):
        model = ObservedModel(self.native, job)
        receipts = []
        try:
            await self.run_flat(job, payload, model, receipts)
        except asyncio.CancelledError:
            job.update(status='cancelled', phase='已取消', provider_execution_cancelled=None)
        except TimeoutError:
            job.update(status='timeout', phase='请求达到运行时限')
        except Exception as error:
            job.update(status='failed', phase='本次请求未完成', error_type=type(error).__name__)
        finally:
            job['finished_at'] = datetime.now(timezone.utc).isoformat()
            record = {**job, 'model_calls': model.events, 'acquisition': receipts}
            job.update(await persist_record(self, record))

    def local_parsed_source(self, file_id, item, revision):
        """Admin without /parsed-source: read its own parsed snapshot, bound by the catalog's hashes."""
        uri = revision.get('parsed_snapshot_uri') or ''
        if not uri.startswith('file://') or not revision.get('parsed_snapshot_sha256'):
            return None
        path = Path(unquote(urlsplit(uri).path))
        uploads = Path(self.config.get('admin_uploads_dir', '/nonexistent')).resolve()
        if not path.resolve().is_relative_to(uploads) or not path.is_file():
            return None
        encoded = path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != revision['parsed_snapshot_sha256']:
            raise ValueError('parsed snapshot hash differs from catalog')
        parsed = json.loads(encoded)
        if (parsed.get('schema') != 'rwkvrag-parsed-source-v1' or parsed.get('file_id') != file_id
                or parsed.get('knowledge_base_id') != item['knowledge_base_id']
                or parsed.get('source_sha256') != item['sha256']):
            raise ValueError('parsed snapshot identity mismatch')
        pages = [d for d in parsed['documents'] if isinstance(d.get('text'), str) and d['text'].strip()]
        text = '\n\n'.join(d['text'].strip() for d in pages)
        return SourceItem(id='K' + digest(file_id + '\0' + text)[:16], document_id=file_id,
            title=item['filename'], source='knowledge_base', score=0, snippet=text,
            metadata={'knowledge_base_id': item['knowledge_base_id'], 'source_sha256': item['sha256'],
                      'parsed_snapshot_sha256': revision['parsed_snapshot_sha256'], 'pages': len(pages),
                      'source_kind': 'parsed_snapshot', 'content_status': 'stored_original'})

    async def uploaded_source(self, file_id, item):
        """Read the current parsed upload, independent of its subject or KB name."""
        path = '/v1/admin/files/' + quote(file_id, safe='')
        response = await self.admin.get(path + '/parsed-source')
        if response.status_code != 404:
            response.raise_for_status()
            source = SourceItem.model_validate(response.json())
            if (source.document_id != file_id or source.source != 'knowledge_base'
                    or source.title != item['filename']
                    or source.metadata.get('knowledge_base_id') != item['knowledge_base_id']
                    or source.metadata.get('source_sha256') != item['sha256']
                    or source.metadata.get('parsed_snapshot_sha256') !=
                    (item.get('last_indexed_revision') or {}).get('parsed_snapshot_sha256')):
                raise ValueError('parsed upload changed after catalog read')
            return source
        revision = item.get('last_indexed_revision') or {}
        local = self.local_parsed_source(file_id, item, revision)
        if local is not None:
            return local
        # Older admin deployments expose raw downloads only. Never decode a PDF
        # or office file as text; their canonical parsed snapshot is required.
        if Path(item['filename']).suffix.lower() not in {'.md', '.markdown', '.mdx', '.txt', '.csv', '.json'}:
            raise ValueError('parsed upload service required for this file format')
        response = await self.admin.get(path + '/download')
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != item['sha256']:
            raise ValueError('uploaded file changed after catalog read')
        text = response.content.decode('utf-8')
        return SourceItem(id='K' + digest(file_id + '\0' + text)[:16], document_id=file_id,
            title=item['filename'], source='knowledge_base', score=0, snippet=text,
            metadata={'knowledge_base_id': item['knowledge_base_id'], 'source_sha256': item['sha256'],
                      'source_kind': 'uploaded_original', 'content_status': 'stored_original'})

    def conversation(self, payload, turns=3, reused_sources=6):
        """Code-kept turn state: user words, resolved question, objects and cited pages. No model summary."""
        history, previous, job_id = [], [], payload.parent_job_id
        while job_id and len(history) < turns:
            record = self.jobs.get(job_id)
            if record is None:
                record = read_record(self, job_id)
            if record['status'] != 'completed':
                raise ValueError('parent_question_not_completed')
            result = record.get('result') or {}
            history.insert(0, {'job_id': record['id'], 'question': record['question'],
                               'resolved_question': result.get('resolved_question'),
                               'entities': result.get('entities') or [], 'names': result.get('names') or [],
                               'relation': result.get('relation'),
                               'answer_opening': answer_opening(result.get('display_text'))})
            if not previous:
                cited = {s['source_id'] for s in (result.get('graph') or {}).get('spans', [])}
                previous = [SourceItem.model_validate(s) for s in (result.get('graph') or {}).get('source_snapshots', [])
                            if s['id'] in cited][:reused_sources]
            job_id = (record.get('request') or {}).get('parent_job_id')
        return history, previous

    async def run_flat(self, job, payload, model, receipts):
        if self.web.replay.mode == 'case_pool':
            raise ValueError('flat trial uses live uploads, not evaluation case pools')
        job['status'] = 'running'
        bases = {b['id']: b for b in await self.knowledge_bases()}
        if any(k not in bases for k in payload.knowledge_base_ids):
            raise ValueError('unknown_knowledge_base')
        catalog, files = (await self.catalog(payload.knowledge_base_ids)
                          if payload.retrieval_mode != 'web' else ([], {}))
        history, previous = self.conversation(payload)

        async def search(query):
            hits, trace = await self.web.search(query, full_snapshot=True)
            receipts.append({'channel': 'web', 'query': query, 'trace': trace})
            return [SourceItem(id='W' + digest(h.metadata['uri'] + '\0' + h.text)[:16],
                    document_id=h.document_id, title=h.metadata['title'], source='web',
                    uri=h.metadata['uri'], score=0, snippet=h.text, metadata=h.metadata) for h in hits]

        async def select_files(entities, queries, publish):
            chosen, mode = interactive_flat.title_candidates(catalog, entities, queries,
                **self.config.get('flat_title_options', {}))
            errors = []

            async def one(file_id):
                try:
                    await publish(await self.uploaded_source(file_id, files[file_id]))
                except Exception as error:
                    errors.append({'file_id': file_id, 'error_type': type(error).__name__})
            await gather([one(i) for i in chosen])
            return {'mode': mode, 'catalog_size': len(catalog), 'selected_file_ids': chosen, 'file_errors': errors}

        channels = ([] if payload.retrieval_mode == 'web' else ['knowledge_base'])
        if payload.retrieval_mode != 'knowledge_base':
            channels.append('web')
        result = await interactive_flat.answer(model, payload.question, channels=channels, search=search,
            select_files=select_files, history=history, previous_sources=previous,
            progress=lambda text: job.update(phase=text), **self.config.get('flat_options', {}))
        result['acquisition_traces'] = receipts
        result['telemetry']['total_calls'] = len(model.events)
        result['telemetry']['submitted_generation_calls'] = sum(
            bool(e.get('trace', {}).get('completion_attempted')) for e in model.events)
        job.update(result=result, status='completed', phase='回答完成',
                   context={'resolved_question': result['resolved_question'], 'relation': result['relation'],
                            'parent_job_id': payload.parent_job_id})


@asynccontextmanager
async def lifespan(app):
    config = json.loads(Path(os.environ['RWKVRAG_TRIAL_CONFIG']).read_text())
    trial = Trial(config)
    try:
        try:
            if trial.settings.native_require_fp32_runtime:
                await trial.native.verify_fp32_runtime()
            else:
                await trial.native.verify_model_identity()
            trial.model_readiness = 'available'
        except (httpx.TransportError, httpx.HTTPStatusError) as error:
            unavailable = not isinstance(error, httpx.HTTPStatusError) or error.response.status_code >= 500
            if (not config.get('allow_model_unavailable_startup') or not unavailable
                    or trial.settings.native_require_fp32_runtime
                    or getattr(trial.native, '_local_tokenizer', None) is None):
                raise
            # UI/history can run during an outage. No readiness is fabricated:
            # the native client still verifies identity before any generation.
            trial.model_readiness = 'unavailable'
        app.state.trial = trial
        yield
    finally:
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
        'default_pipeline': 'flat', 'pipeline': 'flat', 'pipelines': ['flat'],
        'default_knowledge_base_ids': trial.config.get('default_knowledge_base_ids', []),
        'model_profile': trial.config.get('active_model'),
        'model_readiness': ('available' if getattr(getattr(trial, 'native', None), 'runtime_observation', None)
                            else getattr(trial, 'model_readiness', 'unknown')),
        'context_window_tokens': trial.settings.native_context_window_tokens,
        'web_configured': (provider == 'searxng' or bool(key and key.get_secret_value()) or
            bool(key_file and key_file.is_file()) or
            bool(getattr(trial.settings, f'web_{provider}_api_key_pool_file', None))),
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
        try:
            value = read_record(trial, job_id)
            value.pop('model_calls', None)
            return value
        except FileNotFoundError:
            pass
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
        result.update(await persist_record(trial, {**result, 'model_calls': [], 'acquisition': []}))
    return result


@app.get('/')
async def root():
    return RedirectResponse('/admin/#/try')


app.mount('/admin', StaticFiles(directory=os.environ.get('RWKVRAG_TRIAL_STATIC', '/tmp/rwkvrag-trial-unbuilt'),
                                html=True, check_dir=False), name='admin')
