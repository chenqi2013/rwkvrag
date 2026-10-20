"""Bounded retrieval: combined planning, parallel navigation/reading, one answer.

All semantic selections come from the model. The scheduler only budgets, binds
IDs, ranks model scores and labels immutable outputs. No automatic key rotation.
"""
import asyncio
from itertools import islice
from math import ceil
import re
from time import monotonic

from . import kb_navigation
from .interactive_needs import SCORE, score_input, score_schema
from .interactive_sources import add_sources, gather
from .interactive_spans import Reader, generate, strings

PROTOCOL = 'interactive-lean-v1'
PLAN = ('理解当前问题和上一轮上下文，写出独立完整的question。needs列出最多4个必须核实的信息项；'
        'kb_queries和web_queries各最多2条独立检索语句，只使用允许的渠道。sites最多2个官网裸域名，'
        '仅在需要限定官网时填写，不确定则为空。不要回答问题。')
PAGES = ('以下是全部联网检索结果的标题、网址与搜索摘要。选择最多3份最可能直接回答信息项的网页ID。'
         '优先符合用户要求的原始或官方来源；都不合适时返回空数组。摘要仅用于定位，不是回答依据。')
ANSWER = ('只依据给出的入选逐字原文回答完整问题，保留条件和限制，不添加常识或猜测。'
          'answer写简短回答，事实后用[片段ID]指向依据，citations列出实际引用的片段ID。'
          'not_found_needs列出没有找到可靠依据的信息项ID；对这些信息项只说明未找到，绝不提供答案或推测。')
OUTLINE = ('以下是长文的结构目录。选择最可能包含信息项答案的一个条目ID；没有则返回空数组。'
           '目录只用于定位，不能作为事实依据。')
_DOMAIN = re.compile(r'(?=.{1,253}\Z)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+'
                     r'[A-Za-z](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\Z')


def plan_schema(scope):
    return {'question': {'type': 'string', 'minLength': 1},
            'needs': {**strings(), 'minItems': 1, 'maxItems': 4},
            'kb_queries': {**strings(), 'maxItems': 2 if scope.get('knowledge_bases') else 0},
            'web_queries': {**strings(), 'maxItems': 2 if scope.get('web') else 0},
            'sites': {**strings(), 'maxItems': 2}}


class Lean:
    def __init__(self, model, question, *, catalog, document, search, scope, progress, parent=None,
                 deadline_seconds=45, deadline_at=None, answer_reserve_seconds=8, concurrency=16,
                 title_batch=250, reading_mode='first_two', final_chars=6000):
        if (deadline_seconds <= 0 or answer_reserve_seconds < 0 or concurrency < 1
                or title_batch not in {100, 150, 250} or reading_mode not in {'first_two', 'outline'}
                or final_chars < 640):
            raise ValueError('invalid lean limits')
        self.model, self.question = model, question
        self.catalog, self.document, self.search = catalog, document, search
        self.scope, self.progress, self.parent = scope, progress, parent
        self.started = monotonic()
        self.deadline = deadline_at if deadline_at is not None else self.started + deadline_seconds
        self.reserve = answer_reserve_seconds
        self.concurrency, self.title_batch = concurrency, title_batch
        self.reading_mode, self.final_chars = reading_mode, final_chars
        self.events, self.acquisition, self.sources, self.errors = [], [], [], []
        self.stages, self.paths, self.rounds = {}, [], []
        self.needs = [{'id': 'I1', 'text': question}]
        self.reader = Reader(model, question, [question], progress)
        self.scores = {}
        self.plan = None
        self.answer_event = None
        self.timed_out = False
        self.read_count = 0

    def remaining(self, final=False):
        return max(0., self.deadline - monotonic() - (0 if final else self.reserve))

    async def call(self, instruction, data, properties, *, limit=384, stage='reader', evidence=(), depth=1):
        kind = ('navigation_titles' if instruction.startswith(kb_navigation.TITLE_PICK[:20]) else
                {PLAN: 'plan', PAGES: 'page_selection', SCORE: 'score', OUTLINE: 'outline', ANSWER: 'answer'}[instruction])
        record = {'kind': kind, 'depth': depth, 'started_ms': (monotonic()-self.started)*1000,
                  'status': 'pending', 'raw_text': None, 'parsed': None}
        self.events.append(record)
        if stage == 'writer':
            self.answer_event = record
        try:
            async with asyncio.timeout(self.remaining(final=stage == 'writer')):
                record.update(await generate(self.model, instruction, data, properties, limit=limit,
                                             stage=stage, evidence=evidence))
            if record['parsed'] is None and record['status'] == 'completed':
                record['status'] = 'invalid_output'
            if record['status'] in {'error', 'failed', 'http_error', 'unavailable', 'transport_error'}:
                self.errors.append({'stage': stage, 'status': record['status']})
        except asyncio.CancelledError:
            record['status'] = 'cancelled'
            raise
        except TimeoutError:
            record['status'] = 'timeout'
            self.timed_out = True
        except Exception as error:
            record.update(status='failed', error_type=type(error).__name__)
            self.errors.append({'stage': stage, 'error_type': type(error).__name__})
        finally:
            record['elapsed_ms'] = (monotonic()-self.started)*1000-record['started_ms']
        return record

    async def planning(self):
        self.progress('理解问题并规划检索')
        event = await self.call(PLAN, {'message': self.question, 'previous': self.parent,
                                      'channels': self.scope}, plan_schema(self.scope), stage='plan', depth=1)
        value = event['parsed']
        if value and any(not _DOMAIN.fullmatch(d) for d in value['sites']):
            event.update(parsed=None, status='invalid_output', validation_error='sites must be bare domains')
            value = None
        if value is None:
            value = {'question': self.question, 'needs': [self.question],
                     'kb_queries': [self.question] if self.scope.get('knowledge_bases') else [],
                     'web_queries': [self.question] if self.scope.get('web') else [], 'sites': []}
            event['fallback'] = 'raw_question_permitted_channels'
        self.plan = value
        self.question = value['question']
        self.needs = [{'id': f'I{i+1}', 'text': text} for i, text in enumerate(value['needs'])]
        self.reader = Reader(self.model, self.question, value['needs'], self.progress)

    async def titles(self, queries):
        if not queries or not self.catalog:
            return []
        query = '\n'.join(queries)
        # The merged queries describe one question; the catalog is traversed once.
        groups = [self.catalog[i:i+self.title_batch] for i in range(0, len(self.catalog), self.title_batch)]
        slots = asyncio.Semaphore(self.concurrency)

        async def choose(group, n, depth):
            local = {f'D{i+1}': row for i, row in enumerate(group)}
            event = await self.call(kb_navigation.TITLE_PICK.format(n=n),
                {'question': query, 'documents': [{'id': k, 'title': v['title']} for k, v in local.items()]},
                kb_navigation.pick_schema(local, n), limit=48, depth=depth)
            selected = kb_navigation.picks(event, list(local), n)
            return [local[k] for k in selected]

        async def batch(group, index):
            async with slots:
                return await choose(group, 3, 2+index//self.concurrency)

        winners = [row for group in await gather([batch(g, i) for i, g in enumerate(groups)]) for row in group]
        depth = 1+ceil(len(groups)/self.concurrency)
        if len(groups) > 1 and winners:
            winners = await choose(winners, 3, depth+1)
            depth += 1
        self.paths.append(depth)
        self.acquisition.append({'channel': 'knowledge_base', 'query': query,
                                 'selected_file_ids': [r['id'] for r in winners], 'catalog_size': len(self.catalog)})
        return await gather([self.document(row['id']) for row in winners])

    async def web(self, queries, sites):
        if not queries:
            return []

        async def one(query):
            receipt = {'channel': 'web', 'query': query, 'include_domains': sites or None}
            self.acquisition.append(receipt)
            try:
                try:
                    result = await self.search(query, include_domains=sites or None)
                except Exception as error:
                    # Retry domain rejection only; quota/auth/network errors are never retried here.
                    if not sites or getattr(error, 'safe_code', '') not in {'web_upstream_http_400', 'web_upstream_http_422'}:
                        raise
                    receipt['domain_retry'] = True
                    result = await self.search(query, include_domains=None)
                receipt['status'] = 'completed'
                return result
            except Exception as error:
                receipt.update(status='failed', error=getattr(error, 'safe_code', 'web_unavailable'),
                               error_type=type(error).__name__)
                self.errors.append({'stage': 'web', 'error': receipt['error']})
                return []

        pool = []
        for group in await gather([one(q) for q in queries]):
            add_sources(pool, group)
        add_sources(self.sources, pool)
        if not pool:
            return []
        local = {f'W{i+1}': s for i, s in enumerate(pool)}
        event = await self.call(PAGES, {'question': self.question, 'needs': self.needs, 'pages': [
            {'id': k, 'title': s.title, 'url': s.uri, 'snippet': s.metadata.get('search_snippet', '')}
            for k, s in local.items()]}, kb_navigation.pick_schema(local, 3), limit=48, depth=2)
        self.paths.append(2)
        return [local[k] for k in kb_navigation.picks(event, list(local), 3)]

    async def retrieval(self):
        self.progress('并发查找知识库与联网资料')

        async def branch(work):
            try:
                async with asyncio.timeout(self.remaining()):
                    result = await work
                    add_sources(self.sources, result)
                    return result
            except TimeoutError:
                self.timed_out = True
                return []
            except Exception as error:
                self.errors.append({'stage': 'retrieval', 'error_type': type(error).__name__})
                return []
        groups = await gather([branch(self.titles(self.plan['kb_queries'])),
                               branch(self.web(self.plan['web_queries'], self.plan['sites']))])
        selected = []
        for group in groups:
            add_sources(selected, group)
        return selected

    async def reading(self, sources):
        self.progress('并发阅读入选原文')
        depth = max(self.paths, default=1)+1
        ranges = {}
        if self.reading_mode == 'outline':
            async def locate(source):
                entries = self.reader.entries(source)
                if len(source.snippet) <= 3000 or not entries:
                    return
                offered = {f'E{i+1}': row for i, row in enumerate(entries[:80])}
                event = await self.call(OUTLINE, {'question': self.question, 'needs': self.needs,
                    'entries': [{'id': i, 'heading': row['heading'], 'opening': row['preview']}
                                for i, row in offered.items()]}, kb_navigation.pick_schema(offered, 1), limit=32, depth=depth)
                picked = kb_navigation.picks(event, list(offered), 1)
                if picked:
                    row = offered[picked[0]]
                    ranges[source.id] = (row['start'], row['end'])
            await gather([locate(s) for s in sources])
            depth += 1
        windows = []
        for source in sources:
            # Keep only two windows per source; islice avoids splitting entire long documents.
            head = list(islice(self.reader.windows(source), 1 if self.reading_mode == 'outline' else 2))
            if source.id in ranges:
                head += list(islice(self.reader.windows(source, *ranges[source.id]), 1))
            windows.append([(source, w, ids) for w, ids in head])
        # Round robin gives each selected source its first window before any second.
        work = [rows[i] for i in range(2) for rows in windows if i < len(rows)][:9]
        self.read_count += len(work)
        slots = asyncio.Semaphore(self.concurrency)

        async def one(source, window, ids, index):
            async with slots:
                event = await self.call(SCORE, score_input(self.question, self.needs, source,
                    window.metadata['context_spans'], [{'id': i, 'text': self.reader.spans[i]['text']} for i in ids]),
                    score_schema(ids, [n['id'] for n in self.needs]), evidence=ids,
                    depth=depth+index//self.concurrency)
                rows = (event['parsed'] or {}).get('scores', [])
                need_ids = {n['id'] for n in self.needs}
                if any(set(r) != {'span_id', 'need_id', 'score'} or r['span_id'] not in ids
                       or r['need_id'] not in need_ids or r['score'] not in {'2', '3'} for r in rows):
                    event.update(parsed=None, status='invalid_output', validation_error='invalid scores')
                    return
                for row in rows:
                    key = (row['need_id'], row['span_id'])
                    self.scores[key] = max(self.scores.get(key, 0), int(row['score']))
        await gather([one(s, w, ids, i) for i, (s, w, ids) in enumerate(work)])

    def evidence(self):
        selected, used = [], 0
        # Stable ties follow source/window order, independent of response completion order.
        order = {s: i for i, s in enumerate(self.reader.spans)}
        ranked = {n['id']: sorted(((sid, score) for (need, sid), score in self.scores.items() if need == n['id']),
                                key=lambda row: (-row[1], order[row[0]])) for n in self.needs}
        for i in range(3):
            for need in self.needs:
                rows = ranked[need['id']]
                if i >= len(rows):
                    continue
                sid = rows[i][0]
                if sid in selected:
                    continue
                size = len(self.reader.spans[sid]['text'])
                if used+size <= self.final_chars:
                    selected.append(sid)
                    used += size
        return selected

    async def answering(self, ids):
        self.progress('根据入选证据回答')
        depth = max((e['depth'] for e in self.events), default=0)+1
        event = await self.call(ANSWER, {'question': self.question, 'needs': self.needs,
            'spans': [self.reader.span_input(i) for i in ids]},
            {'answer': {'type': 'string', 'minLength': 1}, 'citations': {**strings(ids), 'minItems': 1},
             'not_found_needs': strings([n['id'] for n in self.needs])},
            stage='writer', limit=1024, evidence=ids, depth=depth)
        if event['parsed']:
            value = event['parsed']
            if (not value['answer'].strip() or not value['citations']
                    or len(value['not_found_needs']) == len(self.needs)):
                event.update(parsed=None, status='invalid_output', validation_error='answer requires supported needs')

    async def stage(self, name, work):
        start = monotonic()
        try:
            return await work
        finally:
            self.stages[name] = (monotonic()-start)*1000

    async def run(self):
        await self.stage('plan', self.planning())
        selected = await self.stage('retrieve', self.retrieval())
        if selected and self.remaining() > 0:
            await self.stage('read', self.reading(selected))
        ids = self.evidence()
        if ids and self.remaining(final=True) > 0:
            await self.stage('answer', self.answering(ids))
        return self.result(ids)

    def result(self, ids=()):
        event = self.answer_event
        value = event['parsed'] if event else None
        if value:
            response_type = 'partial' if value['not_found_needs'] else 'answered'
            text = value['answer']
        elif event and event['status'] in {'failed', 'http_error', 'transport_error'}:
            response_type, text = 'error', '模型服务暂时不可用，本次未能核实答案。'
        elif event:
            response_type, text = 'unverified', event['raw_text'] or ''
        elif self.errors and not ids:
            response_type, text = 'error', '资料服务暂时不可用，本次未能核实答案。'
        else:
            response_type, text = 'not_found', '资料中未找到可靠依据'
        bound = {i: [self.reader.spans[i]['source'].id] for i in ids}
        parsed = {'answer': value['answer'], 'source_ids': value['citations']} if value else None
        return {'response_type': response_type, 'resolved_question': self.question,
                'not_found_needs': [n for n in self.needs if response_type == 'not_found'
                                    or (value and n['id'] in value['not_found_needs'])],
                'answer': {'answer': event['raw_text'] or '' if event else text,
                           'sources': [s.model_dump(mode='json') for s in self.sources],
                           'generation': {'status': event['status'] if event else 'no_evidence',
                               'parsed_answer': parsed, 'citation_bindings': bound,
                               'protocol': PROTOCOL, 'raw_text': event['raw_text'] if event else None}},
                'display_text': text, 'acquisition': self.acquisition, 'errors': self.errors,
                'graph': {'nodes': self.events,
                          'source_snapshots': [s.model_dump(mode='json') for s in self.sources],
                          'spans': [{'id': i, 'text': self.reader.spans[i]['text'],
                                     'source_id': self.reader.spans[i]['source'].id,
                                     'span': self.reader.spans[i]['span']} for i in ids]},
                'telemetry': {'total_ms': (monotonic()-self.started)*1000, 'stages_ms': self.stages,
                    'total_calls': len(self.events), 'critical_path_calls': max((e['depth'] for e in self.events), default=0),
                    'calls_by_stage': {kind: sum(e['kind'] == kind for e in self.events)
                                       for kind in dict.fromkeys(e['kind'] for e in self.events)},
                    'generation_calls': sum(bool((e.get('trace') or {}).get('completion_attempted')) for e in self.events),
                    'input_tokens': sum(((e.get('trace') or {}).get('usage') or {}).get('prompt_tokens', 0) for e in self.events),
                    'output_tokens': sum(((e.get('trace') or {}).get('usage') or {}).get('completion_tokens', 0) for e in self.events),
                    'sequential_stages': list(self.stages), 'reading_calls': self.read_count,
                    'deadline_expired': self.timed_out, 'response_type': response_type,
                    'calls': [{'stage': e['kind'], 'status': e['status'], 'elapsed_ms': e.get('elapsed_ms'),
                               'budget': (e.get('trace') or {}).get('budget'),
                               'usage': (e.get('trace') or {}).get('usage')} for e in self.events]},
                'fabrication_review': 'pending_offline_review', 'rounds': self.rounds}


async def answer(model, question, **kwargs):
    pipeline = Lean(model, question, **kwargs)
    try:
        async with asyncio.timeout_at(pipeline.deadline):
            return await pipeline.run()
    except TimeoutError:
        pipeline.timed_out = True
        return pipeline.result(pipeline.evidence())
    except Exception as error:
        pipeline.errors.append({'stage': 'pipeline', 'error_type': type(error).__name__})
        return pipeline.result(pipeline.evidence())
