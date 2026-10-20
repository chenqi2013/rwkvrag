"""Plan -> per-query acquisition -> parallel span selection -> three-child span merges.

Candidate pipeline (opt-in, ``pipeline: "spans"``). The model writes the
information needs and search queries, selects verbatim span IDs while reading,
keeps a span subset and reports missing needs while merging, and answers from
the kept verbatim spans. Code only splits, schedules, binds IDs and validates
structure. Unselected spans never reach a parent. Raw text is immutable.
"""
import asyncio
import json
import re
from time import monotonic

from llama_index.core import Document

from . import interactive_sources as structure
from .interactive_sources import add_sources, gather
from .verbatim_chunking import packed_verbatim_nodes

PROTOCOL = 'interactive-spans-v1'
PLAN = ('为回答当前完整问题制定检索。needs列出回答必须找到的信息项，每项一句。'
        '按渠道分别写检索语句：kb_queries查用户选择的知识库，web_queries查网络；每条可独立检索并写明对象，'
        '用户指定来源时写入来源名称或网站。只有需要某渠道时才写该渠道，用户只要求查某一来源时其他渠道返回空数组。'
        '数量按问题需要决定，不要回答问题。')
LEAF = ('并发阅读中的一段原文，已切成带ID的片段。选出能直接回答信息项的片段，'
        '只选包含所需事实的片段，不选导航、广告或泛泛介绍；没有相关片段时返回空数组。'
        '输出span_ids及这些片段覆盖的信息项need_ids。')
MERGE = ('本组片段是此前阅读选出的逐字原文。保留回答信息项所需的片段，去掉重复、'
         '无关或与问题对象不符的片段。missing列出这些片段仍未覆盖的信息项。'
         '输出span_ids和missing。')
FINAL = ('依据入选的逐字原文片段回答当前完整问题，逐项覆盖信息项，保留条件和限制。'
         '只用片段中的事实，不补猜；片段未覆盖的信息项明确说明未找到。'
         '正文用[片段ID]指向依据。输出answer和实际采用的片段ID source_ids。')
OUTLINE = ('长网页的章节目录，每项给出标题路径和开头原文。选出可能包含信息项所需内容的条目；'
           '不确定时可以选，明显无关的不选。目录只用于定位，不是事实依据。输出entry_ids。')
PAGE = ('按标题、网址、搜索摘要和开头判断这份资料对回答问题有多大用处，给0到3分：3=就是问题所问对象的'
        '直接资料（用户指定来源时须是该来源）；2=讲同一对象，可能包含答案；1=话题相关但对象不同或只是泛泛介绍；'
        '0=无关。输出score。')
REQUERY = ('以下信息项在已检索资料中仍未找到。按渠道为它们写新的检索语句（kb_queries查知识库，'
           'web_queries查网络），换一种说法或指明更可能包含该信息的来源；不需要的渠道返回空数组，'
           '无法继续核实时全部为空。不要重复已用检索语句。')

# Context budgets. A controlled test (context-length-20261019) kept the answer span 9/9 at
# ~2.8k input tokens and 3/9 at ~5.2k; every call is kept inside ~6k tokens and every
# selection call's material near ~2.5k (about 0.6 tokens per character measured).
LEAF_CHARS, LEAF_OVERLAP = 3000, 240  # Reading window per call.
SPAN_CHARS, SPAN_OVERLAP = 640, 80     # Selectable verbatim spans inside a window.
MERGE_CHARS = 4000                     # Span text offered to one merge call.
FINAL_SPAN_CHARS = 6000                # Input evidence capacity; output reservation is separate.
SECTION_CHARS = 8000                   # Outline entries larger than this are split into parts.
OUTLINE_BATCH, PREVIEW_CHARS = 80, 160
PAGE_HEAD_CHARS = 1500                 # Opening text shown to the page gate.
PAGE_MIN_SCORE = 1                     # Pages scored below this are not read.
_HEADING = re.compile(r'(?m)^ {0,3}(#{1,6})[ \t]+(.+)$')


def strings(items=None):
    schema = {'type': 'array', 'items': {'type': 'string', 'minLength': 1}}
    if items is not None:
        schema['items'] = {'type': 'string', 'enum': list(items)}
        if not items:
            schema['maxItems'] = 0
    return schema


async def generate(model, instruction, data, properties, *, limit, stage='reader', evidence=(), budget_only=False):
    """One model call; parsed only when every field matches the bound schema."""
    complete = model.check_budget if budget_only else model.complete
    result = await complete([{'role': 'user', 'content': instruction + '\n' +
        json.dumps(data, ensure_ascii=False, separators=(',', ':'))}],
        structured_schema=structure.obj(properties), max_tokens=limit,
        assistant_prefill='<think></think', temperature=0, seed=11, stage=stage,
        evidence_ids=list(evidence))
    event = {'status': result.status, 'raw_text': result.raw_text,
             'raw_sha256': structure.digest(result.raw_text or ''), 'parsed': None, 'trace': result.trace}
    if budget_only or result.status != 'completed':
        return event
    try:
        value = json.loads(result.raw_text, object_pairs_hook=structure.unique_object)
        if not isinstance(value, dict) or set(value) != set(properties):
            raise ValueError('invalid fields')
        for key, schema in properties.items():
            item = value[key]
            if schema['type'] == 'string':
                if (not isinstance(item, str) or ('enum' in schema and item not in schema['enum'])
                        or len(item) < schema.get('minLength', 0)
                        or ('maxLength' in schema and len(item) > schema['maxLength'])):
                    raise ValueError(f'invalid {key}')
                continue
            if (not isinstance(item, list) or len(item) < schema.get('minItems', 0)
                    or ('maxItems' in schema and len(item) > schema['maxItems'])):
                raise ValueError(f'invalid {key}')
            if schema['items'].get('type') == 'object':
                # Object rows are validated field by field by the caller.
                if not isinstance(item, list) or any(not isinstance(x, dict) for x in item):
                    raise ValueError(f'invalid {key}')
                continue
            allowed = schema['items'].get('enum')
            if (not isinstance(item, list) or any(not isinstance(x, str) or not x.strip() for x in item)
                    or (allowed is not None and (any(x not in allowed for x in item)
                                                 or len(item) != len(set(item))))):
                raise ValueError(f'invalid {key}')
        event['parsed'] = value
    except (ValueError, TypeError) as error:
        event['validation_error'] = str(error)
    return event


class Reader:
    def __init__(self, model, question, needs, progress, final_chars=FINAL_SPAN_CHARS, outline_chars=None,
                 page_gate=False):
        self.model, self.question, self.progress = model, question, progress
        self.needs = {f'I{i+1}': text for i, text in enumerate(needs)}
        self.final_chars, self.outline_chars = final_chars, outline_chars
        self.outlines, self.page_gate, self.gates = [], page_gate, []
        self.nodes, self.spans = [], {}  # span ID -> {'source', 'text', 'span', 'context'}

    def new_node(self, kind, parents=(), **metadata):
        node = {'id': f'N{len(self.nodes)+1}', 'kind': kind, 'parents': [p['id'] for p in parents],
                'status': 'pending', 'raw_text': None, 'parsed': None, 'span_ids': [], **metadata}
        self.nodes.append(node)
        return node

    def need_input(self):
        return [{'id': i, 'text': t} for i, t in self.needs.items()]

    def span_input(self, span_id):
        span = self.spans[span_id]
        source = span['source']
        return {'id': span_id, 'source': source.title, 'channel': source.source,
                'uri': source.uri, 'text': span['text']}

    def windows(self, source, start=0, end=None):
        """Reading windows of selectable spans; spans keep source coordinates."""
        text = source.snippet[start:end]
        document = Document(text=text, id_=source.id, metadata={'kind': source.metadata.get('kind')})
        for window in packed_verbatim_nodes(document, LEAF_CHARS, LEAF_OVERLAP, absorb_whitespace=True):
            offset = start + window.metadata['source_span']['start']
            window.metadata['source_span'] = {**window.metadata['source_span'],
                'start': offset, 'end': start + window.metadata['source_span']['end']}
            part = Document(text=window.text, id_=window.id_)
            ids = []
            for piece in packed_verbatim_nodes(part, SPAN_CHARS, SPAN_OVERLAP, absorb_whitespace=True):
                if not piece.text.strip():
                    continue
                span_id = f'S{len(self.spans)+1}'
                local = piece.metadata['source_span']
                self.spans[span_id] = {'source': source, 'text': piece.text,
                    'span': {'start': offset + local['start'], 'end': offset + local['end'],
                             'unit': 'unicode_code_points'},
                    'context': window.metadata['context_spans']}
                ids.append(span_id)
            if ids:
                yield window, ids

    @staticmethod
    def entries(source):
        """Structural outline: heading sections, oversized ones split into verbatim parts."""
        text, marks, path = source.snippet, [], []
        for match in _HEADING.finditer(text):
            level = len(match.group(1))
            path = [p for p in path if p[0] < level] + [(level, match.group(2).strip())]
            marks.append((match.start(), ' > '.join(t for _, t in path)))
        if not marks or marks[0][0] > 0:
            marks.insert(0, (0, ''))
        bounds = [(start, marks[i+1][0] if i+1 < len(marks) else len(text), heading)
                  for i, (start, heading) in enumerate(marks)]
        rows = []
        for start, end, heading in bounds:
            if not text[start:end].strip():
                continue
            parts = [(start, end)]
            if end - start > SECTION_CHARS:
                part = Document(text=text[start:end], id_=source.id)
                parts = [(start + n.metadata['source_span']['start'], start + n.metadata['source_span']['end'])
                         for n in packed_verbatim_nodes(part, LEAF_CHARS, LEAF_OVERLAP, absorb_whitespace=True)]
            for i, (a, b) in enumerate(parts):
                body = text[a:b]
                first = _HEADING.match(body)
                preview = body[first.end():] if first and i == 0 else body
                rows.append({'start': a, 'end': b, 'heading': heading,
                             'part': f'{i+1}/{len(parts)}' if len(parts) > 1 else None,
                             'preview': ' '.join(preview.split())[:PREVIEW_CHARS], 'chars': b - a})
        return rows

    async def navigate(self, source):
        """Model picks outline entries of a long page; only picked ranges are read."""
        rows = self.entries(source)
        for i, row in enumerate(rows):
            row['id'] = f'E{i+1}'
        record = {'source_id': source.id, 'entries': rows, 'decisions': []}
        self.outlines.append(record)

        async def choose(group):
            ids = [r['id'] for r in group]
            event = await generate(self.model, OUTLINE, {'question': self.question, 'needs': self.need_input(),
                'source': {'title': source.title, 'uri': source.uri, 'chars': len(source.snippet)},
                'entries': [{k: r[k] for k in ('id', 'heading', 'part', 'preview', 'chars')} for r in group]},
                {'entry_ids': strings(ids)}, limit=512, evidence=ids)
            record['decisions'].append(event)
            return event['parsed']['entry_ids'] if event['parsed'] else []
        picked = {i for group in await gather([choose(rows[i:i+OUTLINE_BATCH])
                                                for i in range(0, len(rows), OUTLINE_BATCH)]) for i in group}
        record['selected_entry_ids'] = [r['id'] for r in rows if r['id'] in picked]
        return [(r['start'], r['end']) for r in rows if r['id'] in picked]

    async def gate(self, source):
        """Page-level relevance decision; a failed decision keeps the page (recorded)."""
        event = await generate(self.model, PAGE, {'question': self.question, 'needs': self.need_input(),
            'source': {'title': source.title, 'channel': source.source, 'uri': source.uri,
                       'chars': len(source.snippet), 'content_status': source.metadata.get('content_status'),
                       'search_snippet': source.metadata.get('search_snippet') or None,
                       'opening': source.snippet[:PAGE_HEAD_CHARS]}},
            {'score': {'type': 'string', 'enum': ['0', '1', '2', '3']}}, limit=16, evidence=[source.id])
        if event['parsed'] is not None and event['parsed']['score'] not in {'0', '1', '2', '3'}:
            event.update(parsed=None, validation_error='invalid score')
        keep = event['parsed'] is None or int(event['parsed']['score']) >= PAGE_MIN_SCORE
        self.gates.append({'source_id': source.id, 'keep': keep, **event})
        return keep

    async def read(self, sources):
        web = [s for s in sources if s.source == 'web']
        if self.page_gate and web:
            # Only web results are gated; knowledge-base files were chosen by the user and catalog step.
            self.progress(f'判断网页是否相关 · {len(web)} 份')
            keep = dict(zip([s.id for s in web], await gather([self.gate(s) for s in web])))
            sources = [s for s in sources if keep.get(s.id, True)]
        ranges = {}
        long = [s for s in sources if self.outline_chars and len(s.snippet) > self.outline_chars]
        if long:
            self.progress(f'长网页先看目录 · {len(long)} 份')
            for source, picked in zip(long, await gather([self.navigate(s) for s in long])):
                ranges[source.id] = picked
        work = [(s, w, ids) for s in sources
                for a, b in ranges.get(s.id, [(0, None)]) for w, ids in self.windows(s, a, b)]
        self.progress(f'并发阅读原文 · {len(work)} 个分块')

        async def leaf(source, window, ids):
            node = self.new_node('leaf', original_text_input_ids=[source.id], offered_span_ids=ids,
                                 source_span=window.metadata['source_span'])
            event = await generate(self.model, LEAF, {'question': self.question, 'needs': self.need_input(),
                'source': {'title': source.title, 'channel': source.source, 'uri': source.uri,
                           'parent_context': window.metadata['context_spans']},
                'spans': [{'id': i, 'text': self.spans[i]['text']} for i in ids]},
                {'span_ids': strings(ids), 'need_ids': strings(self.needs)}, limit=256, evidence=ids)
            node.update(event)
            if event['parsed'] is not None:
                node['span_ids'] = event['parsed']['span_ids']
            elif node['status'] == 'completed':
                node['status'] = 'invalid_output'
            return node
        return await gather([leaf(*row) for row in work])

    async def combine(self, children, offered=None):
        offered = offered or list(dict.fromkeys(i for c in children for i in c['span_ids']))
        node = self.new_node('merge', children, offered_span_ids=offered)
        event = await generate(self.model, MERGE, {'question': self.question, 'needs': self.need_input(),
            'spans': [self.span_input(i) for i in offered]},
            {'span_ids': strings(offered), 'missing': strings(self.needs)}, limit=512, evidence=offered)
        node.update(event)
        if event['parsed'] is not None:
            node['span_ids'] = event['parsed']['span_ids']
            node['missing'] = event['parsed']['missing']
        elif node['status'] == 'completed':
            node['status'] = 'invalid_output'
        return node

    def groups(self, current):
        """Size-bounded merge jobs: at most 3 children and MERGE_CHARS of span text each.

        A node holding more than MERGE_CHARS alone is split into consecutive span batches,
        each merged by its own call, so no merge input exceeds the budget.
        """
        jobs, group, size = [], [], 0
        for node in current:
            own = self.chars([node])
            if own > MERGE_CHARS:
                batch, used = [], 0
                for span_id in node['span_ids']:
                    length = len(self.spans[span_id]['text'])
                    if batch and used + length > MERGE_CHARS:
                        jobs.append(([node], batch))
                        batch, used = [], 0
                    batch.append(span_id)
                    used += length
                jobs.append(([node], batch))
                continue
            if group and (len(group) == 3 or size + own > MERGE_CHARS):
                jobs.append((group, None))
                group, size = [], 0
            group.append(node)
            size += own
        if group:
            jobs.append((group, None))
        return jobs

    async def reduce(self, nodes):
        """Only nodes holding selected spans are merged; failed/empty ones stay in the audit graph."""
        current = [n for n in nodes if n['span_ids']]
        while len(current) > 3 or (len(current) > 1 and self.chars(current) > self.final_chars) or (
                len(current) == 1 and self.chars(current) > self.final_chars):
            self.progress(f'并发归并片段 · {len(current)} 个节点')
            before = self.chars(current)

            async def job(children, offered):
                if len(children) == 1 and offered is None:
                    return children
                node = await self.combine(children, offered)
                # A failed merge contributes only its status; the children's validated
                # selections move up unchanged instead of being lost with it.
                if node['parsed'] is not None:
                    return [node]
                return children if offered is None else [{**children[0], 'span_ids': offered}]
            merged = [n for g in await gather([job(*j) for j in self.groups(current)]) for n in g]
            merged = [n for n in merged if n['span_ids']]
            # Stop when a level removes nothing; the final call records whatever exceeds its budget.
            if self.chars(merged) >= before and len(merged) >= len(current):
                current = merged
                break
            current = merged
        return current

    def chars(self, nodes):
        return sum(len(self.spans[i]['text']) for i in dict.fromkeys(s for n in nodes for s in n['span_ids']))

    async def final(self, roots):
        kept, omitted, used = [], [], 0
        for span_id in dict.fromkeys(i for n in roots for i in n['span_ids']):
            size = len(self.spans[span_id]['text'])
            if omitted or used + size > self.final_chars:
                omitted.append(span_id)
            else:
                kept.append(span_id)
                used += size
        node = self.new_node('final', roots, offered_span_ids=kept, omitted_span_ids=omitted,
                             original_text_input_ids=list(dict.fromkeys(self.spans[i]['source'].id for i in kept)))
        event = await generate(self.model, FINAL, {'question': self.question, 'needs': self.need_input(),
            'spans': [self.span_input(i) for i in kept]},
            {'answer': {'type': 'string'}, 'source_ids': {**strings(kept), 'maxItems': len(kept)}},
            limit=2048, stage='writer', evidence=kept)
        node.update(event)
        if event['parsed'] is not None:
            node['span_ids'] = event['parsed']['source_ids']
        elif node['status'] == 'completed':
            node['status'] = 'invalid_output'
        return node

    def missing(self, roots):
        """Needs not covered: reported by the root merge, or by no leaf when there was no merge."""
        if len(roots) == 1 and 'missing' in roots[0]:
            return roots[0]['missing']
        covered = {n for leaf in self.nodes if leaf['kind'] == 'leaf' and leaf['parsed']
                   for n in leaf['parsed']['need_ids']}
        return [i for i in self.needs if i not in covered]


CHANNEL_FIELDS = {'knowledge_base': 'kb_queries', 'web': 'web_queries'}


def channels(scope):
    """Channels the caller permits; the model chooses among them per query."""
    allowed = []
    if scope.get('knowledge_bases'):
        allowed.append('knowledge_base')
    if scope.get('web'):
        allowed.append('web')
    return allowed


def planned(parsed, allowed):
    """(channel, query) pairs in model order; exact duplicates dropped."""
    return list(dict.fromkeys((c, q) for c in allowed for q in parsed[CHANNEL_FIELDS[c]]))


async def fetch(acquire, pairs):
    return await gather([acquire(q, channels=[c]) for c, q in pairs])


async def answer(model, question, *, acquire, scope, progress, supplementary=True,
                 max_rounds=2, supplement_timeout=300, final_chars=FINAL_SPAN_CHARS, outline_chars=None,
                 page_gate=False):
    started = monotonic()
    progress('制定检索')
    allowed = channels(scope)
    fields = {CHANNEL_FIELDS[c]: strings() for c in allowed}
    plan = await generate(model, PLAN, {'question': question, 'available_channels': scope},
                          {'needs': strings(), **fields}, limit=1024, stage='planner')
    output = {'protocol': PROTOCOL, 'scope': scope, 'plan': plan, 'rounds': []}
    # Exact duplicates are dropped (structural dedup); the raw plan stays in the record.
    needs = list(dict.fromkeys(plan['parsed']['needs'])) if plan['parsed'] else []
    needs = needs or [question]
    queries = planned(plan['parsed'], allowed) if plan['parsed'] else []
    # A failed or empty plan still retrieves with the resolved question on every permitted channel.
    queries = queries or [(c, question) for c in allowed]
    output['executed_queries'] = [queries]
    reader = Reader(model, question, needs, progress, final_chars, outline_chars, page_gate)
    pool = []
    for group in await fetch(acquire, queries):
        add_sources(pool, group)
    roots = await reader.reduce(await reader.read(pool))
    stop = 'not_requested'
    if supplementary:
        used = list(queries)
        try:
            async with asyncio.timeout(supplement_timeout):
                while True:
                    missing = reader.missing(roots)
                    if not missing:
                        stop = 'needs_covered'
                        break
                    if len(output['rounds']) >= max_rounds:
                        stop = 'round_limit'
                        break
                    progress(f'按缺失信息项补查 · 第 {len(output["rounds"])+1} 轮')
                    decision = await generate(model, REQUERY, {'question': question,
                        'missing': [{'id': i, 'text': reader.needs[i]} for i in missing],
                        'used_queries': [{'channel': c, 'query': q} for c, q in used],
                        'available_channels': scope}, fields, limit=512, stage='planner')
                    round_ = {'number': len(output['rounds'])+1, 'missing': missing, 'decision': decision}
                    output['rounds'].append(round_)
                    if decision['parsed'] is None:
                        stop = 'decision_failed'
                        break
                    new = [pair for pair in planned(decision['parsed'], allowed) if pair not in used]
                    if not new:
                        stop = 'model_stopped'
                        break
                    used.extend(new)
                    output['executed_queries'].append(new)
                    known = {s.id for s in pool}
                    fresh = []
                    for group in await fetch(acquire, new):
                        add_sources(pool, group)
                        add_sources(fresh, [s for s in group if s.id not in known])
                    round_['new_source_ids'] = [s.id for s in fresh]
                    # Only newly acquired originals are read; earlier roots merge with them.
                    roots = await reader.reduce([*roots, *await reader.read(fresh)])
        except TimeoutError:
            stop = 'supplement_timeout'
    output['stop_reason'] = stop
    progress('根据入选原文生成回答')
    final = await reader.final(roots)
    by_id = {s.id: s for s in pool}
    cited = final['span_ids']
    selected = list(dict.fromkeys(reader.spans[i]['source'].id for i in cited))
    failed = [n['id'] for n in reader.nodes if n['parsed'] is None]
    output['answer'] = {'answer': final['raw_text'] or '',
        'sources': [by_id[i].model_dump(mode='json') for i in selected],
        'retrieval': {'mode': 'model_planned_queries', 'selected_ids': selected},
        'generation': {'protocol': PROTOCOL,
            'status': 'partial' if final['parsed'] is not None and failed else final['status'],
            'parsed_answer': final['parsed'], 'raw_model_answer': final['raw_text'],
            'answer_modified': False, 'semantic_support_verified': False,
            'citation_kind': 'verbatim_span_selected_by_model_not_verified_support',
            'citation_bindings': {i: [reader.spans[i]['source'].id] for i in cited},
            'cited_spans': {i: {'text': reader.spans[i]['text'], 'span': reader.spans[i]['span']} for i in cited},
            'root_node_id': final['id'], 'failed_nodes': failed,
            'omitted_span_ids': final['omitted_span_ids'], 'missing_need_ids': reader.missing(roots)}}
    output.update(needs=reader.needs, outlines=reader.outlines, page_gates=reader.gates, graph={'nodes': reader.nodes, 'fan_in': 3,
        'spans': {i: {'source_id': s['source'].id, 'span': s['span']} for i, s in reader.spans.items()},
        'source_snapshots': [s.model_dump(mode='json') for s in pool]},
        elapsed_ms=round((monotonic()-started)*1000))
    return output
