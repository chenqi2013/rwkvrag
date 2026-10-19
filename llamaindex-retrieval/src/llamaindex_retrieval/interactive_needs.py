"""Plan -> per-channel acquisition -> span scoring -> per-need top-k -> per-need answers -> composition.

Candidate pipeline (opt-in, ``pipeline: "needs"``). Every model call stays small
(see interactive_spans budgets). The model scores spans per information need,
decides between strong candidates, answers each need from at most NEED_SPANS
verbatim spans and composes the final reply from those short answers. Code only
splits, ranks by the model's scores, caps counts, binds IDs and validates
structure. Raw outputs are immutable; a failed call is recorded, never rewritten.
"""
import asyncio
from statistics import median
from time import monotonic

from . import kb_navigation
from .interactive_funnel import CONTEXT
from .interactive_sources import CATALOG, add_sources, gather
from .interactive_spans import (CHANNEL_FIELDS, FINAL_SPAN_CHARS, OUTLINE, PAGE, PLAN, REQUERY, Reader, channels, fetch,
                                generate, planned, strings)

PROTOCOL = 'interactive-needs-v1'
SCORE = ('并发阅读中的一段原文，已切成带ID的片段。对每个信息项，判断哪些片段能直接回答它并打分：'
         '3=片段明确给出该信息项的答案，2=片段包含答案的关键部分，1=只是相关。只输出2分以上的片段，'
         '片段讲的对象必须是问题中的对象；没有就返回空数组。')
PICK = ('以下片段都被判断为可能回答这个信息项。选出最直接、最可靠地回答它的片段（最多三个），'
        '优先用户指定的来源和官方资料，对象或版本不符的不选。')
ANSWER = ('只依据给出的逐字原文片段回答这一个信息项。片段明确给出答案时status为found，answer写简短答案并保留条件，'
          'span_ids列出依据的片段；片段不能回答时status为not_found，answer说明缺什么，span_ids为空。不凭常识作答。')
COMPOSE = ('根据已核对的回答，回答用户的完整问题。只使用answers中的事实，逐项覆盖，不添加其他事实；'
           'not_found中的信息项没有找到依据，只说明未找到，不要给出任何答案或猜测。'
           '正文在每个事实后用[片段ID]标注依据（ID来自对应回答）。source_ids列出实际采用的片段ID。')
NEED_SPANS = 3                # Evidence spans per need answer.
PICK_SPANS = 8                # Candidates offered to one comparison call.
PICK_CHARS = 4000             # Span text per comparison call (~2.5k tokens).
SCORES = ['2', '3']


def score_input(question, needs, source, parent_context, spans):
    """Canonical runtime payload, also usable by local component-data tooling."""
    return {'question': question, 'needs': needs,
            'source': {'title': source.title, 'channel': source.source, 'uri': source.uri,
                       'parent_context': parent_context}, 'spans': spans}


def answer_input(question, need, spans):
    return {'question': question, 'need': need, 'spans': spans}


def score_schema(span_ids, need_ids):
    return {'scores': {'type': 'array', 'maxItems': len(span_ids) * len(need_ids), 'items': {
        'type': 'object', 'additionalProperties': False, 'required': ['span_id', 'need_id', 'score'],
        'properties': {'span_id': {'type': 'string', 'enum': list(span_ids)},
                       'need_id': {'type': 'string', 'enum': list(need_ids)},
                       'score': {'type': 'string', 'enum': SCORES}}}}}


def answer_schema(span_ids):
    return {'status': {'type': 'string', 'enum': ['found', 'not_found']}, 'answer': {'type': 'string'},
            'span_ids': {**strings(span_ids), 'maxItems': len(span_ids)}}


def stage_of(event):
    content = event['messages'][0]['content']
    for name, prefix in (('context', CONTEXT), ('plan', PLAN), ('requery', REQUERY), ('score', SCORE), ('pick', PICK),
                         ('answer', ANSWER), ('compose', COMPOSE),
                         ('page_gate', PAGE), ('outline_navigation', OUTLINE),
                         ('catalog_navigation', CATALOG),
                         ('navigation_category', kb_navigation.CATEGORY_PICK),
                         ('navigation', kb_navigation.CARD_PICK[:20]),
                         ('navigation', kb_navigation.FINAL_PICK[:20])):
        if content.startswith(prefix):
            return name
    return 'other'


def telemetry(events):
    """Per-stage input tokens and non-completed counts from the observed model calls."""
    stages = {}
    for event in events:
        trace = event.get('trace') or {}
        tokens = (trace.get('budget') or {}).get('input_tokens')
        row = stages.setdefault(stage_of(event), {'calls': 0, 'not_completed': 0, 'tokens': [],
                                                 'generation_tokens': [], 'blocked_tokens': []})
        row['calls'] += 1
        row['not_completed'] += event.get('status') != 'completed'
        if tokens is not None:
            row['tokens'].append(tokens)
            row['generation_tokens' if trace.get('completion_attempted') else 'blocked_tokens'].append(tokens)
    return {k: {'calls': v['calls'], 'not_completed': v['not_completed'],
                'median_tokens': median(v['tokens']) if v['tokens'] else None,
                'max_tokens': max(v['tokens'], default=None),
                'max_generation_tokens': max(v['generation_tokens'], default=None),
                'max_not_sent_tokens': max(v['blocked_tokens'], default=None)} for k, v in stages.items()}


class NeedReader(Reader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidates = {i: {} for i in self.needs}  # need -> span -> best score
        self.order = []                                 # Span reading order, for stable ties.

    async def score(self, sources):
        """One SCORE call per reading window; scores are recorded per need and span."""
        if self.page_gate:
            web = [s for s in sources if s.source == 'web']
            if web:
                self.progress(f'判断网页是否相关 · {len(web)} 份')
                keep = dict(zip([s.id for s in web], await gather([self.gate(s) for s in web])))
                sources = [s for s in sources if keep.get(s.id, True)]
        ranges = {}
        long = [s for s in sources if self.outline_chars and len(s.snippet) > self.outline_chars]
        for source, picked in zip(long, await gather([self.navigate(s) for s in long])):
            ranges[source.id] = picked
        work = [(s, w, ids) for s in sources
                for a, b in ranges.get(s.id, [(0, None)]) for w, ids in self.windows(s, a, b)]
        self.progress(f'并发阅读原文 · {len(work)} 个分块')

        async def leaf(source, window, ids):
            needs = self.need_input()
            self.order += ids
            node = self.new_node('leaf', original_text_input_ids=[source.id], offered_span_ids=ids,
                                 source_span=window.metadata['source_span'])
            event = await generate(self.model, SCORE, score_input(self.question, needs, source,
                window.metadata['context_spans'], [{'id': i, 'text': self.spans[i]['text']} for i in ids]),
                score_schema(ids, [n['id'] for n in needs]), limit=384, evidence=ids)
            node.update(event)
            rows = (event['parsed'] or {}).get('scores', [])
            if event['parsed'] is not None and any(
                    not isinstance(r, dict) or r.get('span_id') not in ids or r.get('need_id') not in {n['id'] for n in needs}
                    or r.get('score') not in SCORES for r in rows):
                node.update(parsed=None, validation_error='invalid scores')
                rows = []
            elif event['parsed'] is None and node['status'] == 'completed':
                node['status'] = 'invalid_output'
            for r in rows:
                best = self.candidates[r['need_id']]
                best[r['span_id']] = max(best.get(r['span_id'], 0), int(r['score']))
            node['span_ids'] = list(dict.fromkeys(r['span_id'] for r in rows))
            return node
        return await gather([leaf(*row) for row in work])

    def ranked(self, need):
        position = {s: i for i, s in enumerate(self.order)}
        return sorted(self.candidates[need], key=lambda s: (-self.candidates[need][s], position.get(s, 0)))

    async def evidence(self, need):
        """At most NEED_SPANS spans: by model score, or a comparison call when strong ones exceed it."""
        ranked = self.ranked(need)
        top = [s for s in ranked if self.candidates[need][s] == 3]
        if len(top) <= NEED_SPANS:
            return ranked[:NEED_SPANS], None
        offered, used = [], 0
        for span_id in top[:PICK_SPANS]:
            size = len(self.spans[span_id]['text'])
            if offered and used + size > PICK_CHARS:
                break
            offered.append(span_id)
            used += size
        node = self.new_node('pick', need_id=need, offered_span_ids=offered)
        event = await generate(self.model, PICK, {'question': self.question,
            'need': {'id': need, 'text': self.needs[need]}, 'spans': [self.span_input(i) for i in offered]},
            {'span_ids': {**strings(offered), 'maxItems': NEED_SPANS}}, limit=128, evidence=offered)
        node.update(event)
        if event['parsed'] is not None and event['parsed']['span_ids']:
            node['span_ids'] = event['parsed']['span_ids']
            return node['span_ids'], node
        if node['status'] == 'completed' and event['parsed'] is None:
            node['status'] = 'invalid_output'
        # A failed or empty comparison falls back to the model's own scores; recorded on the node.
        return offered[:NEED_SPANS], node

    async def answer_need(self, need):
        spans, pick = await self.evidence(need)
        node = self.new_node('answer', need_id=need, offered_span_ids=spans,
                             pick_node_id=pick['id'] if pick else None)
        if not spans:
            node.update(status='no_evidence', answer_status='not_found')
            return node
        event = await generate(self.model, ANSWER, answer_input(self.question,
            {'id': need, 'text': self.needs[need]}, [self.span_input(i) for i in spans]),
            answer_schema(spans), limit=384, stage='writer', evidence=spans)
        node.update(event)
        parsed = event['parsed']
        if parsed is not None and (parsed['status'] not in {'found', 'not_found'}
                                   or (parsed['status'] == 'found') != bool(parsed['span_ids'])):
            node.update(parsed=None, validation_error='found answers must cite spans; not_found must not')
        if node['parsed'] is not None:
            node.update(answer_status=parsed['status'], span_ids=parsed['span_ids'])
        else:
            node['answer_status'] = 'failed'
            if node['status'] == 'completed':
                node['status'] = 'invalid_output'
        return node

    def unresolved(self, answers):
        return [a['need_id'] for a in answers if a['answer_status'] != 'found']

    async def compose(self, answers):
        cited = list(dict.fromkeys(s for a in answers for s in a['span_ids']))
        node = self.new_node('final', [a for a in answers], offered_span_ids=cited,
                             original_text_input_ids=list(dict.fromkeys(self.spans[i]['source'].id for i in cited)))
        # Only structurally validated found answers carry facts; missing needs carry names.
        rows = [{'need_id': a['need_id'], 'need': self.needs[a['need_id']], 'answer': a['parsed']['answer'],
                 'span_ids': a['span_ids'],
                 'sources': [{'id': s, 'source': self.spans[s]['source'].title,
                              'uri': self.spans[s]['source'].uri} for s in a['span_ids']]}
                for a in answers if a['answer_status'] == 'found']
        if not rows:
            # Empty-evidence policy: no composition call, so no answer can be written without evidence.
            node.update(status='no_evidence', raw_text=None, parsed=None)
            return node
        missing = [self.needs[a['need_id']] for a in answers if a['answer_status'] != 'found']
        event = await generate(self.model, COMPOSE, {'question': self.question, 'answers': rows,
                                                     'not_found': missing},
            {'answer': {'type': 'string'}, 'source_ids': {**strings(cited), 'maxItems': len(cited)}},
            limit=1024, stage='writer', evidence=cited)
        node.update(event)
        if event['parsed'] is not None:
            node['span_ids'] = event['parsed']['source_ids']
        elif node['status'] == 'completed':
            node['status'] = 'invalid_output'
        return node


async def answer(model, question, *, acquire, scope, progress, supplementary=True, max_rounds=2,
                 supplement_timeout=300, page_gate=True, outline_chars=None, final_chars=FINAL_SPAN_CHARS):
    started = monotonic()
    allowed = channels(scope)
    fields = {CHANNEL_FIELDS[c]: strings() for c in allowed}
    progress('制定检索')
    plan = await generate(model, PLAN, {'question': question, 'available_channels': scope},
                          {'needs': strings(), **fields}, limit=1024, stage='planner')
    output = {'protocol': PROTOCOL, 'scope': scope, 'plan': plan, 'rounds': []}
    needs = list(dict.fromkeys(plan['parsed']['needs'])) if plan['parsed'] else []
    needs = needs or [question]
    queries = (planned(plan['parsed'], allowed) if plan['parsed'] else []) or [(c, question) for c in allowed]
    output['executed_queries'] = [queries]
    reader = NeedReader(model, question, needs, progress, final_chars, outline_chars, page_gate)
    pool = []
    for group in await fetch(acquire, queries):
        add_sources(pool, group)
    await reader.score(pool)
    progress('逐项核对回答')
    answers = await gather([reader.answer_need(n) for n in reader.needs])
    stop = 'not_requested'
    if supplementary:
        used = list(queries)
        try:
            async with asyncio.timeout(supplement_timeout):
                while True:
                    missing = reader.unresolved(answers)
                    if not missing:
                        stop = 'needs_covered'
                        break
                    if len(output['rounds']) >= max_rounds:
                        stop = 'round_limit'
                        break
                    progress(f'按未找到的信息项补查 · 第 {len(output["rounds"])+1} 轮')
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
                    # Only newly acquired originals are read; only unresolved needs are re-answered.
                    await reader.score(fresh)
                    redo = dict(zip(missing, await gather([reader.answer_need(n) for n in missing])))
                    answers = [redo.get(a['need_id'], a) for a in answers]
        except TimeoutError:
            stop = 'supplement_timeout'
    output['stop_reason'] = stop
    progress('根据逐项回答生成最终回答')
    final = await reader.compose(answers)
    by_id = {s.id: s for s in pool}
    cited = final['span_ids']
    selected = list(dict.fromkeys(reader.spans[i]['source'].id for i in cited))
    failed = [n['id'] for n in reader.nodes if n['parsed'] is None and n['status'] != 'no_evidence'
              and n['kind'] != 'final']
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
            'need_answers': {a['need_id']: a['answer_status'] for a in answers},
            'missing_need_ids': reader.unresolved(answers)}}
    events = getattr(model, 'events', None)
    output.update(needs=reader.needs, outlines=reader.outlines, page_gates=reader.gates,
                  telemetry=telemetry(events) if events is not None else None,
                  graph={'nodes': reader.nodes,
                         'candidates': reader.candidates,
                         'spans': {i: {'source_id': s['source'].id, 'span': s['span']} for i, s in reader.spans.items()},
                         'source_snapshots': [s.model_dump(mode='json') for s in pool]},
                  elapsed_ms=round((monotonic()-started)*1000))
    return output
