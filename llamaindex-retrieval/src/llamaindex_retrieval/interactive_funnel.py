"""Context -> parallel original chunks -> parallel three-child reductions.

Only bounded child outputs enter each reduction. The final call additionally
receives the verbatim chunks whose leaves the model selected, within a
separately configured character budget. Raw text is immutable.
"""
import asyncio
import json
from time import monotonic

from llama_index.core import Document

from . import interactive_sources as structure
from .interactive_sources import add_sources, gather
from .verbatim_chunking import packed_verbatim_nodes

CONTEXT = ('将当前消息整理为可独立理解的完整问题，不回答。结合此前上下文理解指代和更正，'
           '保留未撤回的对象、目标和条件；新话题不沿用旧要求。此前回答只用于理解指代，'
           '不是事实依据。不要猜补信息。输出question。')
LEAF = ('并发阅读中的一份原文。围绕当前完整问题总结这份材料，保留主体、数值、时间、'
        '否定、条件和缺口；总结类问题应保留材料主要内容。不要把网页导航当事实。'
        '输出summary及实际采用的本次原文ID source_ids；无相关内容时两者均为空。')
MERGE = ('合并本组模型总结，保留当前问题所需的事实、条件、分歧和缺口。'
         '这些总结是模型派生内容，不是重新读取的原文，不补充常识。'
         '输出summary和实际采用的直接子节点ID source_ids。')
FINAL = ('依据本组模型总结回答当前完整问题，覆盖用户要求并保留限制和未知。'
         '不补猜事实；失败节点不作为事实。正文用[节点ID]指向依据。'
         '输出answer和实际采用的直接子节点ID source_ids。')
# Opt-in candidate (default off): final also reads selected verbatim chunks.
# Paired TEST24 run 2026-10-19 did not pass its gate (6/24 -> 4/24).
FINAL_ORIGINALS = ('依据入选原文和本组模型总结回答当前完整问题，覆盖用户要求并保留限制和未知。'
                   '事实以原文为准，总结只帮助定位；两者不一致时按原文。不补猜事实；失败节点不作为事实。'
                   '正文用[ID]指向依据，优先引用原文ID。输出answer和实际采用的原文或子节点ID source_ids。')
# Input evidence capacity for verbatim originals in the final call; kept apart
# from the 2K output reservation (ARCHITECTURE_RULES.md rule 10).
FINAL_ORIGINAL_CHARS = 16000
PROTOCOL = 'interactive-funnel-v1'
ORIGINALS_PROTOCOL = 'interactive-funnel-v2'
CHECK = ('根据当前完整问题和已有总结判断是否仍有可核实的缺口。需要补查时输出具体questions；'
         '足够、无法继续核实或继续查询无帮助时返回空数组。不要回答问题，不重复最近查询。')


def messages(instruction, data):
    return [{'role': 'user', 'content': instruction + '\n' +
             json.dumps(data, ensure_ascii=False, separators=(',', ':'))}]


async def generate(model, instruction, data, *, field, allowed=(), limit=1024, final=False):
    properties = {field: {'type': 'array', 'items': {'type': 'string', 'minLength': 1}}
                  if field == 'questions' else {'type': 'string'}}
    if field in {'summary', 'answer'}:
        properties['source_ids'] = {'type': 'array', 'items':
            {'type': 'string', 'enum': list(allowed)} if allowed else {'type': 'string'}}
        if not allowed:
            properties['source_ids']['maxItems'] = 0
    result = await model.complete(messages(instruction, data), structured_schema=structure.obj(properties),
        max_tokens=limit, assistant_prefill='<think></think', temperature=0, seed=11,
        stage='writer' if final else 'reader', evidence_ids=list(allowed))
    event = {'status': result.status, 'raw_text': result.raw_text,
             'raw_sha256': structure.digest(result.raw_text or ''), 'parsed': None, 'trace': result.trace}
    if result.status == 'completed':
        try:
            value = json.loads(result.raw_text, object_pairs_hook=structure.unique_object)
            if not isinstance(value, dict) or set(value) != set(properties):
                raise ValueError('invalid fields')
            if field == 'questions':
                if not isinstance(value[field], list) or any(not isinstance(q, str) or not q.strip() for q in value[field]):
                    raise ValueError('invalid questions')
            elif not isinstance(value[field], str) or (field == 'question' and not value[field].strip()):
                raise ValueError('invalid text')
            if 'source_ids' in properties:
                ids = value['source_ids']
                if (not isinstance(ids, list) or any(not isinstance(i, str) or i not in allowed for i in ids)
                        or len(ids) != len(set(ids))):
                    raise ValueError('invalid child identity')
            event['parsed'] = value
        except (ValueError, TypeError) as error:
            event['validation_error'] = str(error)
    return event


async def contextualize(model, question, parent=None):
    data = {'latest_question': question, 'previous_context': None, 'previous_answer_for_reference_only': None}
    if parent:
        data.update(previous_context=parent['context'], previous_answer_for_reference_only=parent['answer'])
    event = await generate(model, CONTEXT, data, field='question')
    return event['parsed']['question'] if event['parsed'] else None, event


class Funnel:
    def __init__(self, model, question, progress, original_chars=None):
        self.model, self.question, self.progress = model, question, progress
        self.original_chars = original_chars
        self.nodes = []
        self.originals = {}  # leaf ID -> (source, verbatim chunk, parent context)

    def new_node(self, kind, parents=(), **metadata):
        node = {'id': f'N{len(self.nodes)+1}', 'kind': kind, 'parents': [p['id'] for p in parents],
                'level': max((p['level'] for p in parents), default=-1) + 1,
                'status': 'pending', 'raw_text': None, 'parsed': None,
                'original_source_ids': [], 'selected_leaf_ids': [], **metadata}
        self.nodes.append(node)
        return node

    @staticmethod
    def child_input(node):
        value = {'id': node['id'], 'status': node['status'], 'raw_sha256': node.get('raw_sha256')}
        if node['parsed'] is not None:
            value['model_summary_raw'] = node['raw_text']
        return value

    async def read(self, sources, subquestion=None):
        work = []
        for source in sources:
            document = Document(text=source.snippet, id_=source.id,
                                metadata={'kind': source.metadata.get('kind')})
            # Continuous prose overlaps; tables/lists/QA/code keep atomic bounds.
            # Oversized atoms fail the native tokenizer guard explicitly.
            for part in packed_verbatim_nodes(document, 4000, 240, absorb_whitespace=True):
                node = self.new_node('leaf', original_text_input_ids=[source.id],
                    source_span=part.metadata['source_span'],
                    context_spans=part.metadata['context_spans'])
                self.originals[node['id']] = (source, part.text, part.metadata['context_spans'])
                work.append((source, part, node))
        self.progress(f'并发总结资料 · {len(work)} 个原文分块')
        async def leaf(source, part, node):
            event = await generate(self.model, LEAF, {'question': self.question,
                'subquestion': subquestion, 'original': {'id': node['id'], 'title': source.title,
                'text': part.text, 'parent_context': part.metadata['context_spans']}},
                field='summary', allowed=[node['id']], limit=768)
            node.update(event)
            if event['parsed'] is not None:
                if event['parsed']['source_ids']:
                    node.update(original_source_ids=[source.id], selected_leaf_ids=[node['id']])
            elif node['status'] == 'completed':
                node['status'] = 'invalid_output'
            return node
        return await gather([leaf(*row) for row in work])

    def final_originals(self, eligible):
        """Selected verbatim chunks in lineage order; overflow is recorded, never cut."""
        included, omitted, used = [], [], 0
        for leaf_id in dict.fromkeys(i for c in eligible.values() for i in c['selected_leaf_ids']):
            size = len(self.originals[leaf_id][1])
            if omitted or used + size > self.original_chars:
                omitted.append(leaf_id)
            else:
                included.append(leaf_id)
                used += size
        return included, omitted

    def original_input(self, leaf_id):
        source, text, context = self.originals[leaf_id]
        return {'id': leaf_id, 'title': source.title, 'channel': source.source,
                'uri': source.uri, 'parent_context': context, 'text': text}

    async def combine(self, children, *, final=False):
        node = self.new_node('final' if final else 'merge', children, original_text_input_ids=[])
        eligible = {c['id']: c for c in children if c['parsed'] is not None}
        data = {'question': self.question, 'children': [self.child_input(c) for c in children],
                'representation': 'derived_summaries_with_source_lineage'}
        with_originals = final and bool(self.original_chars)
        if with_originals:
            included, omitted = self.final_originals(eligible)
            leaves = {n['id']: n for n in self.nodes if n['id'] in included}
            data.update(originals=[self.original_input(i) for i in included],
                        representation='verbatim_originals_with_derived_summaries')
            node.update(original_leaf_input_ids=included, omitted_original_leaf_ids=omitted,
                        original_text_input_ids=list(dict.fromkeys(
                            self.originals[i][0].id for i in included)),
                        original_chars_budget=self.original_chars)
            eligible.update(leaves)
        instruction = (FINAL_ORIGINALS if with_originals else FINAL) if final else MERGE
        event = await generate(self.model, instruction, data,
            field='answer' if final else 'summary', allowed=list(eligible),
            limit=2048 if final else 1024, final=final)
        node.update(event)
        if event['parsed'] is not None:
            selected = event['parsed']['source_ids']
            node['original_source_ids'] = list(dict.fromkeys(
                sid for i in selected for sid in eligible[i]['original_source_ids']))
            node['selected_leaf_ids'] = list(dict.fromkeys(
                lid for i in selected for lid in eligible[i]['selected_leaf_ids']))
        elif node['status'] == 'completed':
            node['status'] = 'invalid_output'
        return node

    async def reduce(self, nodes):
        current = list(nodes)
        while len(current) > 3:
            self.progress(f'并发归并总结 · {len(current)} 个节点')
            groups = [current[i:i+3] for i in range(0, len(current), 3)]
            async def group(children):
                return children[0] if len(children) == 1 else await self.combine(children)
            current = await gather([group(children) for children in groups])
        return current

    def response(self, root, sources):
        by_id = {node['id']: node for node in self.nodes}
        selected = set(root['original_source_ids'])
        failed = [node['id'] for node in self.nodes if node['parsed'] is None]
        return {'answer': root['raw_text'] or '',
            'sources': [s.model_dump(mode='json') for s in sources if s.id in selected],
            'retrieval': {'mode': 'parallel_funnel', 'selected_ids': sorted(selected)},
            'generation': {'protocol': ORIGINALS_PROTOCOL if self.original_chars else PROTOCOL,
                'status': 'partial' if root['parsed'] is not None and failed else root['status'],
                'parsed_answer': root['parsed'], 'raw_model_answer': root['raw_text'],
                'answer_modified': False, 'semantic_support_verified': False,
                'citation_kind': 'original_or_inherited_lineage_not_verified_support',
                'citation_bindings': {cid: by_id[cid]['original_source_ids']
                    for cid in (root['parsed'] or {}).get('source_ids', [])},
                'root_node_id': root['id'], 'failed_nodes': failed,
                'original_leaf_input_ids': root.get('original_leaf_input_ids', []),
                'omitted_original_leaf_ids': root.get('omitted_original_leaf_ids', [])}}


async def answer(model, question, initial, *, acquire, scope, progress, supplementary=True,
                 supplement_timeout=300, original_chars=None):
    started = monotonic()
    funnel = Funnel(model, question, progress, original_chars)
    pool = []
    add_sources(pool, initial)
    nodes = await funnel.reduce(await funnel.read(pool))
    output = {'rounds': [],
              'protocol': ORIGINALS_PROTOCOL if original_chars else PROTOCOL, 'scope': scope}
    if supplementary:
        stop, latest_queries = 'pending', []
        try:
            async with asyncio.timeout(supplement_timeout):
                while True:
                    progress(f'根据漏斗总结检查补查 · 第 {len(output["rounds"])+1} 轮')
                    decision = await generate(model, CHECK, {'question': question,
                        'summaries': [funnel.child_input(n) for n in nodes],
                        'latest_queries': latest_queries, 'available_channels': scope}, field='questions', limit=2048)
                    round_ = {'number': len(output['rounds'])+1, 'decision': decision}
                    output['rounds'].append(round_)
                    if decision['parsed'] is None:
                        stop = 'decision_failed'
                        break
                    queries = decision['parsed']['questions']
                    if not queries:
                        stop = 'model_stopped'
                        break
                    round_['questions'] = queries
                    acquired = await gather([acquire(q) for q in queries])
                    for group in acquired:
                        add_sources(pool, group)
                    readings = await gather([funnel.read(pool, q) for q in queries])
                    # Commit a complete round; all its model outputs remain auditable.
                    nodes = await funnel.reduce([*nodes, *(n for group in readings for n in group)])
                    latest_queries = queries
        except TimeoutError:
            stop = 'supplement_timeout'
        output['stop_reason'] = stop
    progress('根据漏斗总结生成回答')
    final = await funnel.combine(nodes, final=True)
    output['answer'] = funnel.response(final, pool)
    if supplementary and stop != 'model_stopped' and output['answer']['generation']['status'] == 'completed':
        output['answer']['generation']['status'] = 'partial'
    output.update(funnel={'nodes': funnel.nodes, 'fan_in': 3,
                         'source_snapshots': [s.model_dump(mode='json') for s in pool],
                         'original_text_inputs': ['leaf', 'final'] if original_chars else ['leaf']},
                  elapsed_ms=round((monotonic()-started)*1000), subquestion_count_limit=None)
    return output
