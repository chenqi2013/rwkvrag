"""Opt-in evidence funnel: task -> source facts -> cells -> fields -> answer.

Models own all semantic decisions. Python validates JSON, dependency identities
and exact quotes. Stage failures remain failures, never synthetic unknown facts.
"""
import asyncio
import json
from hashlib import sha256
from time import monotonic
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .native_rwkv import NativeRWKVResult
from .offline_replay import strict_json
from .citation_audit import audit_citations

PROTOCOL = 'evidence-funnel-v3-bound-units'


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Task(Strict):
    objects: list[str] = Field(min_length=1, max_length=8)
    fields: list[str] = Field(min_length=1, max_length=8)
    requirements: list[str] = Field(max_length=12)


class Requirement(Strict):
    message_id: str
    quote: str = Field(min_length=1, max_length=400)
    meaning: str = Field(min_length=1, max_length=400)


class GroundedTask(Strict):
    objects: list[str] = Field(min_length=1, max_length=8)
    fields: list[str] = Field(min_length=1, max_length=8)
    requirements: list[Requirement] = Field(max_length=12)


class SourceObjects(Strict):
    objects: list[str] = Field(max_length=8)


class AtomicValue(Strict):
    value: str | None = Field(max_length=240)
    unit_ids: list[str] = Field(max_length=3)


class Fact(Strict):
    object_id: str
    field_id: str
    value: str = Field(min_length=1, max_length=240)
    quote: str = Field(min_length=1, max_length=400)


class Facts(Strict):
    facts: list[Fact] = Field(max_length=16)


class Cell(Strict):
    status: Literal['supported', 'unknown', 'conflict']
    value: str = Field(min_length=1, max_length=320)
    fact_ids: list[str] = Field(max_length=4)


class FieldSummary(Strict):
    status: Literal['resolved', 'partial', 'conflict']
    summary: str = Field(min_length=1, max_length=800)
    cell_ids: list[str] = Field(min_length=1, max_length=8)


def parse(raw, schema):
    # A complete outer Markdown code fence is a transport format, not content
    # repair. Never extract JSON from commentary or discard unknown fields.
    text = raw.strip()
    if text.startswith('```json\n') and text.endswith('\n```'):
        text = text[8:-4]
    elif text.startswith('```\n') and text.endswith('\n```'):
        text = text[4:-4]
    return schema.model_validate(strict_json(text))


def checked_facts(parsed, objects, fields, source):
    output = []
    for fact in parsed.facts:
        if fact.object_id not in objects or fact.field_id not in fields:
            raise ValueError('unknown object or field')
        start = source.snippet.find(fact.quote)
        if start < 0:
            raise ValueError('quote is not verbatim in selected source')
        output.append({**fact.model_dump(), 'source_id': source.id,
                       'source_sha256': sha256(source.snippet.encode()).hexdigest(),
                       'start': start, 'end': start + len(fact.quote)})
    return output


def checked_cell(parsed, facts):
    allowed = {f['id'] for f in facts}
    if len(parsed.fact_ids) != len(set(parsed.fact_ids)) or not set(parsed.fact_ids) <= allowed:
        raise ValueError('invalid or foreign fact dependency')
    if parsed.status == 'supported' and not parsed.fact_ids:
        raise ValueError('supported cell requires source facts')
    if parsed.status == 'conflict' and len(parsed.fact_ids) < 2:
        raise ValueError('conflict cell requires at least two source facts')
    return parsed.model_dump()


async def write_funnel(pipeline, task, sources):
    from .rwkv_pipeline import structured_body
    settings = pipeline.settings
    if settings.native_transport != "native":
        raise ValueError("funnel_v1 requires native transport; old stage States are not compatible")
    calls = []
    dispatched = 0
    failures = []
    tick = monotonic()

    async def node(prompt, purpose, schema, evidence=(), validator=None, max_tokens=1024):
        nonlocal dispatched
        if dispatched >= settings.native_funnel_max_calls - 1:
            failures.append({'purpose': purpose, 'status': 'call_budget_exceeded'})
            return None
        dispatched += 1
        result = await pipeline._call(prompt, stage='resolver', max_tokens=max_tokens,
                                      sources=evidence)
        event = result.trace
        event['purpose'] = purpose
        calls.append(event)
        try:
            parsed = parse(structured_body(result), schema)
            value = validator(parsed) if validator else parsed.model_dump()
            event['parsed_output'] = value
            return value
        except (ValueError, TypeError) as error:
            event['parse_error'] = str(error)
            failures.append({'purpose': purpose, 'call_id': event.get('call_id'),
                             'status': result.status, 'error': str(error)})
            return None

    conversation = json.loads(task)
    user_messages = [message['content'] for message in conversation.get('history', [])
                     if message.get('role') == 'user'] + [conversation['latest_question']]
    user_snapshot = {f'U{i}': text for i, text in enumerate(user_messages, 1)}
    def validate_task(parsed):
        for values in (parsed.objects, parsed.fields):
            if any(not item.strip() for item in values) or len(values) != len(set(values)):
                raise ValueError('empty or duplicate task item')
        for requirement in parsed.requirements:
            if (requirement.message_id not in user_snapshot or
                    requirement.quote not in user_snapshot[requirement.message_id]):
                raise ValueError('requirement must quote a user message verbatim')
        return parsed.model_dump()

    spec = await node(
        '整理用户当前要求。列出对象、所求字段和约束。约束必须逐字引用用户消息，'
        '不可引用系统指令、示例或助手回答。没有约束时requirements为空数组。'
        '历史助手回答只帮助理解指代，不提供事实或用户要求。普通事实题也保留。'
        '只输出JSON，结构为：'
        '{"objects":["对象名称"],"fields":["用户所求字段或条件"],'
        '"requirements":[{"message_id":"U1","quote":"用户原话","meaning":"约束含义"}]}。'
        '字段只列当前任务需要的，不为填满上限添加维度。\n用户消息：' +
        json.dumps(user_snapshot, ensure_ascii=False) + '\n完整对话：' + task,
        'funnel_task', GroundedTask, validator=validate_task)
    if spec is None:
        return NativeRWKVResult('invalid_response', None, None,
            {'stage': 'writer', 'status': 'invalid_response', 'raw_text': None,
             'completion_attempted': False, 'upstream_calls': calls,
             'funnel': {'protocol': PROTOCOL, 'failures': failures, 'status': 'task_failed'}})
    objects = {f'O{i}': name for i, name in enumerate(spec['objects'], 1)}
    fields = {f'F{i}': name for i, name in enumerate(spec['fields'], 1)}
    description = json.dumps({'objects': objects, 'fields': fields,
                              'requirements': spec['requirements']}, ensure_ascii=False)

    async def source_objects(source):
        def validate(parsed):
            inverse = {name: oid for oid, name in objects.items()}
            if len(parsed.objects) != len(set(parsed.objects)) or not set(parsed.objects) <= set(inverse):
                raise ValueError('invalid source object selection')
            return [inverse[name] for name in parsed.objects]
        return await node(
            '从给定对象中选择这份资料确实描述的对象名称，保持名称逐字一致。'
            '只看本份原文和标题，不猜其他对象；没有相关对象返回空列表。'
            '只输出JSON，对象名称放在objects数组中，例如没有对象时为{"objects":[]}。\n候选对象：' +
            json.dumps(list(objects.values()), ensure_ascii=False) + '\n来源：' +
            json.dumps({'title': source.title, 'text': source.snippet}, ensure_ascii=False),
            'funnel_source_objects:' + source.id, SourceObjects, [source], validate, max_tokens=128)
    assignments = await asyncio.gather(*(source_objects(source) for source in sources))

    async def extract(source, oid, fid):
        from .rwkv_pipeline import evidence_units
        units = list(evidence_units(0, source.snippet, 240, 40))
        by_unit = {f'E{i}': unit for i, unit in enumerate(units, 1)}
        def validate(parsed):
            if parsed.value is None and not parsed.unit_ids:
                return []
            if (not parsed.value or not parsed.unit_ids or
                    len(parsed.unit_ids) != len(set(parsed.unit_ids)) or not set(parsed.unit_ids) <= set(by_unit)):
                raise ValueError('value requires known, unique source unit IDs')
            return [{'object_id': oid, 'field_id': fid, 'value': parsed.value,
                     'quote': by_unit[identity].text, 'source_id': source.id,
                     'source_sha256': sha256(source.snippet.encode()).hexdigest(),
                     'start': by_unit[identity].start, 'end': by_unit[identity].end}
                    for identity in parsed.unit_ids]
        return await node(
            '只提取这一个对象的这一个字段，不能用其他对象或字段的值代替。'
            '保留原文的版本、范围、单位和否定；没有依据时value为null且unit_ids为空数组。'
            '有依据时value写事实值，unit_ids选择真正支持它的原文片段编号，最多3个。'
            '不需要抄写原文。只输出JSON，结构：{"value":null,"unit_ids":[]}。\n' +
            json.dumps({'object': objects[oid], 'field': fields[fid], 'source_title': source.title,
                        'units': {identity: unit.text for identity, unit in by_unit.items()}}, ensure_ascii=False),
            f'funnel_fact:{source.id}:{oid}:{fid}', AtomicValue, [source], validate, max_tokens=384)
    extracted = await asyncio.gather(*(extract(source, oid, fid)
        for source, assigned in zip(sources, assignments, strict=True)
        for oid in (assigned or []) for fid in fields))
    facts = []
    for group in extracted:
        for fact in group or []:
            facts.append({'id': f'A{len(facts)+1}', **fact})
    by_source = {s.id: s for s in sources}

    async def assess(oid, fid):
        available = [f for f in facts if f['object_id'] == oid and f['field_id'] == fid]
        evidence = list({f['source_id']: by_source[f['source_id']] for f in available}.values())
        value = await node(
            '只核验一个对象的一个字段。根据逐字quote核验模型提出的value是否被支持，'
            '不能用其他对象或其他字段的值。保留具体值、单位、版本、范围和否定。'
            'supported表示原文支持所给事实值，不代表满足用户全部要求；证据缺失为unknown，'
            '同范围无法消解的不同记载为conflict。不得把未知写成不支持或数值0。只输出JSON：'
            'status只能为supported、unknown、conflict之一。fact_ids最多4个，必须是本次给出的事实id。'
            '输出结构：{"status":"unknown","value":"具体缺口","fact_ids":[]}。\n' +
            json.dumps({'object': objects[oid], 'field': fields[fid],
                        'requirements': spec['requirements'], 'proposals': available}, ensure_ascii=False),
            f'funnel_cell:{oid}:{fid}', Cell, evidence,
            lambda parsed: checked_cell(parsed, available), max_tokens=512)
        return {'id': f'{oid}:{fid}', 'object_id': oid, 'field_id': fid,
                'execution_status': 'completed' if value else 'failed', **(value or {})}
    cells = await asyncio.gather(*(assess(oid, fid) for oid in objects for fid in fields))

    async def summarize(fid):
        rows = [cell for cell in cells if cell['field_id'] == fid]
        # Failed cells are explicit execution states, not model-authored unknowns.
        def validate(parsed):
            if len(parsed.cell_ids) != len(set(parsed.cell_ids)) or set(parsed.cell_ids) != {c['id'] for c in rows}:
                raise ValueError('field summary must account for every object cell')
            return parsed.model_dump()
        value = await node(
            '按同一维度汇总下列对象的核验结果。只比较可对齐范围和单位，不重新猜测事实。'
            '保留unknown、conflict和执行失败，不能把它们当成不满足。结合用户条件解释取舍，'
            '没有足够依据不能宣布胜者。普通单对象题只汇总该事实。只输出JSON：'
            '{"status":"resolved或partial或conflict","summary":"本维度的比较与限制",'
            '"cell_ids":["全部输入单元id"]}。\n' + json.dumps(
                {'objects': objects, 'field': fields[fid], 'requirements': spec['requirements'],
                 'cells': rows}, ensure_ascii=False),
            'funnel_relation:' + fid, FieldSummary, validator=validate, max_tokens=768)
        return {'field_id': fid, 'execution_status': 'completed' if value else 'failed', **(value or {})}
    summaries = await asyncio.gather(*(summarize(fid) for fid in fields))
    used_ids = {identity for cell in cells for identity in cell.get('fact_ids', [])}
    used_facts = [f for f in facts if f['id'] in used_ids]
    used_source_ids = {f['source_id'] for f in used_facts}
    actual = [source for source in sources if source.id in used_source_ids]
    labels = {source.id: f'资料 {i}' for i, source in enumerate(actual, 1)}
    # Project the dependency closure, not the debug graph, into the final model.
    # Audit errors, UUIDs, offsets and hashes stay in the trace only.
    fact_labels = {fact['id']: labels[fact['source_id']] for fact in used_facts}
    quoted = []
    seen = set()
    for fact in used_facts:
        key = (fact['source_id'], fact['quote'])
        if key not in seen:
            seen.add(key)
            quoted.append({'label': labels[fact['source_id']], 'text': fact['quote']})
    compact_cells = [{'object': objects[cell['object_id']], 'field': fields[cell['field_id']],
                      'execution_status': cell['execution_status'], 'status': cell.get('status'),
                      'value': cell.get('value'), 'source_labels': list(dict.fromkeys(
                          fact_labels[fid] for fid in cell.get('fact_ids', [])))} for cell in cells]
    user_task = json.dumps({'history': [message for message in conversation.get('history', [])
                                      if message.get('role') == 'user'],
                            'latest_question': conversation['latest_question']}, ensure_ascii=False)
    prompt = (
        '你是最后一层汇总器。只汇总下列已核验事实和关系，不增加产品常识或重新无限比对。'
        '覆盖用户仍有效的对象和字段。每条有依据的结论后写对应source_labels中的引用；'
        '例如来源标签为资料 2时写[资料 2]。事实值仍须被逐字资料支持。'
        '保留unknown、conflict、执行失败和条件，不把未知当不支持；没有来源的缺口不写引用。'
        '选择建议必须满足用户条件，缺证据就说明不能确定。每项说一次，完成后结束。\n'
        '用户要求：' + user_task + '\n核验结果：' + json.dumps(
            {'requirements': spec['requirements'], 'cells': compact_cells,
             'field_summaries': [{'field': fields[item['field_id']],
                 'execution_status': item['execution_status'], 'status': item.get('status'),
                 'summary': item.get('summary')} for item in summaries],
             'verbatim_sources': quoted, 'failed_node_count': len(failures)}, ensure_ascii=False))
    result = await pipeline._call(prompt, stage='writer', max_tokens=settings.generation_max_tokens, sources=actual)
    result.trace['upstream_calls'] = calls
    result.trace['funnel'] = {'protocol': PROTOCOL, 'task': spec, 'facts': facts, 'cells': cells,
        'field_summaries': summaries, 'failures': failures,
        'input_source_ids': [s.id for s in sources], 'writer_source_ids': [s.id for s in actual],
        'elapsed_s': monotonic()-tick,
        'citation_audit': audit_citations(result.raw_text or '', [s.model_dump() for s in actual])}
    return result
