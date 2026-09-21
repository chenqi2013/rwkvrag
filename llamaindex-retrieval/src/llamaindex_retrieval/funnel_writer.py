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

PROTOCOL = 'evidence-funnel-v1'


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Task(Strict):
    objects: list[str] = Field(min_length=1, max_length=8)
    fields: list[str] = Field(min_length=1, max_length=8)
    requirements: list[str] = Field(max_length=12)


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
    fact_ids: list[str] = Field(max_length=16)


class FieldSummary(Strict):
    status: Literal['resolved', 'partial', 'conflict']
    summary: str = Field(min_length=1, max_length=800)
    cell_ids: list[str] = Field(min_length=1, max_length=8)


def parse(raw, schema):
    return schema.model_validate(strict_json(raw))


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

    def validate_task(parsed):
        for values in (parsed.objects, parsed.fields, parsed.requirements):
            if any(not item.strip() for item in values) or len(values) != len(set(values)):
                raise ValueError('empty or duplicate task item')
        return parsed.model_dump()

    spec = await node(
        '整理用户仍有效的对象、要回答的维度和约束。历史助手的话只能帮助理解指代，不能当作事实或用户要求。'
        '保留用户未撤回的对象和条件；普通事实问题也照常列对象和所求字段。只输出JSON：'
        '{"objects":["对象名"],"fields":["完整所求字段或条件"],"requirements":["用户限制"]}。'
        '对象和字段各最多8项，不添加示例里的文字。任务：' + task,
        'funnel_task', Task, validator=validate_task)
    if spec is None:
        return NativeRWKVResult('invalid_response', None, None,
            {'stage': 'writer', 'status': 'invalid_response', 'raw_text': None,
             'completion_attempted': False, 'upstream_calls': calls,
             'funnel': {'protocol': PROTOCOL, 'failures': failures, 'status': 'task_failed'}})
    objects = {f'O{i}': name for i, name in enumerate(spec['objects'], 1)}
    fields = {f'F{i}': name for i, name in enumerate(spec['fields'], 1)}
    description = json.dumps({'objects': objects, 'fields': fields,
                              'requirements': spec['requirements']}, ensure_ascii=False)

    async def extract(source):
        return await node(
            '从这一份原文提取能回答指定对象和字段的原子事实。不要比较或推荐，不补常识，'
            '不从用户要求推导产品能力。每条quote必须逐字摘自原文、最多400字；value保留版本、日期、'
            '单位、否定和适用范围。没有相关事实输出空facts。只输出JSON：'
            '{"facts":[{"object_id":"O1","field_id":"F1","value":"事实值及范围",'
            '"quote":"逐字原文"}]}。编号只能来自任务。\n任务：' + description +
            '\n来源标题：' + json.dumps(source.title, ensure_ascii=False) +
            '\n原文：' + json.dumps(source.snippet, ensure_ascii=False),
            'funnel_fact:' + source.id, Facts, [source],
            lambda parsed: checked_facts(parsed, objects, fields, source))
    extracted = await asyncio.gather(*(extract(source) for source in sources))
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
            '{"status":"supported或unknown或conflict","value":"核验后的事实值或具体缺口",'
            '"fact_ids":["实际采用的A编号"]}。\n' +
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
    quoted = [{**fact, 'label': labels[fact['source_id']]} for fact in used_facts]
    conversation = json.loads(task)
    user_task = json.dumps({'history': [message for message in conversation.get('history', [])
                                      if message.get('role') == 'user'],
                            'latest_question': conversation['latest_question']}, ensure_ascii=False)
    prompt = (
        '你是最后一层汇总器。上游已经按对象核验事实、按维度完成比较；请汇总，不再补产品常识或反复比对。'
        '模型小结不是来源，只有quote是逐字证据。按用户要求覆盖全部对象和字段，保留条件、未知、冲突和失败。'
        '每条事实后引用对应quote的label，例如[资料 2]，没有来源的缺口不引用。'
        '选择题给出有证据支持的建议和仍需核实的条件；证据不足不要强行选。每项说一次，完成后结束。\n' +
        '用户要求（助手历史仅在任务整理层用于指代）：' + user_task + '\n漏斗结果：' + json.dumps(
            {'objects': objects, 'fields': fields, 'requirements': spec['requirements'],
             'cells': cells, 'field_summaries': summaries, 'source_facts': quoted,
             'stage_failures': failures}, ensure_ascii=False))
    result = await pipeline._call(prompt, stage='writer', max_tokens=settings.generation_max_tokens, sources=actual)
    result.trace['upstream_calls'] = calls
    result.trace['funnel'] = {'protocol': PROTOCOL, 'task': spec, 'facts': facts, 'cells': cells,
        'field_summaries': summaries, 'failures': failures,
        'input_source_ids': [s.id for s in sources], 'writer_source_ids': [s.id for s in actual],
        'elapsed_s': monotonic()-tick,
        'citation_audit': audit_citations(result.raw_text or '', [s.model_dump() for s in actual])}
    return result
