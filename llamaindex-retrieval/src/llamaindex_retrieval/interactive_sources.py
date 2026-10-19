"""Model catalog selection and structural source/queue handling only."""
import asyncio
from hashlib import sha256
import json

CATALOG = '根据问题选择值得阅读的文档ID。标题只是查找线索，不是事实依据。选择可能包含所问信息或必要关联的文档；没有合适文档时返回空数组。只输出source_ids。'


def digest(text):
    return sha256(text.encode()).hexdigest()


def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate JSON key')
        value[key] = item
    return value


async def gather(jobs):
    tasks = [asyncio.create_task(job) for job in jobs]
    try:
        return await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def add_sources(pool, incoming):
    by_id = {s.id: s for s in pool}
    for source in incoming:
        if source.id in by_id:
            old = by_id[source.id]
            if (source.title, source.snippet) != (old.title, old.snippet):
                raise ValueError('source ID reused for different original')
        else:
            pool.append(source)
            by_id[source.id] = source


async def choose_catalog(model, question, catalog):
    """Offer every title; 128 batches prompts, never caps selected documents."""
    async def choose(group):
        ids = [r['id'] for r in group]
        schema = obj({'source_ids': {'type': 'array', 'items': {'type': 'string', 'enum': ids}}})
        result = await model.complete([{'role': 'user', 'content': CATALOG + '\n' +
            json.dumps({'question': question, 'documents': group}, ensure_ascii=False, separators=(',', ':'))}],
            structured_schema=schema, max_tokens=2048, assistant_prefill='<think></think',
            temperature=0, seed=11, stage='reader', evidence_ids=ids)
        event = {'status': result.status, 'raw_text': result.raw_text, 'parsed': None,
                 'raw_sha256': digest(result.raw_text or ''), 'trace': result.trace}
        if result.status == 'completed':
            try:
                value = json.loads(result.raw_text, object_pairs_hook=unique_object)
                if (not isinstance(value, dict) or set(value) != {'source_ids'}
                        or not isinstance(value['source_ids'], list)
                        or any(not isinstance(i, str) or i not in ids for i in value['source_ids'])):
                    raise ValueError('invalid catalog selection')
                event['parsed'] = value
            except (ValueError, TypeError) as error:
                event['validation_error'] = str(error)
        return event
    decisions = await gather([choose(catalog[i:i+128]) for i in range(0, len(catalog), 128)])
    selected = {sid for d in decisions if d['parsed'] is not None for sid in d['parsed']['source_ids']}
    return [r['id'] for r in catalog if r['id'] in selected], decisions
