"""Exact-token packing of whole Resolver-selected sources; no answer rewriting."""
from collections import deque


def diverse_order(sources):
    """Round-robin physical source groups, retaining rank within each group."""
    groups = {}
    for source in sources:
        identity = (source.uri or source.document_id or
                    source.metadata.get('parent_source_id') or source.id)
        groups.setdefault(identity, deque()).append(source)
    queues = deque(groups.values())
    while queues:
        queue = queues.popleft()
        yield queue.popleft()
        if queue:
            queues.append(queue)


async def write_with_budget(pipeline, task, sources, fields):
    settings = pipeline.settings
    if settings.native_transport != 'native':
        raise ValueError('whole_sources requires the native tokenizer transport')
    checks = []

    async def check(items):
        result = await pipeline.model.complete(
            [{'role': 'user', 'content': pipeline._writer_prompt(task, items, fields)}],
            max_tokens=settings.generation_max_tokens,
            assistant_prefill=settings.native_writer_prefill,
            stage='writer_budget', evidence_ids=tuple(s.id for s in items),
            check_only=True,
            **({'temperature': 1.0, 'top_p': 1.0, 'top_k': 1, 'seed': 11}
               if settings.native_completion_protocol == 'g1j_plain' else {}))
        checks.append(result.trace)
        return result

    def attach(result, included, status):
        result.trace['evidence_budget'] = {
            'policy': 'whole_sources_v1', 'status': status,
            'input_source_ids': [s.id for s in sources],
            'included_source_ids': [s.id for s in included],
            'omitted_source_ids': [s.id for s in sources if s.id not in {x.id for x in included}],
            'coverage_complete': len(included) == len(sources),
            'checks': [dict(check) for check in checks],
        }
        return result

    full = await check(sources)
    if full.status == 'completed':
        included = sources
    elif full.status != 'budget_exceeded':
        # Tokenizer failure is an execution failure, not an empty-evidence answer.
        return attach(full, [], 'tokenizer_failed')
    else:
        empty = await check([])
        if empty.status != 'completed':
            return attach(empty, [], 'task_exceeds_budget' if empty.status == 'budget_exceeded'
                          else 'tokenizer_failed')
        included = []
        for source in diverse_order(sources):
            candidate = [*included, source]
            measured = await check(candidate)
            if measured.status == 'completed':
                included = candidate
            elif measured.status != 'budget_exceeded':
                return attach(measured, [], 'tokenizer_failed')
        if sources and not included:
            # Do not misrepresent oversized evidence as an empty search result.
            return attach(full, [], 'no_whole_source_fits')
    result = await pipeline._call(pipeline._writer_prompt(task, included, fields),
        stage='writer', max_tokens=settings.generation_max_tokens, sources=included)
    return attach(result, included, 'complete' if len(included) == len(sources) else 'partial')
