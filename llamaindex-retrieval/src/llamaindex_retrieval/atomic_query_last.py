"""Same atomic protocol; move source data before the object/field query."""
import json
from . import typed_funnel_contract_v7 as c
from .typed_funnel_v7 import atomic_prompt as original_prompt


def atomic_prompt(object_name, field, source, units):
    instruction, encoded = original_prompt(object_name, field, source, units).split('\n', 1)
    content = json.loads(encoded)
    reordered = {key: content[key] for key in ('source_title', 'evidence', 'object', 'field')}
    return instruction + '\n' + json.dumps(reordered, ensure_ascii=False)


async def extract_atomic(runner, source, object_name, field, *, purpose='funnel_fact'):
    from .rwkv_pipeline import evidence_units
    units = {f'E{i}': unit for i, unit in enumerate(evidence_units(0, source.snippet, 320, 64), 1)}
    return await runner.node(atomic_prompt(object_name, field, source, units), purpose, c.Atomic, [source],
        lambda value: c.validate_atomic(value, field, units), document=c.atomic_schema(field, units), max_tokens=512)
