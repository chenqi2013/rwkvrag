"""Use the established evidence-first Writer after the bounded reasoning funnel."""
import json
from .writer_prompt import writer_prompt_v2


def grounded_baseline_prompt(task, spec, objects, fields, cells, summaries, facts, labels):
    original = json.loads(task)
    task_data = {
        'history': [message for message in original.get('history', []) if message.get('role') == 'user'],
        'latest_question': original['latest_question'],
        'model_assessments_not_source_facts': {
            'requirements': spec['requirements'],
            'cells': [{'object': objects[cell['object_id']], 'field': fields[cell['field_id']],
                       'execution_status': cell['execution_status'], 'status': cell.get('status'),
                       'value': cell.get('value')} for cell in cells],
            'relations': [{'field': fields[row['field_id']], 'execution_status': row['execution_status'],
                           'status': row.get('status'), 'summary': row.get('summary')} for row in summaries],
        },
    }
    evidence = []
    seen = set()
    for fact in facts:
        key = (fact['source_id'], fact['quote'])
        if key not in seen:
            seen.add(key)
            evidence.append({'label': labels[fact['source_id']], 'text': fact['quote']})
    return writer_prompt_v2(json.dumps(task_data, ensure_ascii=False), evidence, [])
