"""Transport/format audit and full manual-review packet, not an accuracy scorer."""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from llamaindex_retrieval.native_rwkv import inspect_envelope
from llamaindex_retrieval.offline_replay import strict_json


class Judgment(BaseModel):
    # Same schema as the frozen old layered_evidence.Judgment.
    model_config = ConfigDict(extra='forbid')
    conclusion: str = Field(min_length=1, max_length=1200)
    evidence_ids: list[str] = Field(max_length=16)


def parse(raw, finish):
    if finish != 'stop':
        return None, ['finish_' + str(finish)]
    bounds = inspect_envelope(raw, '<think></think')
    if bounds is None:
        return None, ['invalid_envelope']
    start, end = bounds
    while start < end and raw[start].isspace():
        start += 1
    while end > start and raw[end-1].isspace():
        end -= 1
    if raw.startswith('```json\n', start):
        if not raw[start:end].endswith('\n```'):
            return None, ['invalid_structure']
        start += len('```json\n')
        end -= len('\n```')
    try:
        return Judgment.model_validate(strict_json(raw[start:end])).model_dump(), []
    except (ValueError, TypeError):
        return None, ['invalid_structure']


def analyze(inputs, expectations, root):
    rows = json.loads(inputs.read_text())
    gold = {r['call_id']: r for r in json.loads(expectations.read_text())['rows']}
    packet, counts, errors = [], Counter(), []
    for n, row in enumerate(rows):
        record = {'ordinal': n, 'index': row['index'], 'call_id': row['call_id'],
                  'question_and_evidence': row['payload']['messages'][0]['content'],
                  'expected_semantics': gold[row['call_id']]['expected_semantics'],
                  'allowed_evidence_ids': row['evidence_ids'], 'outputs': []}
        for repetition in (1, 2):
            for arm in ('zero', '2000'):
                path = root / f'round-{repetition}' / arm / f'{n:04d}.json'
                if not path.exists():
                    errors.append({'ordinal': n, 'round': repetition, 'arm': arm, 'error': 'missing'})
                    continue
                raw = path.read_bytes()
                d = json.loads(raw)
                assert d['call_id'] == row['call_id'] and d['arm'] == arm
                assert d['round'] == repetition and d['ordinal'] == n
                wire = base64.b64decode(d['request_body_base64'], validate=True)
                assert hashlib.sha256(wire).hexdigest() == d['request_sha256']
                request = json.loads(wire)
                request['vllm_xargs']['rwkv_state_read_ref'] = 'FRESH_ZERO_STATE_REF'
                assert request == row['payload']
                if d['status'] != 'recorded':
                    errors.append({'ordinal': n, 'round': repetition, 'arm': arm, 'error': d.get('error')})
                    continue
                body = base64.b64decode(d['response_body_base64'], validate=True)
                assert hashlib.sha256(body).hexdigest() == d['response_sha256']
                response = json.loads(body)
                assert response['prompt_token_ids'] == row['prompt_token_ids']
                assert response['choices'][0]['message']['content'] == d['raw_text']
                assert hashlib.sha256(d['raw_text'].encode()).hexdigest() == d['raw_text_sha256']
                parsed, issues = parse(d['raw_text'], d['finish_reason'])
                if parsed is not None:
                    if not set(parsed['evidence_ids']) <= set(row['evidence_ids']):
                        issues.append('unknown_evidence_id')
                    if len(parsed['evidence_ids']) != len(set(parsed['evidence_ids'])):
                        issues.append('duplicate_evidence_id')
                counts[f'{repetition}:{arm}:recorded'] += 1
                for issue in issues:
                    counts[f'{repetition}:{arm}:{issue}'] += 1
                record['outputs'].append({'round': repetition, 'arm': arm, 'path': str(path.relative_to(root)),
                    'file_sha256': hashlib.sha256(raw).hexdigest(), 'raw_text': d['raw_text'],
                    'raw_text_sha256': d['raw_text_sha256'], 'finish_reason': d['finish_reason'],
                    'parsed': parsed, 'issues': issues, 'elapsed_ms': d['elapsed_ms']})
        packet.append(record)
    drift = {}
    for arm in ('zero', '2000'):
        drift[arm] = [r['ordinal'] for r in packet
                      if len([d for d in r['outputs'] if d['arm'] == arm]) != 2
                      or len({(d['raw_text_sha256'], d['finish_reason'])
                              for d in r['outputs'] if d['arm'] == arm}) != 1]
    return {'planned': 564, 'counts': dict(counts), 'errors': errors, 'cross_round_drift': drift,
            'semantic_accuracy_verified': False}, packet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('inputs', 'expectations', 'root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    summary, packet = analyze(args.inputs, args.expectations, args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in [('STRUCTURAL-SUMMARY.json', summary), ('REVIEW-PACKET.json', packet)]:
        (args.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
