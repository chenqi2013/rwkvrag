"""Read-only analysis of R0; byte stability is not semantic accuracy."""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def analyze(inputs, root):
    rows = json.loads(inputs.read_text())
    results, errors, counts, changed = [], [], Counter(), []
    for n, row in enumerate(rows):
        rounds = []
        for r in range(1, 4):
            path = root / f'round-{r}' / f'{n:04d}.json'
            if not path.exists():
                errors.append({'ordinal': n, 'round': r, 'error': 'missing'})
                continue
            d = json.loads(path.read_text())
            assert d['call_id'] == row['call_id'] and d['ordinal'] == n and d['round'] == r
            if d['status'] != 'recorded':
                errors.append({'ordinal': n, 'round': r, 'error': d.get('error', d['status'])})
                continue
            wire = base64.b64decode(d['request_body_base64'], validate=True)
            assert sha(wire) == d['request_sha256']
            payload = json.loads(wire)
            if row['stage'] == 'assess':
                payload['vllm_xargs']['rwkv_state_read_ref'] = 'FRESH_ZERO_STATE_REF'
            assert payload == row['payload']
            response = base64.b64decode(d['response_body_base64'], validate=True)
            assert sha(response) == d['response_sha256']
            response = json.loads(response)
            assert response['prompt_token_ids'] == row['prompt_token_ids']
            assert response['choices'][0]['message']['content'] == d['raw_text']
            assert sha(d['raw_text'].encode()) == d['raw_text_sha256']
            rounds.append({'round': r, 'path': str(path.relative_to(root)), 'file_sha256': sha(path.read_bytes()),
                           'raw_text': d['raw_text'], 'sha256': d['raw_text_sha256'],
                           'output_token_ids': d['output_token_ids'], 'finish_reason': d['finish_reason'],
                           'elapsed_ms': d['elapsed_ms'], 'historical_raw_equal': d['historical_raw_equal']})
        signatures = {(d['sha256'], tuple(d['output_token_ids'] or []), d['finish_reason']) for d in rounds}
        stable = len(rounds) == 3 and len(signatures) == 1
        counts[row['stage'] + ':total'] += 1
        counts[row['stage'] + ':stable' if stable else row['stage'] + ':unstable_or_incomplete'] += 1
        record = {'ordinal': n, 'index': row['index'], 'call_id': row['call_id'], 'stage': row['stage'],
                  'stable': stable, 'rounds': rounds, 'source_path': row['source_path'],
                  'historical_raw_text': row['historical_raw_text'],
                  'historical_finish_reason': row['historical_finish_reason']}
        results.append(record)
        if not stable:
            changed.append(record)
    summary = {'planned': len(rows) * 3, 'recorded': sum(len(r['rounds']) for r in results),
               'input_count': len(rows), 'stable_inputs': sum(r['stable'] for r in results),
               'unstable_or_incomplete_inputs': len(changed), 'by_stage': dict(counts), 'errors': errors,
               'historical_text_differences_by_round': {str(i): sum(not d['historical_raw_equal'] for r in results for d in r['rounds'] if d['round'] == i) for i in (1, 2, 3)},
               'finish_reason_counts': dict(Counter(d['finish_reason'] for r in results for d in r['rounds'])),
               'semantic_accuracy_verified': False}
    return summary, results, changed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary, results, changed = analyze(args.inputs, args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in [('SUMMARY.json', summary), ('ALL-REPEATS.json', results), ('DRIFT.json', changed)]:
        (args.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
