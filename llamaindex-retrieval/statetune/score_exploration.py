"""Score complete, hash-verified exploratory generations using the production parser."""
import argparse
import hashlib
import json
from pathlib import Path

from score_state import score_case


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose(scores):
    zero = next(s for s in scores if s['state'] == 'zero')
    def eligible(s):
        return (s['correct'] > zero['correct'] and s['positive_correct'] > zero['positive_correct']
                and s['negative_correct'] >= zero['negative_correct']
                and s['negative_false_positives'] <= zero['negative_false_positives']
                and s['invalid'] == 0)
    def rank(i):
        return (-scores[i]['correct'], scores[i]['negative_false_positives'], scores[i]['invalid'], i)
    qualified = [i for i, s in enumerate(scores) if eligible(s)]
    selected = scores[min(qualified, key=rank)] if qualified else zero
    return selected, scores[min(range(len(scores)), key=rank)], bool(qualified)


def score(config_path, run, gold_path):
    config = json.loads(config_path.read_text())
    root = Path(__file__).resolve().parents[2]
    for path, expected in config['bindings'].items():
        if digest(root/path) != expected:
            raise ValueError('bound source/receipt changed: '+path)
    spec = config['evaluation']
    receipt = json.loads((run/'COMPLETED.json').read_text())
    if (receipt['status'] != 'GENERATION_COMPLETE' or receipt['gold_opened']
            or receipt['optimizer_updates'] != 0 or receipt['config_sha256'] != digest(config_path)
            or receipt['split'] != spec['split']):
        raise ValueError('evaluation receipt mismatch')
    inputs_path = root/spec['inputs']
    if digest(inputs_path) != spec['inputs_sha256']:
        raise ValueError('input hash mismatch')
    inputs = {r['id']: r for r in map(json.loads, inputs_path.read_text().splitlines())}
    states = {s['name']: s for s in spec['states']}
    records = {}
    for pin in receipt['records']:
        path = run/pin['path']
        if path.parent != run or digest(path) != pin['sha256']:
            raise ValueError('raw output changed')
        record = json.loads(path.read_text())
        key = (record['state'], record['id'])
        if (key in records or record['state_sha256'] != states[record['state']]['sha256']
                or record['prompt_sha256'] != inputs[record['id']]['prompt_sha256']):
            raise ValueError('raw identity mismatch')
        records[key] = record
    if set(records) != {(name, identity) for name in states for identity in inputs}:
        raise ValueError('incomplete/duplicate generation coverage')
    # Gold content is first opened only after the entire raw set is verified.
    raw = gold_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec['gold_sha256']:
        raise ValueError('gold changed')
    gold = {r['id']: r for r in map(json.loads, raw.splitlines())}
    if set(gold) != set(inputs):
        raise ValueError('gold coverage mismatch')
    scores = []
    for name, state in states.items():
        cases = [{'id': identity, **score_case(records[name, identity]['output'],
                  gold[identity]['expected_unit_ids'], len(row['units']))}
                 for identity, row in inputs.items()]
        scores.append({'state': name, 'sha256': state['sha256'], 'count': len(cases),
              'correct': sum(c['correct'] for c in cases),
              'positive_correct': sum(c['correct'] and bool(c['expected']) for c in cases),
              'negative_correct': sum(c['correct'] and not c['expected'] for c in cases),
              'invalid': sum(not c['valid'] for c in cases),
              'negative_false_positives': sum(c['negative_false_positive'] for c in cases),
              'cases': cases})
    # Input state order is frozen before generation; earlier checkpoint wins exact ties.
    best, overall, improves = choose(scores)
    return {'scores': scores, 'selected_state': best['state'],
            'overall_highest_scoring_state': overall['state'], 'improves_over_zero': improves,
            'split': spec['split'], 'gold_opened_after_complete_raw': True,
            'config_sha256': digest(config_path), 'receipt_sha256': digest(run/'COMPLETED.json'),
            'production_promoted': False, 'blind_benchmark': False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--gold', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = score(args.config, args.run, args.gold)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({**result, 'scores': [{k:v for k,v in s.items() if k != 'cases'}
                                          for s in result['scores']]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
