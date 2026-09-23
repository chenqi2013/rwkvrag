"""Score exact single-condition labels without posthoc content repair."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main(args):
    raw = args.cases.read_bytes()
    cases = [json.loads(line) for line in raw.splitlines()]
    if len(cases) != 30 or len({row['id'] for row in cases}) != 30:
        raise ValueError('V3 case membership changed')
    run = json.loads((args.eval / 'RUN.json').read_text())
    if run['inputs_sha256'] != sha(raw):
        raise ValueError('Model inputs differ')
    records = [json.loads(p.read_text()) for p in (args.eval / 'records').glob('*.json')]
    actual = {(row['id'], row['round'], row['arm']): row for row in records}
    expected = {(row['id'], round_no, arm) for row in cases
                for round_no in (1, 2) for arm in ('zero', 'trained')}
    if len(records) != 120 or set(actual) != expected:
        raise ValueError('Incomplete or duplicate outputs')
    details, counts = [], Counter()
    for case in cases:
        for arm in ('zero', 'trained'):
            first, second = (actual[(case['id'], round_no, arm)] for round_no in (1, 2))
            if (first['prompt_sha256'] != case['prompt_sha256']
                    or first['state_role'] != 'resolver'):
                raise ValueError('Prompt or role differs')
            identical = (first['raw_text'] == second['raw_text']
                         and first['generated_ids'] == second['generated_ids']
                         and first['status'] == second['status'])
            for round_no, record in ((1, first), (2, second)):
                label = record['raw_text'].strip()
                row = {'id': case['id'], 'arm': arm, 'round': round_no,
                       'expected': case['expected'], 'predicted': label,
                       'exact': label == case['expected'], 'status': record['status'],
                       'raw_sha256': sha(record['raw_text'].encode()),
                       'rounds_identical': identical}
                details.append(row)
                prefix = arm + '/'
                counts[prefix + 'records'] += 1
                counts[prefix + 'exact'] += row['exact']
                counts[prefix + 'output_cap'] += record['status'] == 'length'
                counts[prefix + 'predicted/' + label] += 1
                counts[prefix + 'gold/' + case['expected']] += 1
                counts[prefix + 'confusion/' + case['expected'] + '->' + label] += 1
            counts[arm + '/distinct_cases'] += 1
            counts[arm + '/stable_cases'] += identical
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'DETAILS.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in details))
    summary = {'members': len(cases), 'raw_records': len(records),
               'counts': dict(sorted(counts.items())),
               'inputs_sha256': sha(raw),
               'run_sha256': sha((args.eval / 'RUN.json').read_bytes()),
               'scorer_sha256': sha(Path(__file__).read_bytes()),
               'limits': 'Seen fictional Gold-field classification only; no retrieval or independent evaluation.'}
    (args.out / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--eval', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
