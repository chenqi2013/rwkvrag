"""Score V2 protocol outputs without changing any saved model text."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from score_v1 import FACT_KEYS, score_writer


def sha(data):
    return hashlib.sha256(data).hexdigest()


def score_facts(raw, gold):
    predicted = json.loads(raw)
    if not isinstance(predicted, list) or any(
        not isinstance(row, dict) or set(row) != set(FACT_KEYS)
        or any(not isinstance(row[key], str) for key in FACT_KEYS)
        for row in predicted
    ):
        raise ValueError('Expected an array of five-field fact objects')
    rows = [tuple(row[key] for key in FACT_KEYS) for row in predicted]
    if len(rows) != len(set(rows)):
        raise ValueError('Duplicate fact tuple')
    expected = {tuple(row[key] for key in FACT_KEYS) for row in gold}
    actual = set(rows)
    return {'exact': actual == expected, 'tp': len(actual & expected),
            'fp': len(actual - expected), 'fn': len(expected - actual),
            'gold': len(expected), 'extra': sorted(actual - expected),
            'missing': sorted(expected - actual)}


def main(args):
    raw = args.cases.read_bytes()
    cases = [json.loads(line) for line in raw.splitlines()]
    if len(cases) != 23 or len({row['id'] for row in cases}) != 23:
        raise ValueError('V2 case membership changed')
    run = json.loads((args.eval / 'RUN.json').read_text())
    if run['inputs_sha256'] != sha(raw):
        raise ValueError('Model inputs differ')
    records = [json.loads(p.read_text()) for p in (args.eval / 'records').glob('*.json')]
    actual = {(row['id'], row['round'], row['arm']): row for row in records}
    expected_keys = {(row['id'], round_no, arm) for row in cases
                     for round_no in (1, 2) for arm in ('zero', 'trained')}
    if len(records) != 92 or set(actual) != expected_keys:
        raise ValueError('Incomplete or duplicate outputs')
    details, counts = [], Counter()
    for case in cases:
        for round_no in (1, 2):
            for arm in ('zero', 'trained'):
                record = actual[(case['id'], round_no, arm)]
                if (record['prompt_sha256'] != case['prompt_sha256']
                        or record['state_role'] != case['state_role']):
                    raise ValueError('Wrong prompt or State role')
                result = {'id': case['id'], 'stage': case['category'],
                          'round': round_no, 'arm': arm, 'status': record['status'],
                          'raw_sha256': sha(record['raw_text'].encode())}
                stage = case['category']
                if stage == 'extract_array':
                    try:
                        result.update(score_facts(record['raw_text'], case['expected']))
                    except (ValueError, TypeError, KeyError) as exc:
                        result.update(exact=False, parse_error=f'{type(exc).__name__}: {exc}',
                                      tp=0, fp=0, fn=len(case['expected']), gold=len(case['expected']))
                elif stage == 'judge_project':
                    result.update(exact=record['raw_text'].strip() == case['expected'],
                                  expected=case['expected'], predicted=record['raw_text'].strip())
                else:
                    result.update(score_writer(record['raw_text'],
                                               case['writer_source_labels'], [], record['status']))
                    result['citation_present'] = '[资料 ' in record['raw_text']
                details.append(result)
                prefix = stage + '/' + arm + '/'
                counts[prefix + 'records'] += 1
                counts[prefix + 'stopped'] += record['status'] == 'stop'
                counts[prefix + 'output_cap'] += record['status'] == 'length'
                if stage == 'writer_standard':
                    counts[prefix + 'citation_present'] += result['citation_present']
                    counts[prefix + 'invalid_citation'] += bool(result['invalid_citations'] or result['malformed_citations'])
                    counts[prefix + 'repeated'] += result['repeated_long_span']
                else:
                    counts[prefix + 'exact'] += result['exact']
                    if stage == 'extract_array':
                        for key in ('tp', 'fp', 'fn', 'gold'):
                            counts[prefix + key] += result[key]
                        counts[prefix + 'parse_error'] += 'parse_error' in result
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'DETAILS.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in details))
    summary = {'members': len(cases), 'raw_records': len(records),
               'counts': dict(sorted(counts.items())),
               'inputs_sha256': sha(raw), 'run_sha256': sha((args.eval / 'RUN.json').read_bytes()),
               'scorer_sha256': sha(Path(__file__).read_bytes()),
               'limits': 'Seen fictional fixed materials; Writer factual support still requires manual review; no live retrieval.'}
    (args.out / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--eval', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
