"""Explicitly post-hoc analysis of content hidden by a schema failure.

This does not change the strict pre-registered score or any model output.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from score_v1 import FACT_KEYS, score_extract, TAG, CANDIDATE_TAG


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main(args):
    cases = {x['id']: x for x in map(json.loads, args.cases.read_text().splitlines())}
    records = [json.loads(p.read_text()) for p in (args.eval / 'records').glob('*.json')]
    if len(records) != 48 or {r['id'] for r in records} != set(cases):
        raise ValueError('Incomplete fixed run')
    run = json.loads((args.eval / 'RUN.json').read_text())
    if run['inputs_sha256'] != sha(args.cases.read_bytes()):
        raise ValueError('Input pin changed')
    details, totals = [], Counter()
    for r in sorted(records, key=lambda x: (x['id'], x['round'], x['arm'])):
        case = cases[r['id']]
        stage = case['category']
        if stage not in {'extract', 'writer'}:
            continue
        row = {'id': r['id'], 'arm': r['arm'], 'round': r['round'],
               'raw_sha256': sha(r['raw_text'].encode()), 'stage': stage}
        if stage == 'extract':
            try:
                parsed = json.loads(r['raw_text'])
                row['raw_shape'] = 'array' if isinstance(parsed, list) else 'object' if isinstance(parsed, dict) else 'other'
                if isinstance(parsed, list):
                    parsed = {'facts': parsed}
                if not isinstance(parsed, dict) or set(parsed) != {'facts'} or not isinstance(parsed['facts'], list):
                    raise ValueError('No facts list')
                facts = parsed['facts']
                if any(not isinstance(x, dict) or set(x) != set(FACT_KEYS) for x in facts):
                    raise ValueError('Invalid fact entry')
                signatures = [tuple(x[k] for k in FACT_KEYS) for x in facts]
                row['duplicate_tuples'] = len(signatures) - len(set(signatures))
                unique = [dict(zip(FACT_KEYS, values)) for values in dict.fromkeys(signatures)]
                content = score_extract(json.dumps({'facts': unique}, ensure_ascii=False), case['expected'])
                row.update({'content_exact_after_shape_and_duplicate_normalization': content['exact'],
                            'tp': content['tp'], 'fp': content['fp'], 'fn': content['fn'],
                            'swap_hints': content['swap_hints']})
                key = 'extract/' + r['arm']
                totals[key + '/tp'] += content['tp']
                totals[key + '/fp'] += content['fp']
                totals[key + '/fn'] += content['fn']
                totals[key + '/content_exact'] += content['exact']
                totals[key + '/array_shape'] += row['raw_shape'] == 'array'
                totals[key + '/duplicate_tuples'] += row['duplicate_tuples']
            except (ValueError, TypeError, KeyError) as exc:
                row['content_error'] = f'{type(exc).__name__}: {exc}'
                totals['extract/' + r['arm'] + '/content_errors'] += 1
        else:
            tags = TAG.findall(r['raw_text'])
            row['citation_count'] = len(tags)
            row['no_citation'] = len(tags) == 0
            row['malformed_citations'] = [x for x in CANDIDATE_TAG.findall(r['raw_text']) if not TAG.fullmatch(x)]
            totals['writer/' + r['arm'] + '/no_citation'] += row['no_citation']
        details.append(row)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'DETAILS.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in details))
    report = {'counts': dict(sorted(totals.items())), 'records_examined': len(details),
              'input_sha256': sha(args.cases.read_bytes()), 'run_sha256': sha((args.eval / 'RUN.json').read_bytes()),
              'analysis_sha256': sha(Path(__file__).read_bytes()),
              'status': 'post_hoc_exploratory_not_pre_registered',
              'interpretation': 'Array wrapping and duplicate removal are scoring views only; strict schema failures remain failures.'}
    (args.out / 'SUMMARY.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--eval', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
