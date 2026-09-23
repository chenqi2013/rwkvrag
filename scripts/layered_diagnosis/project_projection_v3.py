"""Exploratory score-only projection of two raw condition labels to project status."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from prepare_v1 import SCENARIOS


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main(args):
    run = json.loads((args.eval / 'RUN.json').read_text())
    if run['inputs_sha256'] != sha(args.cases.read_bytes()):
        raise ValueError('Inputs differ')
    cases = [json.loads(line) for line in args.cases.read_text().splitlines()]
    if len(cases) != 30:
        raise ValueError('Case membership changed')
    records = [json.loads(p.read_text()) for p in (args.eval / 'records').glob('*.json')]
    by = {(r['id'], r['arm'], r['round']): r for r in records}
    if len(records) != 120:
        raise ValueError('Raw records incomplete')
    scenarios = json.loads(SCENARIOS.read_text())
    rows, counts = [], Counter()
    for scenario in scenarios:
        for project in scenario['projects']:
            gold = next(status for status in ('eligible', 'rejected', 'unresolved')
                        if project in scenario['decision'][status])
            for arm in ('zero', 'trained'):
                labels = [by[(f'{scenario["id"]}/condition/{project}/{field}', arm, 1)]['raw_text'].strip()
                          for field in ('windows', 'license')]
                # Evaluation projection only, never used by the serving pipeline.
                status = ('rejected' if 'not_satisfied' in labels else
                          'eligible' if labels == ['satisfied', 'satisfied'] else 'unresolved')
                row = {'scenario': scenario['id'], 'project': project, 'arm': arm,
                       'raw_labels': labels, 'expected': gold, 'projected': status,
                       'correct': status == gold,
                       'scope': 'exploratory_score_only_not_product_decision'}
                rows.append(row)
                counts[arm + '/correct'] += row['correct']
                counts[arm + '/total'] += 1
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'DETAILS.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    summary = {'counts': dict(sorted(counts.items())), 'rows': len(rows),
               'run_sha256': sha((args.eval / 'RUN.json').read_bytes()),
               'script_sha256': sha(Path(__file__).read_bytes()),
               'limits': 'Known scenario hard-condition rule applied to raw labels for diagnosis; not a trained aggregator, production decision, or held-out result.'}
    (args.out / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--eval', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
