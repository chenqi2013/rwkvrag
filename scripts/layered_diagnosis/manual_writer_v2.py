"""Bind implementer judgments to every distinct V2 Writer raw answer."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


# Authored after reading all eight distinct raw answers and frozen source text.
# Decision preservation counts literal JSON category reports, but full_answer
# requires an actual concise user-facing answer with supported citations.
REVIEW = {
    ('two-project-basic', 'zero'): (False, False, False, 'Reassigns GPU preference as hard, contradicts eligibility, repeats until output cap.'),
    ('two-project-basic', 'trained'): (True, True, False, 'Correct choice and source facts, but calls unchecked GPU a project qualification gap; conclusions lack adjacent citations.'),
    ('three-project-open-world', 'zero'): (False, False, False, 'Treats GPU preference as hard and repeats false rejection until output cap.'),
    ('three-project-open-world', 'trained'): (True, False, False, 'Correct candidate categories but falsely says 蓝铆 has GPU acceleration and invents nonnumeric citation labels.'),
    ('four-project-conflict', 'zero'): (True, True, False, 'Copies correct Gold categories as JSON, but gives no requested prose or citations.'),
    ('four-project-conflict', 'trained'): (False, False, False, 'Turns unknown 梭羽 license into hard rejection and contradicts unresolved 岚阙 in the final sentence.'),
    ('six-project-version-and-names', 'zero'): (True, True, False, 'Copies correct Gold categories as JSON, but gives no requested prose or citations.'),
    ('six-project-version-and-names', 'trained'): (False, False, False, 'Swaps several similar project facts and eligibility classes; gives no citations.'),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main(args):
    records = [json.loads(p.read_text()) for p in (args.eval / 'records').glob('*.json')]
    writers = {(r['id'].split('/')[0], r['arm'], r['round']): r
               for r in records if r['category'] == 'writer_standard'}
    if len(writers) != 16 or {(scenario, arm) for scenario, arm, _ in writers} != set(REVIEW):
        raise ValueError('Writer membership changed')
    rows, counts = [], Counter()
    for scenario, arm in sorted(REVIEW):
        first, second = writers[(scenario, arm, 1)], writers[(scenario, arm, 2)]
        if (first['raw_text'] != second['raw_text']
                or first['generated_ids'] != second['generated_ids']
                or first['status'] != second['status']):
            raise ValueError('Rounds differ; separate review required')
        decision, supported, full, reason = REVIEW[(scenario, arm)]
        row = {'scenario': scenario, 'arm': arm, 'rounds': [1, 2],
               'raw_sha256': sha(first['raw_text'].encode()),
               'decision_preserved': decision,
               'all_claims_supported_by_selected_text': supported,
               'full_answer': full, 'reason': reason,
               'reviewer': 'implementer_not_independent'}
        rows.append(row)
        counts[f'{arm}/distinct_cases'] += 1
        for key in ('decision_preserved', 'all_claims_supported_by_selected_text', 'full_answer'):
            counts[f'{arm}/{key}'] += row[key]
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'REVIEW.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    summary = {'counts': dict(sorted(counts.items())), 'distinct_answers': len(rows),
               'raw_records_bound': len(writers),
               'run_sha256': sha((args.eval / 'RUN.json').read_bytes()),
               'review_script_sha256': sha(Path(__file__).read_bytes()),
               'reviewer': 'implementer_not_independent',
               'limits': 'Seen fictional evidence; subjective review; duplicate rounds are not new coverage.'}
    (args.out / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
