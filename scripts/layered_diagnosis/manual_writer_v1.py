"""Implementer review of all distinct Writer outputs; never rewrites raw text."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


# Judgments were authored after reading every first-round raw Writer answer and
# the frozen evidence. Two rounds are checked for byte identity, not treated as
# independent semantic cases. This is not blinded independent review.
REVIEW = {
    ('two-project-basic', 'zero'): (False, False, True, 'Calls the explicitly ineligible 银榫 uncertain and gives no supported explanation.'),
    ('two-project-basic', 'trained'): (True, False, True, 'Keeps eligibility but says 星槐 has MIT or Apache-2.0 when its source only says MIT.'),
    ('three-project-open-world', 'zero'): (False, False, True, 'Treats untested 赤桁 as rejected and says GPU support information was not provided.'),
    ('three-project-open-world', 'trained'): (True, True, True, 'Correctly describes the three projects and recommends the only eligible one.'),
    ('four-project-conflict', 'zero'): (True, True, True, 'Preserves the same-version conflict, missing license and sole eligible project.'),
    ('four-project-conflict', 'trained'): (True, True, True, 'Preserves the same-version conflict, missing license and sole eligible project.'),
    ('six-project-version-and-names', 'zero'): (False, False, False, 'Omits 木桥 Cloud, turns GPU preference into a hard condition and calls untested 穆桥 Core eligible.'),
    ('six-project-version-and-names', 'trained'): (False, False, False, 'Omits 木桥 Cloud and repeatedly puts 沐桥 Cloud/Edge into contradictory categories.'),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main(args):
    records = [json.loads(p.read_text()) for p in (args.eval / 'records').glob('*.json')]
    writer = {(r['id'].split('/')[0], r['arm'], r['round']): r for r in records if r['id'].endswith('/writer')}
    if len(writer) != 16 or {(s, a) for s, a, _ in writer} != set(REVIEW):
        raise ValueError('Writer membership changed')
    rows, totals = [], Counter()
    for scenario, arm in sorted(REVIEW):
        first, second = writer[(scenario, arm, 1)], writer[(scenario, arm, 2)]
        if (first['raw_text'] != second['raw_text'] or first['generated_ids'] != second['generated_ids']
                or first['status'] != second['status']):
            raise ValueError('Second round differs; requires separate review')
        decision, supported, covered, reason = REVIEW[(scenario, arm)]
        citations = '[资料 ' in first['raw_text']
        row = {'scenario': scenario, 'arm': arm, 'rounds': [1, 2],
               'raw_sha256': sha(first['raw_text'].encode()),
               'decision_preserved': decision, 'all_claims_supported_by_selected_text': supported,
               'all_projects_covered': covered, 'citation_present': citations,
               'full_answer': decision and supported and covered and citations,
               'reason': reason, 'reviewer': 'implementer_not_independent'}
        rows.append(row)
        for key in ('decision_preserved', 'all_claims_supported_by_selected_text',
                    'all_projects_covered', 'citation_present', 'full_answer'):
            totals[arm + '/' + key] += row[key]
        totals[arm + '/distinct_cases'] += 1
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'REVIEW.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))
    summary = {'counts': dict(sorted(totals.items())), 'distinct_answers': len(rows),
               'raw_records_bound': len(writer), 'reviewer': 'implementer_not_independent',
               'run_sha256': sha((args.eval / 'RUN.json').read_bytes()),
               'review_script_sha256': sha(Path(__file__).read_bytes()),
               'limits': 'Subjective full-answer judgments on fictional short evidence; second rounds are duplicates, not new coverage.'}
    (args.out / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
