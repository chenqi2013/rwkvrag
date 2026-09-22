"""Freeze disjoint evaluation runs without changing any case or prompt bytes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def shard(suite):
    if suite == 'fresh_github':
        return 'fresh-github'
    if suite == 'paired188':
        return 'seen-188'
    if suite == 'restored434' or suite.startswith('handoff36_'):
        return 'seen-retrieval-and-handoff'
    if suite in {'dev', 'holdout'}:
        return 'source-separated-dev-holdout'
    raise ValueError('Unregistered evaluation suite: ' + suite)


def main(args):
    raw = args.cases.read_bytes()
    lines = raw.splitlines(keepends=True)
    if not lines or any(not line.endswith(b'\n') for line in lines):
        raise ValueError('Input must be nonempty complete JSONL')
    grouped = {name: [] for name in (
        'fresh-github', 'seen-188', 'seen-retrieval-and-handoff',
        'source-separated-dev-holdout')}
    ids = set()
    suites = Counter()
    for line in lines:
        row = json.loads(line)
        if row['id'] in ids:
            raise ValueError('Duplicate evaluation identity')
        ids.add(row['id'])
        grouped[shard(row['suite'])].append(line)
        suites[row['suite']] += 1
    if any(not group for group in grouped.values()):
        raise ValueError('All registered evaluation shards must be present')
    if (suites['fresh_github'] != 24 or suites['paired188'] != 188
            or suites['restored434'] != 434
            or suites['handoff36_baseline'] != 36
            or suites['handoff36_candidate'] != 36):
        raise ValueError('Frozen historical/fresh membership changed')
    args.out.mkdir(parents=True, exist_ok=False)
    reports = {}
    for name, group in grouped.items():
        payload = b''.join(group)
        (args.out / (name + '.jsonl')).write_bytes(payload)
        reports[name] = {'cases': len(group), 'sha256': digest(payload),
                         'planned_paired_records': 4 * len(group)}
    if sum(x['cases'] for x in reports.values()) != len(lines):
        raise ValueError('Evaluation case lost during sharding')
    summary = {'input_sha256': digest(raw), 'members': len(lines),
               'suites': dict(suites), 'shards': reports,
               'case_bytes_unchanged': True, 'overlap': False,
               'model_comparison': 'same 7.2B model, prompt and evidence; zero versus final role-specific State, two rounds',
               'limits': 'Seen suites are regressions; fresh GitHub cases are independent fixed-evidence questions, not live retrieval.'}
    (args.out / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
