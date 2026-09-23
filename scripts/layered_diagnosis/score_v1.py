"""Score exact oracle outputs; preserve every raw failure and leave Writer semantics for review."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

FACT_KEYS = ('project', 'attribute', 'value', 'source_id', 'status')
DECISION_KEYS = ('eligible', 'rejected', 'unresolved', 'recommended')
TAG = re.compile(r'\[资料\s*(\d+)\]')
CANDIDATE_TAG = re.compile(r'\[(?:资料|Source)[^\]\r\n]*(?:\]|$)')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def repeated_long_span(text):
    normalized = re.sub(r'\[资料[^\]\r\n]*\]', '', text)
    normalized = ''.join(normalized.lower().split())
    for width, minimum in ((96, 2), (32, 3)):
        seen = {}
        for index in range(len(normalized) - width + 1):
            span = normalized[index:index+width]
            positions = seen.setdefault(span, [])
            if not positions or index - positions[-1] >= width:
                positions.append(index)
                if len(positions) >= minimum:
                    return True
    return False


def parse_exact_json(raw):
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('Output is not one JSON object')
    return value


def score_extract(raw, expected):
    data = parse_exact_json(raw)
    if set(data) != {'facts'} or not isinstance(data['facts'], list):
        raise ValueError('Fact schema mismatch')
    facts = data['facts']
    if any(not isinstance(f, dict) or set(f) != set(FACT_KEYS)
           or any(not isinstance(f[k], str) for k in FACT_KEYS) for f in facts):
        raise ValueError('Invalid fact entry')
    predicted = [tuple(f[k] for k in FACT_KEYS) for f in facts]
    if len(predicted) != len(set(predicted)):
        raise ValueError('Duplicate fact entry')
    gold = {tuple(f[k] for k in FACT_KEYS) for f in expected['facts']}
    got = set(predicted)
    swap = Counter()
    for fact in got - gold:
        if any(fact[1:] == valid[1:] for valid in gold):
            swap['project'] += 1
        if any((fact[0], fact[1], fact[2], fact[4]) ==
               (valid[0], valid[1], valid[2], valid[4]) for valid in gold):
            swap['source'] += 1
        if any((fact[0], fact[1], fact[3], fact[4]) ==
               (valid[0], valid[1], valid[3], valid[4]) for valid in gold):
            swap['value'] += 1
        if any(fact[:4] == valid[:4] for valid in gold):
            swap['status'] += 1
    return {'exact': got == gold, 'tp': len(got & gold), 'fp': len(got - gold),
            'fn': len(gold - got), 'gold': len(gold), 'predicted': len(got),
            'swap_hints': dict(swap), 'extra': sorted(got - gold), 'missing': sorted(gold - got)}


def score_compare(raw, expected, projects):
    data = parse_exact_json(raw)
    if set(data) != set(DECISION_KEYS) or any(not isinstance(data[k], list)
            or any(not isinstance(x, str) for x in data[k])
            or len(set(data[k])) != len(data[k]) for k in DECISION_KEYS):
        raise ValueError('Decision schema mismatch')
    sets = {k: set(data[k]) for k in DECISION_KEYS}
    if (any(not value <= set(projects) for value in sets.values())
            or sets['eligible'] & sets['rejected']
            or sets['eligible'] & sets['unresolved']
            or sets['rejected'] & sets['unresolved']):
        raise ValueError('Invalid project reference or decision partition')
    gold_status = {name: k for k in DECISION_KEYS[:3] for name in expected[k]}
    got_status = {name: k for k in DECISION_KEYS[:3] for name in data[k]}
    correct = sum(got_status.get(name) == status for name, status in gold_status.items())
    return {'exact': all(sets[k] == set(expected[k]) for k in DECISION_KEYS),
            'status_correct': correct, 'status_total': len(projects),
            'recommended_exact': sets['recommended'] == set(expected['recommended']),
            'missing_status': sorted(set(projects) - set(got_status)),
            'wrong_status': {name: got_status.get(name) for name, status in gold_status.items()
                             if got_status.get(name) != status}}


def score_writer(raw, source_labels, projects, finish):
    allowed = set(source_labels)
    valid = {f'资料 {x}' for x in TAG.findall(raw)}
    malformed = [tag for tag in CANDIDATE_TAG.findall(raw) if not TAG.fullmatch(tag)]
    return {'stopped': finish == 'stop', 'output_cap': finish == 'length',
            'invalid_citations': sorted(valid - allowed), 'malformed_citations': malformed,
            'repeated_long_span': repeated_long_span(raw),
            'missing_project_mentions': [name for name in projects if name not in raw],
            'semantic_support_reviewed': False}


def main(args):
    cases_raw = args.cases.read_bytes()
    cases = [json.loads(line) for line in cases_raw.splitlines()]
    if len(cases) != 12 or len({x['id'] for x in cases}) != 12:
        raise ValueError('Diagnostic case membership changed')
    run = json.loads((args.eval / 'RUN.json').read_text())
    if run['inputs_sha256'] != sha(cases_raw):
        raise ValueError('Model run used different inputs')
    records = [json.loads(p.read_text()) for p in (args.eval / 'records').glob('*.json')]
    actual = {(x['id'], x['round'], x['arm']): x for x in records}
    expected = {(case['id'], round_no, arm) for case in cases
                for round_no in (1, 2) for arm in ('zero', 'trained')}
    if len(records) != 48 or set(actual) != expected:
        raise ValueError('Incomplete or duplicate paired outputs')
    details, totals = [], Counter()
    for case in cases:
        stage = case['category']
        projects = list(dict.fromkeys(x['project'] for x in case['expected']['facts'])) if stage == 'extract' else None
        if stage == 'compare':
            projects = [*case['expected']['eligible'], *case['expected']['rejected'],
                        *case['expected']['unresolved']]
        if stage == 'writer':
            projects = [*case['expected']['eligible'], *case['expected']['rejected'],
                        *case['expected']['unresolved']]
        for round_no in (1, 2):
            for arm in ('zero', 'trained'):
                record = actual[(case['id'], round_no, arm)]
                if record.get('prompt_sha256') != case['prompt_sha256'] or record['state_role'] != case['state_role']:
                    raise ValueError('Prompt or State role changed')
                result = {'id': case['id'], 'stage': stage, 'projects': case['project_count'],
                          'round': round_no, 'arm': arm, 'status': record['status'],
                          'raw_sha256': sha(record['raw_text'].encode()),
                          'generated_tokens': len(record['generated_ids'])}
                if stage == 'writer':
                    result.update(score_writer(record['raw_text'], case['writer_source_labels'], projects, record['status']))
                else:
                    try:
                        if stage == 'extract':
                            result.update(score_extract(record['raw_text'], case['expected']))
                        else:
                            result.update(score_compare(record['raw_text'], case['expected'], projects))
                    except (ValueError, TypeError, KeyError) as exc:
                        result.update(exact=False, parse_error=f'{type(exc).__name__}: {exc}')
                details.append(result)
                key = f'{stage}/{arm}'
                totals[key + '/records'] += 1
                totals[key + '/stopped'] += record['status'] == 'stop'
                totals[key + '/output_cap'] += record['status'] == 'length'
                if stage == 'writer':
                    totals[key + '/invalid_citation_records'] += bool(result['invalid_citations'] or result['malformed_citations'])
                    totals[key + '/repeated_records'] += result['repeated_long_span']
                    totals[key + '/all_projects_mentioned'] += not result['missing_project_mentions']
                else:
                    totals[key + '/exact'] += result['exact']
                    totals[key + '/parse_errors'] += 'parse_error' in result
                    if stage == 'extract':
                        totals[key + '/gold'] += len(case['expected']['facts'])
                    if stage == 'compare':
                        totals[key + '/status_total'] += len(projects)
                    if stage == 'extract' and 'parse_error' not in result:
                        for field in ('tp', 'fp', 'fn'):
                            totals[key + '/' + field] += result[field]
                    if stage == 'extract' and 'parse_error' in result:
                        totals[key + '/fn'] += len(case['expected']['facts'])
                    if stage == 'compare' and 'parse_error' not in result:
                        totals[key + '/status_correct'] += result['status_correct']
                        totals[key + '/recommended_exact'] += result['recommended_exact']
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'DETAILS.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in details))
    summary = {'members': len(cases), 'raw_records': len(records), 'counts': dict(sorted(totals.items())),
               'inputs_sha256': sha(cases_raw), 'run_sha256': sha((args.eval / 'RUN.json').read_bytes()),
               'score_sha256': sha(Path(__file__).read_bytes()),
               'semantics': 'extract/compare exact gold only; Writer semantics not yet reviewed',
               'limits': 'Fictional fixed evidence and implementer-authored gold; no live retrieval or independent blind review.'}
    (args.out / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--eval', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
