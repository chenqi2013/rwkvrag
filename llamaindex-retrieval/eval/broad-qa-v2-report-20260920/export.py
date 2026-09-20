"""Post-run evidence export; no semantic grader or answer rewriting."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'data/quality-runs/broad-qa-v2-20260920/run1'
ART = ROOT / 'artifacts/broad-regression-20260920'
CASES = ROOT / 'llamaindex-retrieval/eval/broad-qa-v2-20260920/cases.json'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    rows = json.loads((RUN / 'ROWS.json').read_text())
    cases = {c['id']: c for c in json.loads(CASES.read_text())}
    assert len(rows) == len(cases) == 478
    assert {r['case_id'] for r in rows} == set(cases)
    destination = ART / 'qa-v2'
    destination.mkdir(exist_ok=False)
    # ROWS contains the same huge raw responses as calls; archive those original
    # bytes once, and expose compact result/answer views for direct review.
    for path in RUN.iterdir():
        if path.is_file() and path.name != 'ROWS.json':
            shutil.copy2(path, destination / path.name)
    shutil.copy2('/tmp/broad-qa-v2-run1.log', destination / 'execution.log')
    manifest = {}
    with tarfile.open(destination / 'raw-calls.tar.gz', 'w:gz') as archive:
        for path in sorted((RUN / 'calls').glob('*.json')):
            manifest[path.name] = sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname='calls/' + path.name)
    assert len(manifest) == 478
    with tarfile.open(destination / 'raw-calls.tar.gz') as archive:
        for member in archive:
            assert sha256(archive.extractfile(member).read()).hexdigest() == manifest[Path(member.name).name]
    save(destination / 'RAW-MANIFEST.json', manifest)
    compact = []
    answers = []
    groups = defaultdict(list)
    review_by_id = {r['case_id']: r for r in json.loads((ART / 'MANUAL-REVIEWS.json').read_text())['items']}
    plan = json.loads((ART / 'MANUAL-REVIEW-PLAN.json').read_text())
    assert set(review_by_id) == set(plan['case_ids']) and len(review_by_id) == 66
    (destination / 'review-cases').mkdir()
    for row in rows:
        case = cases[row['case_id']]
        response = json.loads(row.get('raw_response') or '{}')
        generation = response.get('generation', {})
        record = {k: v for k, v in row.items() if k != 'raw_response'}
        record['execution_scope'] = case['execution_scope']
        record['raw_file'] = f'calls/{row["index"]:04d}.json'
        compact.append(record)
        answer = {'case_id': row['case_id'], 'suite': row['suite'],
            'question': case['payload']['question'], 'execution_scope': case['execution_scope'],
            'http_status': row.get('http_status'), 'error': row.get('error'),
            'answer': response.get('answer'), 'sources': response.get('sources', []),
            'status': generation.get('status'), 'answer_span': generation.get('answer_span'),
            'answer_modified': generation.get('answer_modified'),
            'diagnostics': row.get('diagnostics'), 'raw_file': record['raw_file']}
        answers.append(answer)
        groups[case['execution_scope']].append(row)
        if row['case_id'] in review_by_id:
            save(destination / 'review-cases' / (row['case_id'] + '.json'),
                {'result': answer, 'review': review_by_id[row['case_id']],
                 'input': case, 'raw_file_sha256': manifest[f'{row["index"]:04d}.json']})
    save(destination / 'ROWS-COMPACT.json', compact)
    save(destination / 'ANSWERS.json', answers)
    summary = {}
    for name, items in groups.items():
        diags = [r.get('diagnostics', {}) for r in items]
        summary[name] = {'n': len(items), 'http_completed': sum(r['completed'] for r in items),
            'http_statuses': dict(Counter(str(r.get('http_status', 'transport_error')) for r in items)),
            'generation_statuses': dict(Counter(d.get('generation_status', 'http_failed') for d in diags)),
            'no_sources': sum(d.get('source_count') == 0 for d in diags),
            'missing_numeric_citations': sum(not d.get('has_numeric_citation') for d in diags if d),
            'out_of_range_citations': sum(d.get('citation_out_of_range', False) for d in diags),
            'empty_answers': sum(d.get('empty_answer', False) for d in diags),
            'semantic_accuracy_verified': False}
    reviews = list(review_by_id.values())
    save(destination / 'SCOPE-SUMMARY.json', {'scopes': summary,
        'manual_review_n': len(reviews), 'manual_verdicts': dict(Counter(r['verdict'] for r in reviews)),
        'manual_by_suite': {suite: dict(Counter(r['verdict'] for r in reviews if cases[r['case_id']]['suite'] == suite))
            for suite in sorted({cases[r['case_id']]['suite'] for r in reviews})},
        'full_historical_retrieval_coverage_restored': False,
        'historical_reference_cases_with_missing_original_corpus': 400,
        'manual_review_independent': False, 'full_dataset_semantic_accuracy_verified': False})
    files = {str(p.relative_to(destination)): sha256(p.read_bytes()).hexdigest()
        for p in destination.rglob('*') if p.is_file()}
    save(destination / 'INTEGRITY.json', {'raw_calls': 478, 'archive_roundtrip_verified': True,
        'cases_sha256': sha256(CASES.read_bytes()).hexdigest(),
        'review_plan_sha256': sha256((ART / 'MANUAL-REVIEW-PLAN.json').read_bytes()).hexdigest(),
        'reviews_sha256': sha256((ART / 'MANUAL-REVIEWS.json').read_bytes()).hexdigest(),
        'files': files})
    print(json.dumps({'exported': 478, 'reviewed': 66, 'scopes': summary}, ensure_ascii=False))


if __name__ == '__main__':
    main()
