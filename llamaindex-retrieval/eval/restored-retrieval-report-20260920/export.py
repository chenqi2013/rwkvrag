"""Export the complete frozen retrieval run and verify evidence bytes."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'data/quality-runs/restored-retrieval-v2-20260920/run1'
ART = ROOT / 'artifacts/broad-regression-20260920'
SUITE = ROOT / 'llamaindex-retrieval/eval/restored-retrieval-v2-20260920'
CORPUS = ROOT / 'data/corpora/finewiki-restored-20260920'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    rows = json.loads((RUN / 'ROWS.json').read_text())
    cases = {c['id']: c for c in json.loads((SUITE / 'cases.json').read_text())}
    assert len(rows) == len(cases) == 434
    assert {r['case_id'] for r in rows} == set(cases)
    reviews = json.loads((ART / 'RESTORED-MANUAL-REVIEWS.json').read_text())['items']
    review_by_id = {r['case_id']: r for r in reviews}
    plan = json.loads((SUITE / 'REVIEW-PLAN.json').read_text())
    assert set(review_by_id) == set(plan['case_ids']) and len(reviews) == 48
    destination = ART / 'restored-retrieval-v2'
    destination.mkdir(exist_ok=False)
    for path in RUN.iterdir():
        if path.is_file() and path.name != 'ROWS.json':
            shutil.copy2(path, destination / path.name)
    shutil.copy2('/tmp/restored-retrieval-v2-run1.log', destination / 'execution.log')
    manifest = {}
    with tarfile.open(destination / 'raw-calls.tar.gz', 'w:gz') as archive:
        for path in sorted((RUN / 'calls').glob('*.json')):
            manifest[path.name] = sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname='calls/' + path.name)
    assert len(manifest) == 434
    with tarfile.open(destination / 'raw-calls.tar.gz') as archive:
        for member in archive:
            assert sha256(archive.extractfile(member).read()).hexdigest() == manifest[Path(member.name).name]
    save(destination / 'RAW-MANIFEST.json', manifest)
    compact, answers, span_errors = [], [], []
    grouped = {key: defaultdict(list) for key in ('suite', 'category', 'style')}
    model_calls = Counter()
    source_checks = 0
    text_cache = {}
    (destination / 'review-cases').mkdir()
    for row in rows:
        case = cases[row['case_id']]
        response = json.loads(row.get('raw_response') or '{}')
        sources = response.get('sources', [])
        generation = response.get('generation', {})
        record = {k: v for k, v in row.items() if k != 'raw_response'}
        record['raw_file'] = f'calls/{row["index"]:04d}.json'
        compact.append(record)
        answer = {'case_id': row['case_id'], 'suite': row['suite'],
            'question': case['payload']['question'], 'http_status': row.get('http_status'),
            'error': row.get('error'), 'answer': response.get('answer'), 'sources': sources,
            'status': generation.get('status'), 'answer_span': generation.get('answer_span'),
            'answer_modified': generation.get('answer_modified'),
            'diagnostics': row.get('diagnostics'), 'raw_file': record['raw_file']}
        answers.append(answer)
        for key in grouped:
            grouped[key][row[key]].append(row)
        for call in generation.get('model_calls', []):
            model_calls[(call.get('stage', 'unspecified'), call.get('status', 'unspecified'))] += 1
        for source in sources:
            source_checks += 1
            try:
                meta = source['metadata']
                path = Path(meta['file']).resolve()
                assert path.is_relative_to(CORPUS)
                if path not in text_cache:
                    text_cache[path] = path.read_text()
                text = text_cache[path]
                assert sha256(text.encode()).hexdigest() == meta['source_text_sha256']
                span = meta['source_span']
                parent = text[span['start']:span['end']]
                assert sha256(parent.encode()).hexdigest() == span['sha256']
                selected = parent[meta['span_start']:meta['span_end']]
                assert selected == source['snippet']
                assert sha256(selected.encode()).hexdigest() == meta['span_sha256']
                assert meta['knowledge_base_id'] == 'wiki-restored-20260920'
            except Exception as error:
                span_errors.append({'case_id': row['case_id'], 'source_id': source.get('id'),
                    'error_type': type(error).__name__, 'error': str(error)})
        if row['case_id'] in review_by_id:
            save(destination / 'review-cases' / (row['case_id'] + '.json'),
                {'result': answer, 'review': review_by_id[row['case_id']], 'input': case,
                 'raw_file_sha256': manifest[f'{row["index"]:04d}.json']})
    save(destination / 'ROWS-COMPACT.json', compact)
    save(destination / 'ANSWERS.json', answers)
    summary = {}
    for key, groups in grouped.items():
        summary[key] = {}
        for name, items in groups.items():
            diags = [r.get('diagnostics', {}) for r in items]
            summary[key][name] = {'n': len(items), 'http_completed': sum(r['completed'] for r in items),
                'http_statuses': dict(Counter(str(r.get('http_status', 'transport_error')) for r in items)),
                'generation_statuses': dict(Counter(d.get('generation_status', 'http_failed') for d in diags)),
                'no_sources': sum(d.get('source_count') == 0 for d in diags),
                'expected_article_in_final_sources': sum(d.get('restored_expected_document_returned', False) for d in diags),
                'missing_numeric_citations': sum(not d.get('has_numeric_citation') for d in diags if d),
                'out_of_range_citations': sum(d.get('citation_out_of_range', False) for d in diags),
                'empty_answers': sum(d.get('empty_answer', False) for d in diags),
                'semantic_accuracy_verified': False}
    save(destination / 'ANALYSIS.json', {'groups': summary,
        'model_call_counts': [{'stage': k[0], 'status': k[1], 'n': v} for k, v in sorted(model_calls.items())],
        'manual_review_n': len(reviews), 'manual_verdicts': dict(Counter(r['verdict'] for r in reviews)),
        'manual_by_suite': {suite: dict(Counter(r['verdict'] for r in reviews if cases[r['case_id']]['suite'] == suite))
            for suite in sorted({cases[r['case_id']]['suite'] for r in reviews})},
        'manual_review_independent': False, 'full_dataset_semantic_accuracy_verified': False,
        'corpus_boundary': 'New versioned full-article corpus; not the deleted original index.'})
    save(destination / 'SOURCE-SPAN-INTEGRITY.json', {'returned_sources_checked': source_checks,
        'all_returned_source_bytes_match_originals': not span_errors, 'errors': span_errors,
        'semantic_entailment_verified_by_this_check': False})
    files = {str(p.relative_to(destination)): sha256(p.read_bytes()).hexdigest()
        for p in destination.rglob('*') if p.is_file()}
    save(destination / 'INTEGRITY.json', {'raw_calls': 434, 'archive_roundtrip_verified': True,
        'cases_sha256': sha256((SUITE / 'cases.json').read_bytes()).hexdigest(),
        'review_plan_sha256': sha256((SUITE / 'REVIEW-PLAN.json').read_bytes()).hexdigest(),
        'reviews_sha256': sha256((ART / 'RESTORED-MANUAL-REVIEWS.json').read_bytes()).hexdigest(),
        'files': files})
    print(json.dumps({'exported': len(rows), 'reviewed': len(reviews), 'sources_checked': source_checks,
        'source_byte_errors': len(span_errors)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
