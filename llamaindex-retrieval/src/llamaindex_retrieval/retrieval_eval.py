"""Frozen-case lean evaluation with source-bound metrics and explicit semantic reviews.

Run with ``python -m llamaindex_retrieval.retrieval_eval --help``. Research inputs
and output directories are supplied by the caller; this module ships no cases or
golds. Semantic correctness is never inferred from string overlap or HTTP success.
"""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
import shutil
from statistics import median
from uuid import uuid4

from .case_pool import CasePool
from .interactive_spans import Reader
from .schemas import SourceItem


STAGES = {'plan', 'navigation', 'page_selection', 'reading', 'answering', 'acquisition', 'runtime'}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def load_cases(paths):
    cases = [case for path in paths for case in json.loads(Path(path).read_text())['cases']]
    ids = [case['id'] for case in cases]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('evaluation needs distinct case IDs')
    return cases


@lru_cache(maxsize=256)
def _coordinates(text, kind=None):
    source = SourceItem(id='canonical', document_id='canonical', title='',
                        source='knowledge_base', score=0, snippet=text, metadata={'kind': kind})
    reader = Reader(None, '', [], lambda _: None)
    for _ in reader.windows(source):
        pass
    return {sid: (row['span']['start'], row['span']['end']) for sid, row in reader.spans.items()}


def canonical_truth(gold, sources):
    """Labels bind source ID + original offsets, never a globally reused S number."""
    grades = {}
    positives = set()
    for need in gold['needs']:
        for source_id, labels in need['span_grades'].items():
            if source_id not in sources:
                continue
            source = sources[source_id]
            coords = _coordinates(source['snippet'], source.get('metadata', {}).get('kind'))
            for sid, pair in coords.items():
                grade = labels['overrides'].get(sid, labels.get('default'))
                if not labels.get('source_admissible', True):
                    # Provenance rejection is independent of unreviewed content.
                    grade = 0
                elif not labels.get('reviewed_all_spans') and sid not in labels['overrides']:
                    grade = None
                key = (source_id, *pair)
                grades.setdefault(key, []).append(grade)
                if grade is not None and grade >= 2:
                    positives.add(key)
    result = {}
    for key, values in grades.items():
        known = [v for v in values if v is not None]
        result[key] = max(known) if known and (max(known) >= 2 or len(known) == len(values)) else None
    return result, positives


def validate_review(review, record, gold, record_sha256):
    if review['case_id'] != gold['case_id'] or review['record_sha256'] != record_sha256:
        raise ValueError('semantic review is bound to another raw record')
    if review['rubric'] not in {'S', 'W', 'F'} or type(review['facts_after_not_found']) is not bool:
        raise ValueError('invalid final rubric or fabrication review')
    rows = review['need_results']
    ids = [r['gold_need_id'] for r in rows]
    if len(set(ids)) != len(ids) or set(ids) != {n['id'] for n in gold['needs']}:
        raise ValueError('semantic review must cover every gold need exactly once')
    for row in rows:
        if (type(row['correct']) is not bool or type(row['not_found']) is not bool
                or (row['citations_supported'] is not None and type(row['citations_supported']) is not bool)
                or not isinstance(row['reason'], str) or not row['reason'].strip()):
            raise ValueError('invalid per-need semantic review')
    failure = review.get('first_failure')
    if failure is not None and (failure['stage'] not in STAGES or not failure['category']
                                or not failure['evidence']):
        raise ValueError('first failure needs stage, category and raw/source evidence')
    if review['rubric'] != 'S' and failure is None:
        raise ValueError('failed final answer needs its first failure classification')
    return review


def score_record(record, gold, *, record_sha256, review=None):
    result = record.get('result') or {}
    graph = result.get('graph') or {}
    sources = {s['id']: s for s in graph.get('source_snapshots', [])}
    spans = {s['id']: s for s in graph.get('reading_spans', graph.get('spans', []))}
    grades, positives = canonical_truth(gold, sources)
    nodes = graph.get('nodes', [])
    reads = [n for n in nodes if n.get('kind') == 'score']
    invalid_bindings = set()

    def coordinate(sid):
        row = spans.get(sid)
        if row is None:
            invalid_bindings.add(sid)
            return None
        source = sources.get(row['source_id'])
        start, end = row['span']['start'], row['span']['end']
        if (source is None or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(source['snippet'])
                or source['snippet'][start:end] != row['text']):
            invalid_bindings.add(sid)
            return None
        return row['source_id'], start, end

    def span_grade(key):
        if key in grades:
            return grades[key]
        # A navigation gold ID is a known positive, not an exhaustive semantic
        # judgment about every other document in the corpus.
        return None

    offered, selected = set(), set()
    per_call = []
    for read in reads:
        offered_keys = {coordinate(sid) for sid in read.get('offered_span_ids', [])} - {None}
        chosen = {coordinate(row['span_id']) for row in (read.get('parsed') or {}).get('scores', [])} - {None}
        offered |= offered_keys
        selected |= chosen
        per_call.append({'status': read['status'], 'offered': len(offered_keys), 'selected': len(chosen),
                         'false_positives': sum(span_grade(k) in (0, 1) for k in chosen),
                         'unknown_selected': sum(span_grade(k) is None for k in chosen),
                         'positive_offered': len(positives & offered_keys),
                         'positive_selected': len(positives & chosen)})
    final = {coordinate(s['id']) for s in graph.get('spans', [])} - {None}
    picked = list(dict.fromkeys(fid for r in result.get('acquisition', [])
                               if r.get('channel') == 'knowledge_base'
                               for fid in r.get('selected_file_ids', [])))
    # Web source snapshots include the whole candidate pool. Read/outline nodes
    # bind actual selected pages; candidate acquisition alone is not a page hit.
    selected_sources = {sid for n in nodes if n.get('kind') == 'page_selection'
                        for sid in n.get('selected_source_ids', [])}
    read_sources = {n['source_id'] for n in reads if n.get('source_id')}
    page_ids = {s['uri'] for s in sources.values() if s['id'] in selected_sources and s.get('uri')}
    read_pages = {s['uri'] for s in sources.values() if s['id'] in read_sources and s.get('uri')}
    gold_files, gold_pages = set(gold['gold_file_ids']), set(gold['gold_pages'])
    nav_events = [n for n in nodes if n.get('kind', '').startswith('navigation_')]
    nav_start = min((n.get('started_ms', 0) for n in nav_events), default=0)
    nav_end = max((n.get('started_ms', 0)+n.get('elapsed_ms', 0) for n in nav_events), default=0)
    navigation = {'gold_files': sorted(gold_files), 'picked_file_ids': picked,
                  'gold_in_top3': bool(gold_files & set(picked[:3])) if gold_files else None,
                  'required_file_recall': len(gold_files & set(picked))/len(gold_files) if gold_files else None,
                  'calls': len(nav_events),
                  'wall_ms': nav_end-nav_start,
                  'active_call_ms': sum(n.get('elapsed_ms', 0) for n in nav_events),
                  'paths': [r['path'] for r in result.get('acquisition', []) if r.get('mode') == 'entity_index']}
    # Denominator includes frozen positives from unselected original sources too.
    reading = {'gold_positive_spans': len(positives), 'offered_gold_spans': len(positives & offered),
               'selected_gold_spans': len(positives & selected), 'final_gold_spans': len(positives & final),
               'offered_recall': len(positives & offered)/len(positives) if positives else None,
               'selected_recall': len(positives & selected)/len(positives) if positives else None,
               'false_positives': sum(r['false_positives'] for r in per_call),
               'unknown_selected': sum(r['unknown_selected'] for r in per_call),
               'invalid_binding_ids': sorted(invalid_bindings),
               'calls': len(reads), 'per_call': per_call}
    scored = {'case_id': gold['case_id'], 'record_sha256': record_sha256,
              'response_type': result.get('response_type', 'error'), 'navigation': navigation,
              'pages': {'gold_pages': sorted(gold_pages), 'picked_pages': sorted(page_ids),
                        'read_pages': sorted(read_pages),
                        'gold_page_pick_recall': len(gold_pages & page_ids)/len(gold_pages) if gold_pages else None,
                        'gold_page_read_recall': len(gold_pages & read_pages)/len(gold_pages) if gold_pages else None},
              'reading': reading, 'telemetry': result.get('telemetry', {}),
              'semantic_review': None, 'semantic_status': 'pending'}
    if review is not None:
        scored['semantic_review'] = validate_review(review, record, gold, record_sha256)
        scored['semantic_status'] = 'reviewed'
    return scored


def score_run(run, gold_path, out, review_path=None):
    """New scores never overwrite immutable raw records or earlier judgments."""
    run, out = Path(run), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out/'PLAN.md').write_text('Score frozen raw records with source-bound stage golds. '
        'Question-level reading relevance uses max gold grade across atomic needs. '
        'Unknown grades are excluded from negative counts and explicitly reported. '
        'Semantic reviews bind raw SHA and are never replaced by lexical matching.\n')
    golds = {g['case_id']: g for g in json.loads(Path(gold_path).read_text())['cases']}
    review_rows = json.loads(Path(review_path).read_text())['cases'] if review_path else []
    reviews = {r['case_id']: r for r in review_rows}
    if len(reviews) != len(review_rows):
        raise ValueError('duplicate semantic review case ID')
    manifest = json.loads((run/'RUN.json').read_text())
    rows = []
    for item in manifest['records']:
        path = run/'records'/(item['job_id']+'.json')
        if digest(path) != item['sha256']:
            raise ValueError('raw record changed after freezing')
        record = json.loads(path.read_text())
        gold = golds[item['case_id']]
        # Load only truth-referenced originals; the full pool is never passed to
        # semantic scoring as if it were the model's selected evidence.
        pool = CasePool(manifest['pool']).load(item['case_id'])
        if pool.sha256 != gold['pool_sha256']:
            raise ValueError('golds are bound to a different case pool')
        actual_sources = {s['id']: s for s in record.get('result', {}).get('graph', {}).get('source_snapshots', [])}
        truth_sources = {}
        wanted = {sid for n in gold['needs'] for sid, labels in n['span_grades'].items()
                  if labels.get('source_admissible')}
        for fid in gold['gold_file_ids']:
            s = pool.document(fid)
            truth_sources[s.id] = s.model_dump(mode='json')
        for hit in pool.web()['hits']:
            sid = 'W' + sha256((hit['url']+'\0'+hit['reader_text']).encode()).hexdigest()[:16]
            if sid in wanted:
                truth_sources[sid] = {'id': sid, 'snippet': hit['reader_text'], 'uri': hit['url'],
                                      'source': 'web', 'document_id': None}
        required = {sid for n in gold['needs'] for sid, labels in n['span_grades'].items()
                    if any(grade >= 2 for grade in labels['overrides'].values())}
        if not required <= truth_sources.keys() | actual_sources.keys():
            raise ValueError('frozen positive source missing from scoring denominator')
        if any(actual_sources[sid]['snippet'] != source['snippet'] for sid, source in truth_sources.items()
               if sid in actual_sources):
            raise ValueError('candidate snapshot conflicts with frozen truth source identity')
        # Canonical metrics require truth sources in their denominator even when
        # navigation missed them; keep this local to scoring, outside raw records.
        extra = [s for sid, s in truth_sources.items() if sid not in actual_sources]
        augmented = json.loads(json.dumps(record))
        if not augmented.get('result'):
            augmented['result'] = {}
        augmented.setdefault('result', {}).setdefault('graph', {}).setdefault('source_snapshots', []).extend(extra)
        rows.append(score_record(augmented, gold, record_sha256=item['sha256'], review=reviews.get(item['case_id'])))
    lat = sorted(r['telemetry'].get('total_ms', 0)/1000 for r in rows)
    reviewed = [r['semantic_review'] for r in rows if r['semantic_review']]
    needs = [n for r in reviewed for n in r['need_results']]
    expected_nf = [(n, truth) for r in reviewed for n in r['need_results']
                   for truth in golds[r['case_id']]['needs'] if n['gold_need_id'] == truth['id']
                   and not truth['answerable']]
    defects = Counter((r['first_failure']['stage'], r['first_failure']['category'])
                      for r in reviewed if r.get('first_failure'))
    summary = {'population': len(rows), 'raw_complete': manifest['complete'],
               'expected_population': manifest.get('expected_population', len(rows)),
               'missing_case_ids': [cid for cid in manifest.get('expected_case_ids', [])
                                    if cid not in {r['case_id'] for r in rows}],
               'semantic_reviewed': len(reviewed), 'semantic_pending': len(rows)-len(reviewed),
               'rubric_counts': dict(Counter(r['rubric'] for r in reviewed)),
               'facts_after_not_found': sum(r['facts_after_not_found'] for r in reviewed) if reviewed else None,
               'need_correct': sum(n['correct'] for n in needs) if needs else None,
               'need_reviewed': len(needs),
               'not_found_correct': sum(n['correct'] and n['not_found'] for n, _ in expected_nf) if reviewed else None,
               'not_found_reviewed': len(expected_nf),
               'citations_supported': sum(n['citations_supported'] is True for n in needs) if needs else None,
               'citations_reviewed': sum(n['citations_supported'] is not None for n in needs),
               'response_types': dict(Counter(r['response_type'] for r in rows)),
               'latency_p50_seconds': median(lat) if lat else None,
               'latency_p95_seconds': lat[ceil(.95*len(lat))-1] if lat else None,
               'adoption': 'not_decided_by_harness', 'cases': rows,
               'bindings': {'run': str(run.resolve()), 'run_sha256': digest(run/'RUN.json'),
                            'golds_sha256': digest(gold_path),
                            'reviews_sha256': digest(review_path) if review_path else None}}
    save(out/'SCORED.json', summary)
    save(out/'DEFECTS.json', {'reviewed': len(reviewed), 'pending': len(rows)-len(reviewed),
        'ranked': [{'stage': s, 'category': c, 'cases': count} for (s, c), count in defects.most_common()],
        'cases': [{'case_id': r['case_id'], **r['first_failure']} for r in reviewed if r.get('first_failure')]})
    lines = ['| Case | Response | File top3 | Gold spans selected/total | Reader FP/unknown | Final |',
             '|---|---|---|---|---|---|']
    for r in rows:
        rd = r['reading']
        lines.append(f"| {r['case_id']} | {r['response_type']} | {r['navigation']['gold_in_top3']} | "
                     f"{rd['selected_gold_spans']}/{rd['gold_positive_spans']} | "
                     f"{rd['false_positives']}/{rd['unknown_selected']} | "
                     f"{(r['semantic_review'] or {}).get('rubric', 'pending')} |")
    (out/'SCORED.md').write_text('\n'.join(lines)+'\n\nSemantic pending means unscored, not correct. '
        'Reading scores are source-bound question-level relevance; answer/citation correctness requires review.\n')
    return summary


async def run(config_path, case_paths, gold_path, out):
    from .interactive_app import Question, Trial
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out/'records').mkdir()
    (out/'PLAN.md').write_text('Run every supplied dev case once in order, on frozen case pools. '
        'Lean runtime factories, configured deadline/budgets, no retries or live web. '
        'Errors remain in population. Preserve source/config/case/pool bindings before generation. '
        'Score mechanical stages separately from pending offline semantic judgments. No blind input.\n')
    config = json.loads(Path(config_path).read_text())
    if config.get('pipeline') != 'lean' or config.get('web_replay', {}).get('mode') != 'case_pool':
        raise ValueError('stage evaluation requires lean + case_pool; live acquisition forbidden')
    pool = CasePool(config['web_replay']['store'])
    cases = load_cases(case_paths)
    for case in cases:
        pool.load(case['id'])
    src = Path(__file__).parent
    shutil.copytree(src, out/'source', ignore=shutil.ignore_patterns('__pycache__', 'static'))
    source_hashes = {p.name: digest(p) for p in src.glob('*.py')}
    index_hashes = {str(Path(p).resolve()): digest(p) for field in ('entity_indexes', 'kb_cards')
                    for p in config.get(field, {}).values()}
    save(out/'BINDINGS.json', {'config_sha256': digest(config_path), 'source_hashes': source_hashes,
         'case_sha256': {str(p): digest(p) for p in case_paths}, 'gold_sha256': digest(gold_path),
         'pool_sha256': {c['id']: pool.load(c['id']).sha256 for c in cases},
         'index_sha256': index_hashes,
         'started_at': datetime.now(timezone.utc).isoformat()})
    config['records_dir'] = str((out/'records').resolve())
    trial = Trial(config)
    records, complete = [], False
    try:
        await trial.native.verify_model_identity()
        if trial.settings.native_require_fp32_runtime:
            await trial.native.verify_fp32_runtime()
        for case in cases:
            request = case['request']
            parent = None
            if request.get('history'):
                parent = uuid4().hex
                trial.jobs[parent] = {'status': 'completed', 'question': request['history'][-2]['content'],
                    'result': {'display_text': request['history'][-1]['content']}}
            kb = request.get('knowledge_base_id')
            payload = Question(question=request['question'], parent_job_id=parent,
                retrieval_mode='hybrid' if kb else 'web', knowledge_base_ids=[kb] if kb else [],
                pipeline='lean', case_id=case['id'])
            job = {'id': uuid4().hex, 'case_id': case['id'], 'question': request['question'],
                   'request': payload.model_dump(), 'status': 'queued'}
            await trial.run(job, payload)
            path = out/'records'/(job['id']+'.json')
            records.append({'case_id': case['id'], 'job_id': job['id'], 'sha256': digest(path)})
            print(case['id'], job.get('result', {}).get('response_type'), flush=True)
            raw = json.loads(path.read_text())
            codes = [h.get('http_status') for e in raw['model_calls']
                     for h in (e.get('trace') or {}).get('http', [])]
            if any(c in {401, 402, 403, 429} for c in codes):
                raise RuntimeError('model authentication/quota/rate failure; stop without retry')
            if any(digest(src/name) != h for name, h in source_hashes.items()):
                raise RuntimeError('source changed during frozen run')
            if any(digest(path) != h for path, h in index_hashes.items()):
                raise RuntimeError('index changed during frozen run')
        complete = True
    finally:
        await trial.native.aclose()
        await trial.admin.aclose()
        save(out/'RUN.json', {'complete': complete, 'records': records,
                             'pool': str(pool.store.resolve()), 'expected_population': len(cases),
                             'expected_case_ids': [c['id'] for c in cases]})
    return score_run(out, gold_path, out/'scores')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    execute = commands.add_parser('run')
    execute.add_argument('--config', type=Path, required=True)
    execute.add_argument('--cases', type=Path, nargs='+', required=True)
    execute.add_argument('--golds', type=Path, required=True)
    execute.add_argument('--out', type=Path, required=True)
    score = commands.add_parser('score')
    score.add_argument('--run', type=Path, required=True)
    score.add_argument('--golds', type=Path, required=True)
    score.add_argument('--reviews', type=Path)
    score.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        asyncio.run(run(args.config, args.cases, args.golds, args.out))
    else:
        score_run(args.run, args.golds, args.out, args.reviews)


if __name__ == '__main__':
    main()
