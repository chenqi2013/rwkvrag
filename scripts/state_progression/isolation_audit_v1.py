"""Fail-closed comparison of generated jobs with historical evaluation questions and evidence.

This is a leakage screen, not a proof of semantic independence. It never edits raw data.
"""
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[2]


def norm(value):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', value).casefold())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def shingles(value, size=8):
    value = norm(value)
    return {value[i:i + size] for i in range(max(0, len(value) - size + 1))}


def historical():
    path = ROOT / 'data/statetune/progression-evaluation-20260922/historical-v1/cases.jsonl'
    if not path.exists():
        raise FileNotFoundError('Freeze historical evaluation cases first: ' + str(path))
    cases = [json.loads(line) for line in path.read_text().splitlines()]
    questions, sources = [], []
    for case in cases:
        question = case.get('question')
        if isinstance(question, str) and len(norm(question)) >= 8:
            questions.append((case['id'], norm(question)))
        for source in case.get('evidence', []):
            if not isinstance(source, dict):
                continue
            title = source.get('title')
            raw_text = source.get('text') or source.get('snippet')
            if isinstance(title, str) or isinstance(raw_text, str):
                sources.append((case['id'], norm(title or ''), norm(raw_text or '')))
    return path, cases, questions, sources


def main(args):
    historic_path, cases, questions, historic_sources = historical()
    by_title = defaultdict(set)
    # A long literal span detects reused evidence even when its displayed title changed.
    span_index = defaultdict(set)
    for case_id, title, value in historic_sources:
        if len(title) >= 3:
            by_title[title].add(case_id)
        for offset in range(0, max(0, len(value) - 47), 24):
            span_index[value[offset:offset + 48]].add(case_id)
    by_shingle = defaultdict(set)
    for index, (_, question) in enumerate(questions):
        for token in shingles(question):
            by_shingle[token].add(index)

    results = []
    for root in args.input:
        results += sorted(root.glob('*/results/*.json'))
    if not results:
        raise ValueError('No generated results to screen')
    excluded = {}
    findings = []
    scanned = Counter()
    seen_jobs = set()
    for path in results:
        raw = path.read_bytes()
        try:
            job = json.loads(raw)
        except json.JSONDecodeError:
            scanned['incomplete_result_files'] += 1
            continue
        identity = job['job_id']
        if identity in seen_jobs:
            raise ValueError('Duplicate job identity: ' + identity)
        seen_jobs.add(identity)
        if not job.get('accepted'):
            scanned['jobs_without_candidates'] += 1
            continue
        scanned['jobs_with_candidates'] += 1
        hits = []
        for source in job.get('sources', []):
            title, value = norm(source.get('title', '')), norm(source.get('text', ''))
            if title in by_title:
                hits.append(('source_title', source['id'], sorted(by_title[title])[:5]))
            sampled = set()
            for offset in range(0, max(0, len(value) - 47), 12):
                sampled.update(span_index.get(value[offset:offset + 48], ()))
            if sampled:
                hits.append(('evidence_span_48', source['id'], sorted(sampled)[:5]))
        for item in job['accepted']:
            question = item.get('prompt_body', '')
            raw_item = job.get('draft', {}).get('items', [])[item['index']]
            question = raw_item.get('question') or raw_item.get('field', {}).get('question') or ''
            value = norm(question)
            if len(value) < 8:
                continue
            candidates = Counter()
            for token in shingles(value):
                for index in by_shingle.get(token, ()):
                    candidates[index] += 1
            for index, overlap in candidates.most_common(12):
                old_id, old_value = questions[index]
                if overlap < 4:
                    continue
                similarity = SequenceMatcher(None, value, old_value, autojunk=False).ratio()
                if value == old_value or (min(len(value), len(old_value)) >= 14 and similarity >= 0.86):
                    hits.append(('question_near', item['id'], [old_id, round(similarity, 4)]))
        if hits:
            # Any reused source can affect all targets in the batch, so quarantine the whole job.
            for item in job['accepted']:
                excluded[item['id']] = 'historical_evaluation_overlap_job:' + hits[0][0]
            findings.append({'job_id': identity, 'split': job['split'], 'input_sha256': digest(raw),
                             'excluded_items': len(job['accepted']), 'hits': hits[:30]})
        scanned['candidate_items'] += len(job['accepted'])
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'EXCLUSIONS.json').write_text(json.dumps(excluded, ensure_ascii=False, indent=2) + '\n')
    (args.out / 'FINDINGS.json').write_text(json.dumps(findings, ensure_ascii=False, indent=2) + '\n')
    report = {'historical_cases': len(cases), 'historical_cases_sha256': digest(historic_path.read_bytes()),
              'historic_questions': len(questions), 'historic_evidence_records': len(historic_sources),
              'input_roots': [str(x) for x in args.input], 'scanned': dict(scanned),
              'quarantined_jobs': len(findings), 'quarantined_items': len(excluded),
              'screen': 'normalized title; sampled 48-character literal evidence spans; exact or >=0.86 question similarity',
              'limits': 'This screen cannot prove semantic independence; all historic cases remain seen regression only.'}
    (args.out / 'SUMMARY.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
