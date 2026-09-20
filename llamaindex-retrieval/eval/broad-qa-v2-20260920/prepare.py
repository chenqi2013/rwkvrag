"""Retain every historical question while declaring available-corpus boundaries."""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'broad-qa-20260920'
MANIFEST = Path('data/corpora/finewiki-zh-5000/manifest.jsonl')
KB = 'wiki-5000-20260908'


def main():
    old = json.loads((PREVIOUS / 'cases.json').read_text())
    rows = deepcopy(old)
    corpus = [json.loads(s) for s in MANIFEST.read_text().splitlines()]
    titles = {r['title'] for r in corpus}
    coverage = []
    for row in rows:
        c = row['original_case']
        if row['suite'] in ('wiki_verified', 'wiki_batch2'):
            assert c['reference'].strip()
            coverage.append({'id': row['id'], 'expected_title': c['expected_title'],
                'expected_document_id': c['expected_document_id'],
                'exact_title_in_current_corpus': c['expected_title'] in titles,
                'execution': 'fixed_historical_reference_not_retrieval'})
            row['endpoint'] = '/v1/material-ask'
            row['payload'] = {'question': c['question'], 'history': c.get('history', []),
                'materials': [{'id': row['id'] + '-reference',
                    'document_id': c['expected_document_id'], 'title': c['expected_title'],
                    'source': 'retained-historical-reference', 'snippet': c['reference'],
                    'score': 1.0, 'metadata': {'reference_only': True,
                        'origin': row['origin'], 'origin_index': row['origin_index']}}]}
            row['execution_scope'] = 'fixed_historical_reference'
        elif row['endpoint'] == '/v1/ask':
            row['payload']['knowledge_base_id'] = KB
            row['execution_scope'] = 'current_corpus_retrieval'
        else:
            row['execution_scope'] = 'original_fixed_material'
    # Read-only validation; no corpus or production settings are changed.
    query = {'query': {'term': {'knowledge_base_id': KB}}}
    req = urllib.request.Request('http://127.0.0.1:18438/rwkvrag-local-use-v1/_count',
        data=json.dumps(query).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=15) as response:
        count = json.load(response)
    assert count['count'] > 0
    origins = json.loads((PREVIOUS / 'ORIGINS.json').read_text())
    for p in (PREVIOUS / 'cases.json', MANIFEST):
        origins[str(p)] = sha256(p.read_bytes()).hexdigest()
    assert len(rows) == 478 and [r['id'] for r in rows] == [r['id'] for r in old]
    assert all(a['original_case'] == b['original_case'] for a, b in zip(rows, old))
    proof = {'knowledge_base_id': KB, 'scoped_chunk_count': count,
        'manifest_rows': len(corpus), 'manifest_sha256': origins[str(MANIFEST)],
        'historical_wiki': coverage,
        'scope_counts': dict(Counter(r['execution_scope'] for r in rows)),
        'historical_wiki_title_present': sum(r['exact_title_in_current_corpus'] for r in coverage),
        'limitation': 'Exact-title corpus check, not semantic retrieval. No missing old question is removed.'}
    for name, value in [('cases.json', rows), ('ORIGINS.json', origins), ('CORPUS-PREFLIGHT.json', proof)]:
        with (HERE / name).open('x') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
    print(json.dumps(proof['scope_counts']))


if __name__ == '__main__':
    main()
