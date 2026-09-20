import json
from pathlib import Path


def test_corpus_gap_is_explicit_and_no_question_or_reference_is_removed():
    root = Path(__file__).parents[1] / 'eval'
    old = json.loads((root / 'broad-qa-20260920/cases.json').read_text())
    new = json.loads((root / 'broad-qa-v2-20260920/cases.json').read_text())
    assert len(new) == len(old) == 478
    for a, b in zip(old, new):
        assert a['id'] == b['id'] and a['original_case'] == b['original_case']
        assert a['payload']['question'] == b['payload']['question']
        assert a['payload']['history'] == b['payload']['history']
        if b['execution_scope'] == 'fixed_historical_reference':
            assert b['payload']['materials'][0]['snippet'] == a['original_case']['reference']
        elif b['execution_scope'] == 'current_corpus_retrieval':
            assert b['payload']['knowledge_base_id'] == 'wiki-5000-20260908'
        else:
            assert a['payload'] == b['payload']
    assert sum(r['endpoint'] == '/v1/ask' for r in new) == 34
    assert sum(r['endpoint'] == '/v1/material-ask' and not r['payload']['materials'] for r in new) == 3
    proof = json.loads((root / 'broad-qa-v2-20260920/CORPUS-PREFLIGHT.json').read_text())
    assert proof['scoped_chunk_count']['count'] > 0
    assert len(proof['historical_wiki']) == 400
    assert proof['historical_wiki_title_present'] == 0
