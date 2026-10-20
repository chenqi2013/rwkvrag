from copy import deepcopy
import json

import pytest

from llamaindex_retrieval.case_pool import CasePool
from llamaindex_retrieval.retrieval_eval import digest, score_record, score_run, validate_review
from llamaindex_retrieval.schemas import SourceItem


def source(sid, fid, text='Original complete fact.'):
    return SourceItem(id=sid, document_id=fid, title=fid, source='knowledge_base',
                      score=0, snippet=text).model_dump(mode='json')


def fixture():
    a, b = source('positive', 'file-a'), source('negative', 'file-b')
    gold = {'case_id': 'case', 'gold_file_ids': ['file-a'], 'gold_pages': [], 'needs': [
        {'id': 'G1', 'answerable': True, 'span_grades': {
            'positive': {'source_admissible': True, 'reviewed_all_spans': True,
                         'default': 0, 'overrides': {'S1': 3}},
            'negative': {'source_admissible': False, 'reviewed_all_spans': False,
                         'default': None, 'overrides': {}}}}]}
    spans = [{'id': sid, 'source_id': src['id'], 'text': src['snippet'],
              'span': {'start': 0, 'end': len(src['snippet'])}} for sid, src in [('S1', a), ('S2', b)]]
    read = {'kind': 'score', 'status': 'completed', 'source_id': 'negative',
            'offered_span_ids': ['S2'], 'parsed': {'scores': [{'span_id': 'S2', 'need_id': 'I1', 'score': '3'}]}}
    record = {'result': {'response_type': 'answered', 'graph': {'source_snapshots': [a, b],
        'reading_spans': spans, 'spans': [spans[1]], 'nodes': [read]}, 'acquisition': [
            {'channel': 'knowledge_base', 'selected_file_ids': ['file-b']}], 'telemetry': {'total_ms': 2000}}}
    return gold, record


def test_source_local_s1_never_matches_another_document():
    gold, record = fixture()
    scored = score_record(record, gold, record_sha256='raw')
    assert scored['navigation']['gold_in_top3'] is False
    assert scored['reading']['gold_positive_spans'] == 1
    assert scored['reading']['selected_gold_spans'] == 0
    assert scored['reading']['selected_recall'] == 0
    assert scored['reading']['false_positives'] == 1
    assert scored['semantic_status'] == 'pending'


def test_unknown_content_is_not_an_implicit_negative():
    gold, record = fixture()
    gold['gold_file_ids'] = []
    gold['needs'][0]['span_grades']['negative']['source_admissible'] = True
    scored = score_record(record, gold, record_sha256='raw')
    assert scored['reading']['false_positives'] == 0
    assert scored['reading']['unknown_selected'] == 1


def test_same_id_different_offsets_does_not_count_as_canonical_positive():
    gold, record = fixture()
    span = record['result']['graph']['reading_spans'][1]
    span.update(source_id='positive', span={'start': 2, 'end': 10})
    span['text'] = record['result']['graph']['source_snapshots'][0]['snippet'][2:10]
    scored = score_record(record, gold, record_sha256='raw')
    assert scored['reading']['selected_gold_spans'] == 0
    assert scored['reading']['unknown_selected'] == 1


def test_altered_span_text_is_an_invalid_binding_not_a_positive():
    gold, record = fixture()
    span = record['result']['graph']['reading_spans'][1]
    span.update(source_id='positive', text='Changed evidence.')
    scored = score_record(record, gold, record_sha256='raw')
    assert scored['reading']['selected_gold_spans'] == 0
    assert scored['reading']['invalid_binding_ids'] == ['S2']


def test_selected_page_and_read_page_are_distinct():
    gold, record = fixture()
    gold['gold_pages'] = ['https://example.org/original']
    record['result']['graph']['source_snapshots'][0].update(
        source='web', uri='https://example.org/original')
    record['result']['graph']['nodes'].append({'kind': 'page_selection',
        'selected_source_ids': ['positive']})
    pages = score_record(record, gold, record_sha256='raw')['pages']
    assert pages['gold_page_pick_recall'] == 1
    assert pages['gold_page_read_recall'] == 0


def test_semantic_review_must_bind_raw_and_every_need():
    gold, record = fixture()
    review = {'case_id': 'case', 'record_sha256': 'raw', 'rubric': 'F',
        'facts_after_not_found': False, 'first_failure': {'stage': 'navigation',
        'category': 'wrong_document', 'evidence': 'picked file-b, required file-a'},
        'need_results': [{'gold_need_id': 'G1', 'correct': False, 'not_found': False,
                          'citations_supported': False, 'reason': 'Wrong original.'}]}
    assert validate_review(review, record, gold, 'raw') is review
    wrong = deepcopy(review)
    wrong['record_sha256'] = 'changed'
    with pytest.raises(ValueError, match='another raw'):
        validate_review(wrong, record, gold, 'raw')
    wrong = deepcopy(review)
    wrong['need_results'] = []
    with pytest.raises(ValueError, match='every gold need'):
        validate_review(wrong, record, gold, 'raw')


def test_score_run_counts_missed_original_in_recall_denominator(tmp_path):
    gold, record = fixture()
    a, b = record['result']['graph']['source_snapshots']
    pool = CasePool(tmp_path/'pool')
    frozen = pool.freeze('case', catalog=[{'id': 'file-a'}, {'id': 'file-b'}],
                         documents={'file-a': a, 'file-b': b}, web_payload={'hits': []})
    gold['pool_sha256'] = frozen.sha256
    record['result']['graph']['source_snapshots'] = [b]
    run = tmp_path/'run'
    (run/'records').mkdir(parents=True)
    path = run/'records'/'job.json'
    path.write_text(json.dumps(record))
    (run/'RUN.json').write_text(json.dumps({'complete': True, 'pool': str(pool.store),
        'records': [{'case_id': 'case', 'job_id': 'job', 'sha256': digest(path)}]}))
    gold_path = tmp_path/'gold.json'
    gold_path.write_text(json.dumps({'cases': [gold]}))
    scored = score_run(run, gold_path, tmp_path/'scores')
    assert scored['cases'][0]['reading']['gold_positive_spans'] == 1
    assert scored['cases'][0]['reading']['selected_recall'] == 0
    assert scored['rubric_counts'] == {}
    assert scored['need_correct'] is None
    assert scored['semantic_pending'] == 1
    assert (tmp_path/'scores'/'SCORED.md').exists()
    path.write_text('{}')
    with pytest.raises(ValueError, match='raw record changed'):
        score_run(run, gold_path, tmp_path/'other-scores')
