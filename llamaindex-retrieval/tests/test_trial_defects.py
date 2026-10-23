import gzip
import hashlib
import json

import pytest

from llamaindex_retrieval import trial_defects


def test_capture_preserves_exact_record_and_separates_observations_from_semantic_labels(tmp_path):
    record = {'id': 'a'*32, 'status': 'completed', 'model_calls': [
        {'stage': 'plan', 'status': 'completed', 'messages': [{'content': 'plan\n{"message":"最近状态"}'}]},
        {'stage': 'writer', 'status': 'length', 'raw_text': '不完整原始输出 ',
         'messages': [{'content': 'final\n{"representation":"derived_readings_with_source_lineage"}'}]}],
        'result': {'errors': [{'stage': 'final', 'status': 'length'}]}}
    path = tmp_path / 'raw.json'
    raw = json.dumps(record, ensure_ascii=False).encode() + b'\n'
    path.write_bytes(raw)
    folder = trial_defects.capture_record(path, tmp_path / 'captures')
    assert gzip.decompress((folder / 'record.json.gz').read_bytes()) == raw
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['record_sha256'] == hashlib.sha256(raw).hexdigest()
    assert manifest['training_eligible'] is False and manifest['semantic_review'] == 'pending'
    assert {i['category'] for i in manifest['observations']} == {
        'model_execution', 'originals_not_forwarded', 'query_clock_not_supplied', 'pipeline_incomplete'}
    assert (folder / 'record.json.gz').stat().st_mode & 0o777 == 0o600
    assert trial_defects.capture_record(path, tmp_path / 'captures') == folder
    (folder / 'record.json.gz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        trial_defects.capture_record(path, tmp_path / 'captures')


def test_completed_execution_is_not_an_automatic_semantic_success_label():
    assert trial_defects.observations({'id': 'a'*32, 'status': 'completed', 'model_calls': []}) == []


async def test_capture_failure_preserves_completed_answer_and_records_capture_error(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from llamaindex_retrieval import interactive_app as api
    def fail(*args):
        raise PermissionError('fixture storage failure')
    monkeypatch.setattr(trial_defects, 'capture_record', fail)
    trial = SimpleNamespace(folder=tmp_path, config={'defects_dir': str(tmp_path / 'captures')})
    record = {'id': 'b'*32, 'status': 'completed', 'answer': 'raw output unchanged'}
    status = await api.persist_record(trial, record)
    assert status['diagnostic_capture'] == 'failed'
    assert json.loads((tmp_path / ('b'*32+'.json')).read_text()) == record
    assert (tmp_path / ('b'*32+'.capture-error.json')).is_file()


async def test_queued_cancellation_is_also_automatically_captured(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from llamaindex_retrieval import interactive_app as api
    async def never(*args):
        pytest.fail('cancelled job should never start')
    trial = SimpleNamespace(jobs={}, tasks={}, folder=tmp_path, run=never,
                            config={'defects_dir': str(tmp_path / 'captures')})
    monkeypatch.setattr(api.app.state, 'trial', trial, raising=False)
    job = await api.ask(api.Question(question='arbitrary fixture'))
    result = await api.cancel(job['id'])
    assert result['status'] == 'cancelled' and result['diagnostic_capture'] == 'saved'
    manifest = json.loads(next((tmp_path / 'captures').glob('*/manifest.json')).read_text())
    assert manifest['training_eligible'] is False
