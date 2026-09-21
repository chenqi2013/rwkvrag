import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('state_analysis', Path(__file__).with_name('analyze_state_replay.py'))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_complete_fence_and_plain_parse_without_rewriting():
    for raw in ['>{"conclusion":"未知","evidence_ids":[]}', '>```json\n{"conclusion":"未知","evidence_ids":[]}\n```']:
        parsed, issues = analysis.parse(raw, 'stop')
        assert parsed == {'conclusion': '未知', 'evidence_ids': []} and not issues


@pytest.mark.parametrize('raw', [
    '>{"conclusion":"是","conclusion":"否","evidence_ids":[]}',
    '>```json\n{"conclusion":"未知","evidence_ids":[]}',
    '>{"conclusion":"未知","evidence_ids":[]} additional prose',
    '>{"conclusion":"未知","evidence_ids":[],"extra":1}',
])
def test_invalid_output_not_repaired(raw):
    parsed, issues = analysis.parse(raw, 'stop')
    assert parsed is None and issues == ['invalid_structure']


def test_length_exit_not_semantic_unknown():
    assert analysis.parse('>{"conclusion":"未知","evidence_ids":[]}', 'length') == (None, ['finish_length'])
