import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('state_replay', HERE/'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_full_membership_and_crossed_order():
    rows = json.loads((HERE/'INPUTS.json').read_text())
    calls = list(runner.schedule(rows))
    assert len(rows) == 141 and len(calls) == 564
    for index, row in enumerate(rows):
        matching = [(r, n, a) for r, n, a, item in calls if item['call_id'] == row['call_id']]
        assert matching == [(1, index, 'zero'), (1, index, '2000'), (2, index, '2000'), (2, index, 'zero')]


def test_gold_binds_exact_stage_inputs():
    rows = json.loads((HERE/'INPUTS.json').read_text())
    gold = json.loads((HERE/'ASSESS-EXPECTATIONS.json').read_text())['rows']
    assert len(rows) == len(gold) == 141
    indexed = {r['call_id']: r for r in gold}
    assert len(indexed) == 141
    for row in rows:
        expected = indexed[row['call_id']]
        assert expected['prompt_sha256'] == row['prompt_sha256']
        assert expected['question_and_evidence'] == row['payload']['messages'][0]['content']
        assert expected['allowed_evidence_ids'] == row['evidence_ids']
