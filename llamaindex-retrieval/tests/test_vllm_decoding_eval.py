import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[1] / 'eval/vllm-decoding-20260920'


def load(name):
    spec = importlib.util.spec_from_file_location('decode_eval_' + name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


protocol = load('protocol')
previous = sys.modules.get('protocol')
try:
    sys.modules['protocol'] = protocol
    analysis = load('analysis')
finally:
    if previous is None:
        sys.modules.pop('protocol', None)
    else:
        sys.modules['protocol'] = previous


def test_exact_canonical_boundary_and_labels_no_repair():
    assert protocol.parse('>\n{"answer":"ANSWERABLE"}') is True
    assert protocol.parse('>{"answer":"INSUFFICIENT"}') is False
    for bad in ['{"answer":"ANSWERABLE"}', '>{"answer":"NO"}',
                '>```json\n{"answer":"ANSWERABLE"}\n```',
                '>{"answer":"ANSWERABLE","answer":"INSUFFICIENT"}']:
        with pytest.raises(ValueError):
            protocol.parse(bad)


def test_schedule_and_policy_controls_preserve_inputs():
    inputs = json.loads((ROOT/'inputs.json').read_text())
    schedule = json.loads((ROOT/'schedule.json').read_text())
    assert len(inputs) == 160 and len(schedule) == 1120
    assert len({(r['case_id'], r['arm'], r['seed']) for r in schedule}) == 1120
    for v in inputs:
        assert v['body'] == protocol.prompt(v['case'])
        assert v['prompt'].endswith('<think></think')
    a, b = (protocol.parameters({'arm': arm, 'seed': 11}) for arm in ['top1', 'fake'])
    assert {k for k in a if a[k] != b[k]} == {'top_k', 'top_p'}


def test_seed_repeats_not_counted_as_new_cases_and_invalid_blocks_win():
    inputs = json.loads((ROOT/'inputs.json').read_text())
    cases = {v['case']['id']: v['case'] for v in inputs}
    rows = []
    for r in json.loads((ROOT/'schedule.json').read_text()):
        c = cases[r['case_id']]
        rows.append({**r, 'expected': c['expected'], 'prediction': c['expected'] if r['arm']=='fake' else False,
                     'status': 'valid', 'pair': c['pair'], 'raw_text': 'same', 'token_ids': [1],
                     'elapsed_ms': 1., 'completion_tokens': 1, 'finish_reason': 'stop'})
    good = analysis.summarize(rows, inputs)
    assert good['unique_seen_regression_cases'] == 160 and good['calls'] == 1120
    assert good['mean_accuracy_delta'] == .5 and good['limited_regression_improvement_gate']
    next(r for r in rows if r['arm']=='fake')['status'] = 'invalid'
    bad = analysis.summarize(rows, inputs)
    assert not bad['limited_regression_improvement_gate']
    assert not bad['criteria']['no_seed_increases_invalid']
    with pytest.raises(AssertionError):
        analysis.summarize(rows[:-1], inputs)
