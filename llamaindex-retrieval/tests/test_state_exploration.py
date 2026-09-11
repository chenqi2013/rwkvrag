"""Quality gate and pre-gold integrity checks for the bounded state search."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'statetune'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('score_exploration', SCRIPTS/'score_exploration.py')
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def write(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle(tmp_path, *, capped=False, negative_false_positive=False):
    run = tmp_path/'run'
    run.mkdir()
    inputs = tmp_path/'inputs.jsonl'
    inputs.write_text('\n'.join(json.dumps({'id': name, 'units': ['a', 'b'],
                                          'prompt_sha256': name}) for name in ['positive', 'negative']))
    gold = tmp_path/'gold.jsonl'
    gold.write_text('\n'.join(json.dumps({'id': name, 'expected_unit_ids': expected})
                              for name, expected in [('positive', ['E2']), ('negative', [])]))
    states = [{'name': name, 'sha256': name} for name in ['zero', 'epoch-1']]
    config = tmp_path/'config.json'
    config_sha = write(config, {'bindings': {}, 'evaluation': {'split': 'dev', 'states': states,
        'inputs': str(inputs), 'inputs_sha256': scorer.digest(inputs),
        'gold_sha256': scorer.digest(gold)}})
    pins = []
    for state in states:
        for name in ['positive', 'negative']:
            text = 'NONE'
            if state['name'] == 'epoch-1' and (name == 'positive' or negative_false_positive):
                text = 'E2'
            hit_cap = capped and state['name'] == 'epoch-1' and name == 'positive'
            record = {'id': name, 'state': state['name'], 'state_sha256': state['sha256'],
                'prompt_sha256': name, 'output': {'raw_text': text, 'eos_observed': not hit_cap,
                                                 'cap_reached': hit_cap, 'utf8_valid': True}}
            path = run/f'{len(pins)}.json'
            pins.append({'path': path.name, 'sha256': write(path, record)})
    write(run/'COMPLETED.json', {'status': 'GENERATION_COMPLETE', 'gold_opened': False,
        'optimizer_updates': 0, 'config_sha256': config_sha, 'split': 'dev', 'records': pins})
    return config, run, gold


def test_source_correct_candidate_improves(tmp_path):
    result = scorer.score(*bundle(tmp_path))
    assert result['selected_state'] == 'epoch-1'
    assert result['improves_over_zero']
    assert result['scores'][1]['correct'] == 2


@pytest.mark.parametrize('kwargs', [{'capped': True}, {'negative_false_positive': True}])
def test_invalid_or_false_positive_candidate_does_not_advance(tmp_path, kwargs):
    result = scorer.score(*bundle(tmp_path, **kwargs))
    assert not result['improves_over_zero']
    assert result['selected_state'] == 'zero'


def test_missing_generation_rejected_before_gold_access(tmp_path):
    config, run, gold = bundle(tmp_path)
    receipt = json.loads((run/'COMPLETED.json').read_text())
    receipt['records'].pop()
    write(run/'COMPLETED.json', receipt)
    gold.unlink()
    with pytest.raises(ValueError, match='coverage'):
        scorer.score(config, run, gold)


def test_changed_raw_output_rejected_before_gold_access(tmp_path):
    config, run, gold = bundle(tmp_path)
    (run/'0.json').write_text('{}')
    gold.unlink()
    with pytest.raises(ValueError, match='raw output changed'):
        scorer.score(config, run, gold)


def test_partial_correct_gain_with_unterminated_case_cannot_advance():
    zero = {'state': 'zero', 'correct': 1, 'positive_correct': 0, 'negative_correct': 1,
            'negative_false_positives': 0, 'invalid': 0}
    candidate = {**zero, 'state': 'candidate', 'correct': 3, 'positive_correct': 2, 'invalid': 1}
    selected, overall, improves = scorer.choose([zero, candidate])
    assert selected['state'] == 'zero'
    assert overall['state'] == 'candidate'
    assert not improves


def test_unqualified_higher_score_cannot_hide_qualified_candidate():
    zero = {'state': 'zero', 'correct': 1, 'positive_correct': 0, 'negative_correct': 1,
            'negative_false_positives': 0, 'invalid': 0}
    eligible = {**zero, 'state': 'eligible', 'correct': 2, 'positive_correct': 1}
    false_positive = {**zero, 'state': 'false-positive', 'correct': 3, 'positive_correct': 3,
                      'negative_correct': 0, 'negative_false_positives': 1}
    selected, overall, improves = scorer.choose([zero, eligible, false_positive])
    assert selected['state'] == 'eligible'
    assert overall['state'] == 'false-positive'
    assert improves


@pytest.mark.parametrize('negative_weight', [1, 2, 4])
def test_negative_weight_is_global_mean_one(negative_weight):
    from train_balanced_state import example_weights
    from types import SimpleNamespace
    vocab = SimpleNamespace(by_id={0: b'', 1: b'NONE', 2: b'E1'})
    rows = [{'id': str(i), 'prompt_tokens': 1, 'input_ids': [2, token, 0]}
            for i, token in enumerate([2, 2, 1])]
    weights = example_weights(rows, vocab, negative_weight)
    assert sum(weights.values()) == pytest.approx(3)
    assert weights['0'] == weights['1']
    assert weights['2']/weights['0'] == pytest.approx(negative_weight)
