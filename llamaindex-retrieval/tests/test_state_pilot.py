import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "statetune"
sys.path.insert(0, str(SCRIPTS))
from evaluate_state import generate
from score_state import score_case, select_state




def test_full_prefix_eos_and_cap_are_not_repaired():
    prefixes = []
    tokens = iter([2, 0])
    def decide(prefix):
        prefixes.append(list(prefix))
        return next(tokens)
    raw = generate([1], {1:b"prompt", 2:b"E1"}, decide)
    assert prefixes == [[1], [1,2]]
    assert raw["eos_observed"] and raw["raw_text"] == "E1"
    capped = generate([1], {2:b"E1"}, lambda p: 2, cap=1)
    assert capped["cap_reached"] and not score_case(capped,["E1"],3)["correct"]
    with pytest.raises(ValueError, match="exceed context"):
        generate([1]*4096, {2:b"E1"}, lambda p: pytest.fail("model called"))


def test_scoring_uses_production_parser_and_rejects_malformed_bytes():
    output = {"eos_observed":True,"cap_reached":False,"utf8_valid":True,"raw_text":" E3,E2 "}
    assert score_case(output,["E2","E3"],3)["correct"]
    for text in ["E4", "E1 extra", "", "NONE E1"]:
        assert not score_case({**output,"raw_text":text},[],3)["valid"]
    assert not score_case({**output,"utf8_valid":False},["E2","E3"],3)["valid"]
    assert score_case({**output,"raw_text":"NONE"},[],3)["correct"]


def test_selection_tie_keeps_zero_and_syntax_gain_is_not_positive_gain():
    zero = {"step":0,"correct":4,"positive_correct":0,"negative_correct":4,"negative_false_positives":0}
    final = {**zero,"step":11}
    assert select_state([final,zero])["selected_step"] == 0
    assert not select_state([zero,{**final,"correct":5}])["heldout_eligible"]
    assert select_state([zero,{**final,"correct":5,"positive_correct":1}])["heldout_eligible"]
    assert not select_state([zero,{**final,"correct":5,"positive_correct":2,
                                  "negative_false_positives":1}])["heldout_eligible"]
    assert not select_state([zero,{**final,"correct":5,"positive_correct":5,
                                  "negative_correct":0}])["heldout_eligible"]
