"""Ensure offline scoring cannot hide invalid output or leak labels into prompts."""
import importlib.util
import json
from pathlib import Path

import pytest

FOLDER = Path(__file__).parents[1] / "eval/evidence-support-20260920"
spec = importlib.util.spec_from_file_location("support_eval_protocol", FOLDER / "protocol.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def row(expected, prediction, pair="one", status="valid"):
    return dict(expected=expected, prediction=prediction, pair=pair, status=status)


def test_confusion_matrix_and_invalid_denominators():
    rows = [row(True, True, "a"), row(False, True, "a"),
            row(True, False, "b"), row(False, False, "b"),
            row(True, None, "c", "invalid"), row(False, None, "c", "invalid")]
    actual = protocol.metrics(rows)
    assert [actual[k] for k in ["tp", "fp", "tn", "fn", "invalid"]] == [1, 1, 1, 1, 2]
    assert actual["recall"] == 1 / 3
    assert actual["precision"] == .5
    assert actual["false_positive_rate"] == 1 / 3
    assert actual["accuracy_including_invalid"] == 1 / 3
    assert actual["pair_accuracy"] == 0 and not actual["pilot_gate_passed"]


def test_always_yes_always_no_and_invalid_do_not_pass():
    for prediction in [True, False, None]:
        actual = protocol.metrics([row(True, prediction), row(False, prediction)])
        assert not actual["pilot_gate_passed"]
    assert not protocol.metrics([])["pilot_gate_passed"]
    assert not protocol.metrics([row(True, True)])["pilot_gate_passed"]
    assert protocol.metrics([row(True, True), row(False, False)])["pilot_gate_passed"]


@pytest.mark.parametrize("name", ["canonical", "strict-v1"])
def test_only_input_fields_enter_prompt(name):
    case = dict(id="opaque-id", question="问题", contexts=[{"text": "上下文"}], text="原文",
                expected="LABEL_CANARY", rationale="RATIONALE_CANARY", split="SPLIT_CANARY",
                category="CATEGORY_CANARY", pair="PAIR_CANARY")
    prompt = protocol.prompt(case, name)
    assert "CANARY" not in prompt
    assert all(x in prompt for x in ["问题", "上下文", "原文"])


def test_frozen_dataset_pairs_and_splits():
    cases = [json.loads(s) for s in (FOLDER / "cases.jsonl").read_text().splitlines()]
    protocol.validate_cases(cases)
    assert len(cases) == 40
    assert len({c["category"] for c in cases}) == 10
    for split in ["development", "validation"]:
        rows = [c for c in cases if c["split"] == split]
        assert len(rows) == 20 and sum(c["expected"] for c in rows) == 10
    for prompt_name in ["canonical", "strict-v1"]:
        for c in cases:
            changed_labels = dict(c, expected=not c["expected"], rationale="LABEL_RATIONALE_CANARY",
                                  category="CATEGORY_CANARY", split="SPLIT_CANARY", pair="PAIR_CANARY")
            assert protocol.prompt(c, prompt_name) == protocol.prompt(changed_labels, prompt_name)
    changed = [dict(c) for c in cases]
    changed[0]["split"] = "development" if changed[0]["split"] == "validation" else "validation"
    with pytest.raises(AssertionError, match="must not cross splits"):
        protocol.validate_cases(changed)
