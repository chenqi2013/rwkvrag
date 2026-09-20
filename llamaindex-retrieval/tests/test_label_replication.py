import importlib.util
import json
from pathlib import Path
import sys

import pytest

FOLDER = Path(__file__).parents[1] / "eval/reader-label-replication-20260920"
spec = importlib.util.spec_from_file_location("replication_protocol", FOLDER / "protocol.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)
spec = importlib.util.spec_from_file_location("replication_analysis", FOLDER / "analysis.py")
analysis = importlib.util.module_from_spec(spec)
previous = sys.modules.get("protocol")
sys.modules["protocol"] = protocol
try:
    spec.loader.exec_module(analysis)
finally:
    if previous is None:
        del sys.modules["protocol"]
    else:
        sys.modules["protocol"] = previous


def test_new_cases_balanced_schedule_and_unchanged_inputs():
    cases = [json.loads(line) for line in (FOLDER / "cases.jsonl").read_text().splitlines()]
    protocol.validate_cases(cases)
    assert len(cases) == 160 and sum(c["expected"] for c in cases) == 80
    assert len({c["family"] for c in cases}) == 20
    for family in {c["family"] for c in cases}:
        assert len([c for c in cases if c["family"] == family]) == 8
    seen = []
    for name in ["evidence-support-20260920", "reader-label-20260920"]:
        seen += [json.loads(s) for s in (FOLDER.parent / name / "cases.jsonl").read_text().splitlines()]
    prior = {(c["question"], c["text"]) for c in seen}
    assert not any((c["question"], c["text"]) in prior for c in cases)
    for c in cases:
        a = protocol.prompt(c, "yes-no")
        b = protocol.prompt(c, "neutral-labels")
        ai, _, ad = a.partition("\n")
        bi, _, bd = b.partition("\n")
        assert ad == bd and bi.replace("ANSWERABLE", "YES").replace("INSUFFICIENT", "NO") == ai
    schedule = json.loads((FOLDER / "schedule.json").read_text())
    assert len(schedule) == 640
    for c in cases:
        rounds = [[s["arm"] for s in schedule if s["case_id"] == c["id"] and s["round"] == r] for r in (1, 2)]
        assert len(rounds[0]) == 2 and set(rounds[0]) == {"yes-no", "neutral-labels"}
        assert rounds[0] == list(reversed(rounds[1]))
    for r in (1, 2):
        first = [s for i, s in enumerate(schedule) if s["round"] == r and i % 2 == 0]
        assert sum(s["arm"] == "yes-no" for s in first) == 80


def fixture():
    cases = [dict(id=f"{f}-{label}", family=f, pair=f, category="demo", expected=label)
             for f in ["a", "b", "c", "d"] for label in [True, False]]
    rows = [dict(round=r, case_id=c["id"], arm=arm, pair=c["pair"], expected=c["expected"],
                 prediction=c["expected"] if arm == "neutral-labels" else False, status="valid")
            for r in (1, 2) for c in cases for arm in ["yes-no", "neutral-labels"]]
    return cases, rows


def test_repeat_is_not_counted_as_independent_sample_and_incomplete_rejected():
    cases, rows = fixture()
    result = analysis.summarize(rows, cases, bootstrap_draws=200)
    assert result["unique_cases"] == 8 and result["calls"] == 32
    assert result["rounds"]["1"]["accuracy_delta"] == .5
    assert result["rounds"]["1"]["cluster_bootstrap_95_percentile_interval"] == [.5, .5]
    assert result["limited_experimental_improvement_gate"]
    with pytest.raises(ValueError, match="incomplete_or_duplicate"):
        analysis.summarize(rows[:-1], cases, bootstrap_draws=200)


def test_invalid_and_repeat_instability_cannot_be_hidden_by_accuracy_gain():
    cases, rows = fixture()
    rows[-1].update(status="invalid", prediction=None)
    result = analysis.summarize(rows, cases, bootstrap_draws=200)
    assert not result["limited_experimental_improvement_gate"]
    assert result["rounds"]["2"]["candidate"]["invalid"] == 1
    assert result["stability"]["neutral-labels"]["agreement"] < .99
