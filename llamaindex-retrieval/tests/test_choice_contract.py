from copy import deepcopy
from hashlib import sha256
import json

import pytest

from llamaindex_retrieval.choice_contract import audit_consistency, parse_record


def fixture():
    user = "重量不超过900克，合格设备中优先续航。"
    source = "甲重800克。乙重950克。"
    def snapshot(identity, kind, text):
        return dict(id=identity, kind=kind, version="v1", text=text,
                    sha256=sha256(text.encode()).hexdigest())
    snapshots = [snapshot("u", "user", user), snapshot("s", "source", source)]
    def span(index, start, end):
        s = snapshots[index]
        return dict(snapshot_id=s["id"], snapshot_sha256=s["sha256"], start=start, end=end)
    return dict(protocol="choice-contract-v1", snapshots=snapshots,
        requirements=[dict(id="h", kind="hard", text="重量不超过900克", active=True,
                           user_spans=[span(0, 0, 10)], supersedes=[]),
                      dict(id="p", kind="preference", text="优先续航", active=True,
                           user_spans=[span(0, 10, len(user))], supersedes=[])],
        candidates=[dict(id=c, name_and_scope=c, origin_spans=[span(1, i, i+1)])
                    for c, i in [("甲", 0), ("乙", 7)]],
        observations=[dict(id="o"+c, candidate_id=c, source_spans=[span(1, a, b)])
                      for c, a, b in [("甲", 0, 7), ("乙", 7, len(source))]],
        judgments=[dict(id="j"+c, candidate_id=c, requirement_id="h", status=status,
                        explanation="模型原始判断", observation_ids=["o"+c])
                   for c, status in [("甲", "satisfied"), ("乙", "not_satisfied")]],
        summaries=[dict(candidate_id=c, status=status, judgment_ids=["j"+c], explanation="模型汇总")
                   for c, status in [("甲", "eligible"), ("乙", "ineligible")]],
        recommended_ids=["甲"], preference_ids=["p"], raw_answer="甲满足限制。[资料 1]")


def parse(data):
    return parse_record(json.dumps(data, ensure_ascii=False))


def test_valid_unicode_offsets_and_immutable_answer():
    data = fixture()
    record = parse(data)
    assert audit_consistency(record) == []
    assert record.raw_answer == data["raw_answer"]
    with pytest.raises(ValueError):
        record.raw_answer = "替换"
    assert isinstance(record.judgments, tuple)


@pytest.mark.parametrize("mutation", [
    lambda d: d["snapshots"][1].update(text="内容被替换"),
    lambda d: d["requirements"][0].update(user_spans=d["observations"][0]["source_spans"]),
    lambda d: d["observations"][0].update(source_spans=d["requirements"][0]["user_spans"]),
    lambda d: d["observations"][0]["source_spans"][0].update(end=999),
    lambda d: d["observations"][0]["source_spans"][0].update(start=True),
    lambda d: d["observations"][0]["source_spans"][0].update(snapshot_sha256="0"*64),
    lambda d: d["judgments"].pop(),
    lambda d: d["judgments"].append(deepcopy(d["judgments"][0])),
    lambda d: d["judgments"][0].update(observation_ids=["o乙"]),
    lambda d: d["judgments"][0].update(observation_ids=[]),
    lambda d: d["judgments"][0].update(status="http_error"),
    lambda d: d["summaries"][0].update(judgment_ids=["j乙"]),
    lambda d: d["summaries"].pop(),
    lambda d: d.update(recommended_ids=["不存在"]),
    lambda d: d.update(preference_ids=["h"]),
    lambda d: d["requirements"][0].update(active="true"),
    lambda d: d["requirements"][0].update(supersedes=["h"]),
    lambda d: d["requirements"][0].update(supersedes=["p"]),
    lambda d: d["candidates"][0].update(id=" "),
])
def test_invalid_dependencies_are_not_coerced_or_repaired(mutation):
    data = fixture()
    mutation(data)
    before = deepcopy(data)
    with pytest.raises(ValueError):
        parse(data)
    assert data == before


def test_duplicate_json_keys_rejected():
    raw = json.dumps(fixture()).replace('"status": "satisfied"',
                                      '"status": "unknown", "status": "satisfied"')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_record(raw)


def test_complete_revised_requirement_history_and_cycle_rejected():
    data = fixture()
    old = deepcopy(data["requirements"][0])
    old.update(id="old", active=False, text="旧条件")
    data["requirements"].append(old)
    data["requirements"][0]["supersedes"] = ["old"]
    assert parse(data).requirements[-1].active is False
    older = deepcopy(old)
    older.update(id="older", supersedes=["old"])
    data["requirements"].append(older)
    old["supersedes"] = ["older"]
    with pytest.raises(ValueError, match="cyclic"):
        parse(data)


def test_unknown_with_or_without_negative_evidence_is_not_false():
    data = fixture()
    data["judgments"][0]["status"] = "unknown"
    data["summaries"][0]["status"] = "unresolved"
    data["recommended_ids"] = []
    for refs in (["o甲"], []):
        data["judgments"][0]["observation_ids"] = refs
        record = parse(data)
        assert audit_consistency(record) == []
        assert record.judgments[0].status == "unknown"


def test_preference_cannot_silently_readmit_failed_candidate():
    data = fixture()
    data["recommended_ids"] = ["乙"]
    data["raw_answer"] = "乙续航更长，所以选乙。[资料 1]"
    record = parse(data)
    assert audit_consistency(record) == [dict(code="recommendation_outside_eligible_candidates", candidate_id="乙")]
    assert record.raw_answer == data["raw_answer"]
    assert record.recommended_ids == ("乙",)


def test_wrong_summary_is_flagged_without_replacing_model_status():
    data = fixture()
    data["summaries"][1]["status"] = "eligible"
    record = parse(data)
    assert audit_consistency(record)[0]["code"] == "eligible_without_all_satisfied"
    assert record.summaries[1].status == "eligible"


def test_structure_does_not_claim_semantic_truth_or_citation_validity():
    data = fixture()
    data["judgments"][1]["status"] = "satisfied"  # Wrong numeric interpretation, model-owned.
    data["summaries"][1]["status"] = "eligible"
    data["recommended_ids"] = ["乙"]
    data["raw_answer"] = "乙低于900克。[资料 99]"
    record = parse(data)
    assert audit_consistency(record) == []
    assert record.raw_answer == data["raw_answer"]


@pytest.mark.parametrize("candidate,status,code", [
    (0, "ineligible", "ineligible_without_failed_condition"),
    (0, "unresolved", "unresolved_despite_all_satisfied"),
    (1, "unresolved", "unresolved_despite_failed_condition"),
])
def test_other_summary_contradictions(candidate, status, code):
    data = fixture()
    data["recommended_ids"] = []
    data["summaries"][candidate]["status"] = status
    assert audit_consistency(parse(data))[0]["code"] == code
