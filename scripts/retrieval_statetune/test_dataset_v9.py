"""Guard against turning two real quotes into a false supported comparison."""

from build_balanced_evidence_v8 import validate
from build_decision_candidates_v7 import evidence_rows


def pair_job():
    return {"id": "pair-test", "split": "train",
            "a": {"repo": "team/alpha", "family": "repo:team/alpha", "source_hash": "a" * 64,
                  "blocks": [{"id": "a1", "text": "Alpha uses PostgreSQL for task storage.",
                              "sha256": "b" * 64, "start_byte": 0}]},
            "b": {"repo": "team/beta", "family": "repo:team/beta", "source_hash": "c" * 64,
                  "blocks": [{"id": "b1", "text": "Beta uses SQLite for task storage.",
                              "sha256": "d" * 64, "start_byte": 0}]}}


def pair_item():
    return {"question": "team/alpha 与 team/beta 分别使用哪种任务存储？",
            "kind": "comparison", "status": "supported",
            "block_a": "a1", "quote_a": "Alpha uses PostgreSQL for task storage.",
            "block_b": "b1", "quote_b": "Beta uses SQLite for task storage."}


def test_pair_accepts_two_exact_source_spans():
    rows, failures = validate(pair_job(), {"items": [pair_item()]})
    assert not failures
    assert rows[0]["source_families"] == ["repo:team/alpha", "repo:team/beta"]
    assert [source["quote"] for source in rows[0]["sources"]] == [
        "Alpha uses PostgreSQL for task storage.",
        "Beta uses SQLite for task storage."]


def test_pair_rejects_a_quote_from_the_wrong_project():
    wrong = pair_item()
    wrong["quote_b"] = wrong["quote_a"]
    rows, failures = validate(pair_job(), {"items": [wrong]})
    assert not rows
    assert failures[0]["error"] == "second quote is not exact"


def test_pair_rejects_a_teacher_qualification_outside_the_label():
    questionable = pair_item()
    questionable["note"] = "Beta excerpt does not answer the same dimension"
    rows, failures = validate(pair_job(), {"items": [questionable]})
    assert not rows
    assert "nonempty qualification" in failures[0]["error"]


def test_single_source_cannot_mark_two_project_question_supported():
    job = {"id": "e-test", "split": "train", "repo": "team/alpha",
           "family": "repo:team/alpha", "source_hash": "a" * 64,
           "peers": ["team/beta"],
           "blocks": [{"id": "a1", "text": "Alpha uses PostgreSQL for task storage.",
                       "sha256": "b" * 64, "start_byte": 0, "end_byte": 40}]}
    item = {"question": "team/alpha 与 team/beta 分别使用哪种任务存储？",
            "kind": "comparison", "status": "supported", "block_id": "a1",
            "quote": "Alpha uses PostgreSQL for task storage.", "peer": "team/beta"}
    rows, failures = evidence_rows(job, {"items": [item]})
    assert not rows
    assert failures[0]["error"] == "supported case must be single-object"
