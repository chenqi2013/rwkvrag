import pytest

from llamaindex_retrieval.evaluate import score_evidence, score_titles


def test_score_titles_requires_every_expected_group() -> None:
    complete, groups = score_titles(
        {"expected_title_groups": [["甲", "甲别名"], ["乙"]]},
        ["甲别名", "其他"],
    )
    assert complete is False
    assert groups == [True, False]


def test_score_titles_keeps_expected_titles_compatibility() -> None:
    complete, groups = score_titles({"expected_titles": ["甲", "乙"]}, ["乙"])
    assert complete is True
    assert groups == [True]


def test_score_evidence_is_bound_to_declared_source() -> None:
    case = {
        "evidence_groups": [{
            "document_ids": ["wanted"],
            "must_include_all": ["北京"],
        }]
    }
    results = [
        {"document_id": "other", "title": "其他", "snippet": "首都是北京。"},
        {"document_id": "wanted", "title": "目标", "snippet": "没有所需事实。"},
    ]
    assert score_evidence(case, results) == (False, [False])


def test_score_evidence_supports_all_and_any_terms() -> None:
    case = {
        "evidence_groups": [{
            "document_ids": ["capital"],
            "must_include_all": ["中华人民共和国", "首都"],
            "must_include_any": ["北京", "北京市"],
        }]
    }
    results = [{
        "document_id": "capital",
        "title": "首都",
        "snippet": "中华人民共和国的首都是北 京。",
    }]
    assert score_evidence(case, results) == (True, [True])


def test_score_evidence_returns_none_without_gold() -> None:
    assert score_evidence({}, []) == (None, [])


def test_score_evidence_rejects_unbound_gold() -> None:
    with pytest.raises(ValueError):
        score_evidence(
            {"evidence_groups": [{"must_include_all": ["答案"]}]},
            [],
        )
