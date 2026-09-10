import pytest

from llamaindex_retrieval.qa_analysis import (
    QuestionAnalysis,
    ambiguity_candidates,
    analyze_question,
    clean_question_shell,
    comparison_subjects,
    counted_list_size,
    remove_unsupported_number_sentences,
    validate_grounding,
    validate_list_answer,
)
from llamaindex_retrieval.schemas import SourceItem


def source() -> SourceItem:
    return SourceItem(
        id="record",
        document_id="document",
        source="test",
        title="统计",
        score=1.0,
        snippet="2024 年登记了 12 项。",
    )


@pytest.mark.parametrize(
    "question",
    [
        "甲和乙有什么区别？",
        "中国四大名著是哪几个？",
        "最早的创始人是谁？",
        "请介绍这个项目",
        "现在有哪些成员？",
    ],
)
def test_semantic_analysis_is_deferred_to_model(question: str) -> None:
    assert analyze_question(question) == QuestionAnalysis()
    assert comparison_subjects(question) is None
    assert counted_list_size(question) is None
    assert ambiguity_candidates(question, [source()]) == []
    result = validate_list_answer(question, "12 项。[资料 1]", [source()])
    assert result.complete is None
    assert result.expected_count is None
    assert result.answer_count is None


def test_question_cleaning_preserves_conversational_meaning() -> None:
    assert clean_question_shell("  請問你知道 RWKV 是什麼嗎？  ") == "请问你知道 rwkv 是什么吗？"


@pytest.mark.parametrize(
    ("answer", "issues", "numbers"),
    [
        ("2024 年登记 12 项。[资料 1]", (), ()),
        ("2024 年登记 12 项。", ("missing_citation",), ()),
        ("登记 12 项。[资料 2]", ("invalid_citation",), ()),
        ("登记 99 项。[资料 1]", ("unsupported_number",), ("99",)),
    ],
)
def test_grounding_diagnostics_do_not_modify_answer(answer, issues, numbers) -> None:
    result = validate_grounding(answer, [source()])
    assert result.answer == answer
    assert result.issues == issues
    assert result.unsupported_numbers == numbers
    assert result.valid == (not issues)
    assert remove_unsupported_number_sentences(answer, result.unsupported_numbers) == answer
