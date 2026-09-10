import pytest

from llamaindex_retrieval.evidence_gate import (
    document_aliases,
    evaluate_answer_support,
    evaluate_evidence_gate,
    repair_answer_citations,
    title_matches_subject,
)
from llamaindex_retrieval.qa_analysis import QuestionAnalysis
from llamaindex_retrieval.schemas import SourceItem


def source(title="资料", snippet="对象由某位研究者建立。") -> SourceItem:
    return SourceItem(
        id="record",
        document_id="document",
        source="test",
        title=title,
        score=1.0,
        snippet=snippet,
    )


@pytest.mark.parametrize(
    ("sources", "available", "count", "issues"),
    [
        ([], False, 0, ("no_evidence",)),
        ([source()], True, 0, ("field_evidence_missing",)),
        ([source()], True, 1, ()),
        ([source()], False, 0, ()),
    ],
)
def test_gate_checks_evidence_presence_and_extraction_completion(
    sources,
    available,
    count,
    issues,
) -> None:
    result = evaluate_evidence_gate(
        "谁建立了这个对象？",
        QuestionAnalysis(),
        sources,
        field_evidence_available=available,
        field_candidate_count=count,
    )
    assert result.issues == issues
    assert result.passed == (not issues)


@pytest.mark.parametrize(
    ("question", "title", "snippet"),
    [
        ("宇树科技创始人是谁？", "公司", "王兴兴于2016年创办该公司。"),
        ("西游记是哪个作者写的？", "文学作品", "一般认为作者是吴承恩。"),
        ("中国四大名著是哪几个？", "四大名著", "红楼梦、西游记、水浒传、三国演义。"),
        ("甲公司的创始人是谁？", "乙公司", "乙公司的创始人是某人。"),
    ],
)
def test_gate_defers_title_and_relation_meaning_to_model(question, title, snippet) -> None:
    result = evaluate_evidence_gate(
        question,
        QuestionAnalysis(),
        [source(title, snippet)],
        subject="模型指定的对象",
        relations=("模型指定的关系",),
        field_evidence_available=True,
        field_candidate_count=1,
    )
    assert result.passed
    assert result.assessment.anchors == set()
    assert result.matched_relation_terms == ()


@pytest.mark.parametrize(
    ("answer", "issues"),
    [
        ("某位研究者。[资料 1]", ()),
        ("另一位研究者。[资料 1]", ()),
        ("某位研究者。", ("missing_valid_citation",)),
        ("某位研究者。[资料 2]", ("unknown_citation",)),
    ],
)
def test_answer_support_checks_references_without_semantic_verdict(answer, issues) -> None:
    result = evaluate_answer_support(answer, [source()])
    assert result.issues == issues
    assert result.supported_terms == ()
    assert result.unsupported_terms == ()
    assert repair_answer_citations(answer, [source()]) == answer


def test_no_sources_cannot_pass_answer_support() -> None:
    result = evaluate_answer_support("根据检索到的资料，无法确定。", [])
    assert not result.passed
    assert result.issues == ("no_evidence",)


def test_aliases_only_come_from_explicit_metadata() -> None:
    assert document_aliases("深圳地铁1号线", {}) == set()
    assert document_aliases("项目（消歧义）", {}) == set()
    assert document_aliases("项目", {"aliases": ["別名", " RWKV ", ""]}) == {"别名", "rwkv"}
    assert document_aliases("项目", {"aliases": "不是列表"}) == set()
    assert title_matches_subject("臺灣 RWKV", "台湾rwkv")
    assert not title_matches_subject("甲公司创始人", "甲公司")
