"""Literal identity checks cannot certify facts or silently change output."""
from copy import deepcopy

import pytest

from llamaindex_retrieval.citation_audit import audit_citations


@pytest.mark.parametrize("label", ["[资料 N]", "[资料abc]", "[资料 0]", "[资料 01]",
                                  "[资料 -1]", "[资料 1.5]", "[资料 1"])
def test_invalid_placeholders_numbers_and_incomplete_labels_are_reported(label):
    audit = audit_citations("答复" + label, [{"snippet": "原文"}])
    assert audit["invalid_labels"] == [label]
    assert audit["label_ids"] == []
    assert audit["semantic_support_verified"] is False


def test_valid_labels_and_unknown_sources_are_separate():
    audit = audit_citations("结论[资料 1] [资料2] [资料 1]", [{"snippet": "内容"}])
    assert audit["label_ids"] == [1, 2]
    assert audit["unknown_label_ids"] == [2]
    assert audit["invalid_labels"] == []


def test_quote_must_match_the_cited_source_not_another_source():
    sources = [{"snippet": "甲值为13。"}, {"snippet": "乙值为0。"}]
    original = deepcopy(sources)
    text = "原文：乙值为0。[资料 1]\n结论：乙值为0。[资料 1]"
    audit = audit_citations(text, sources, check_quotes=True)
    assert audit["unknown_label_ids"] == []
    assert audit["quote_audit"]["failed"] == 1
    assert audit["quote_audit"]["quotes"][0]["verbatim_label_ids"] == []
    assert sources == original


def test_supported_quote_keeps_unicode_coordinates_but_does_not_verify_the_conclusion():
    text = "😀\n原文：甲值为13。[资料 1]\n结论：甲值为99。[资料 1]"
    audit = audit_citations(text, [{"snippet": "甲值为13。"}], check_quotes=True)
    quote = audit["quote_audit"]["quotes"][0]
    assert text[slice(*quote["body_span"])] == "原文：甲值为13。[资料 1]"
    assert quote["verbatim"] is True
    assert audit["semantic_support_verified"] is False


def test_quote_can_match_context_that_was_actually_supplied_to_writer():
    sources = [{"snippet": "片段", "metadata": {"context_spans": [{"text": "父级上下文原文"}]}}]
    audit = audit_citations("原文：「父级上下文原文」[资料 1]", sources, check_quotes=True)
    assert audit["quote_audit"]["failed"] == 0


def test_quote_check_is_explicit_and_missing_quotes_are_not_fabricated():
    assert "quote_audit" not in audit_citations("原文：任意[资料 1]", [])
    audit = audit_citations("资料不足，无法确定。", [], check_quotes=True)
    assert audit["quote_audit"]["checked"] == 0
    assert audit["quote_audit"]["semantic_support_verified"] is False


def test_paraphrase_under_an_original_text_label_is_not_treated_as_verbatim():
    audit = audit_citations("原文：电压是12伏。[资料 1]", [{"snippet": "额定电压为12 V。"}],
                           check_quotes=True)
    assert audit["quote_audit"]["failed"] == 1


@pytest.mark.parametrize("spans", [None, "invalid", [None, {"text": 1}]])
def test_non_text_context_metadata_cannot_crash_answer_auditing(spans):
    sources = [{"snippet": "原文", "metadata": {"context_spans": spans}}]
    assert audit_citations("原文：原文[资料 1]", sources, check_quotes=True)["quote_audit"]["failed"] == 0
