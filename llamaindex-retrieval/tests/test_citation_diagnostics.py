from llamaindex_retrieval.citation_diagnostics import diagnose_citations


def test_citation_diagnostics_ignores_code_examples_without_editing_text() -> None:
    text = "答案 [资料 1]。`[资料 2]`\n\n```text\n[资料 3]\n```"

    result = diagnose_citations(text, source_count=1)

    assert result.references == (1,)
    assert result.unknown == ()
    assert result.in_code == (2, 3)
    assert result.as_dict()["output_modified"] is False


def test_citation_diagnostics_reports_unknown_and_malformed_labels() -> None:
    result = diagnose_citations("结论[资料 4]，另有[资料 xx]和[资料 5", source_count=2)

    assert result.references == (4,)
    assert result.unknown == (4,)
    assert result.malformed == ("[资料 xx]",)
    assert result.unclosed is True


def test_citation_diagnostics_keeps_later_labels_after_unclosed_label() -> None:
    result = diagnose_citations("结论[资料 5\n补充[资料 1]", source_count=1)

    assert result.references == (1,)
    assert result.unknown == ()
    assert result.unclosed is True


def test_citation_diagnostics_ignores_escaped_and_unclosed_code_labels() -> None:
    result = diagnose_citations("正文\\[资料 2]。\n```text\n[资料 3\n", source_count=1)

    assert result.references == ()
    assert result.unknown == ()
    assert result.in_code == ()
    assert result.malformed == ()
    assert result.unclosed is False
