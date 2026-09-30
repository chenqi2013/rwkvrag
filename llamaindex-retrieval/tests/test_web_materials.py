import hashlib

from llamaindex_retrieval.web_materials import ranked_windows


def test_answer_after_old_prefix_is_ranked_without_rewriting_source_or_code():
    code = "```python\napplication = Framework(docs_url='/internal/docs')\n```\n"
    text = "# Overview\n" + ("Unrelated background. " * 300) + "\n# docs_url\n" + code
    windows, trace = ranked_windows(text, "Framework docs_url", "doc", window=600, limit=3)
    assert any(code in node.text for node, _ in windows)
    assert trace["full_snapshot_characters"] == len(text)
    assert trace["unselected_windows"] > 0
    for node, _ in windows:
        span = node.metadata["source_span"]
        assert node.text == text[span["start"]:span["end"]]
        assert span["sha256"] == hashlib.sha256(node.text.encode()).hexdigest()


def test_structured_table_is_not_truncated_and_zero_negative_values_survive():
    table = "| model | count | offline |\n| --- | --- | --- |\n" + "| item | 0 | false |\n" * 30
    windows, _ = ranked_windows("# Capacity\n" + table, "offline count", "table", window=256, limit=2)
    assert any(node.text == table for node, _ in windows)


def test_whitespace_snapshot_has_no_windows():
    windows, trace = ranked_windows(" \t\n", "query", "empty")
    assert windows == [] and trace["selected_windows"] == 0
