import copy
import hashlib

import pytest
from llama_index.core import Document
from llama_index.core.schema import MetadataMode, NodeRelationship

from llamaindex_retrieval.verbatim_chunking import verbatim_nodes


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_verbatim_coverage(document, nodes):
    covered = [False] * len(document.text)
    for node in nodes:
        span = node.metadata["source_span"]
        start, end = span["start"], span["end"]
        assert node.text == document.text[start:end]
        assert node.start_char_idx == start
        assert node.end_char_idx == end
        assert span["unit"] == "unicode_code_points"
        assert span["sha256"] == node.metadata["chunk_text_sha256"] == sha(node.text)
        assert node.metadata["source_text_sha256"] == sha(document.text)
        assert node.metadata["document_id"] == document.id_
        assert node.relationships[NodeRelationship.SOURCE].node_id == document.id_
        covered[start:end] = [True] * (end - start)
        for context in node.metadata["context_spans"]:
            assert context["text"] == document.text[context["start"]:context["end"]]
            assert context["sha256"] == sha(context["text"])
    assert all(covered)
    assert len({node.id_ for node in nodes}) == len(nodes)


def test_prose_windows_cover_raw_unicode_and_whitespace_with_overlap():
    raw = " \r\n" + "甲😀e\u0301é\tAB\r\n" * 17 + "  \n"
    document = Document(id_="unicode", text=raw)
    nodes = verbatim_nodes(document, chunk_characters=31, overlap_characters=7)
    assert_verbatim_coverage(document, nodes)
    assert len(nodes) > 4
    assert nodes[0].text.startswith(" \r\n")
    assert nodes[-1].text.endswith("  \n")
    for previous, current in zip(nodes, nodes[1:]):
        assert current.start_char_idx == previous.end_char_idx - 7
        assert previous.text[-7:] == current.text[:7]
        assert len(previous.text) == 31


@pytest.mark.parametrize("raw", ["", " ", " \t\r\n\n ", "\u2003\u00a0"])
def test_empty_and_whitespace_are_not_stripped(raw):
    document = Document(id_="spaces", text=raw)
    nodes = verbatim_nodes(document)
    assert_verbatim_coverage(document, nodes)
    assert [node.text for node in nodes] == ([raw] if raw else [])


def test_long_pipe_table_atomic_with_verbatim_parent_headings_and_no_prefix():
    table = "| name | value |\r\n| :--- | ---: |\r\n" + "| item | 12 😀 |\r\n" * 30
    raw = "#  Parent  \r\nintro\r\n## Section ###\r\n" + table + "\r\nafter\r\n"
    document = Document(id_="table", text=raw)
    nodes = verbatim_nodes(document, chunk_characters=30, overlap_characters=5)
    assert_verbatim_coverage(document, nodes)
    tables = [node for node in nodes if node.metadata["content_type"] == "table"]
    assert len(tables) == 1
    node = tables[0]
    assert node.text == table
    assert len(node.text) > 30
    assert node.metadata["atomic"] is True
    assert node.metadata["heading_path"] == ["#  Parent  \r\n", "## Section ###\r\n"]
    headings = [item for item in node.metadata["context_spans"] if item["kind"] == "heading"]
    assert [item["text"] for item in headings] == node.metadata["heading_path"]
    assert [item["level"] for item in headings] == [1, 2]
    header = node.metadata["context_spans"][-1]
    assert header == {
        "text": "| name | value |\r\n| :--- | ---: |\r\n",
        "start": raw.index("| name"),
        "end": raw.index("| item"),
        "sha256": sha("| name | value |\r\n| :--- | ---: |\r\n"),
        "kind": "table_header",
        "level": None,
    }
    assert node.get_content(metadata_mode=MetadataMode.NONE) == table
    assert "context_spans" in node.excluded_embed_metadata_keys


def test_table_without_leading_pipes_is_atomic():
    table = "name | value\n:--- | ---:\nalpha | 10\nbeta | 20\n"
    document = Document(id_="table2", text="before\n\n" + table + "\nafter")
    nodes = verbatim_nodes(document, chunk_characters=12, overlap_characters=3)
    assert_verbatim_coverage(document, nodes)
    assert [node.text for node in nodes if node.metadata["content_type"] == "table"] == [
        table,
    ]
    table_node = next(node for node in nodes if node.metadata["content_type"] == "table")
    assert table_node.metadata["context_spans"][0]["text"] == "name | value\n:--- | ---:\n"


def test_header_context_is_available_for_a_selected_tail_row_without_inventing_text():
    header = "| Name | Count |\n| --- | --- |\n"
    text = "# Metrics\n" + header + "| prior | 1 |\n" * 20 + "| tail | 2 |\n"
    document = Document(id_="header-for-tail", text=text)
    nodes = verbatim_nodes(document, chunk_characters=30, overlap_characters=5)
    table = next(node for node in nodes if node.metadata["content_type"] == "table")
    tail = table.text[table.text.index("| tail"):]
    assert "Count" not in tail
    context = next(item for item in table.metadata["context_spans"]
                   if item["kind"] == "table_header")
    assert context["text"] == header
    assert text[context["start"]:context["end"]] == header
    assert_verbatim_coverage(document, nodes)


def test_malformed_pipe_run_is_atomic_without_guessing_a_header():
    text = "# Fragment\n| a | b |\n| c | d |\n"
    document = Document(id_="fragment", text=text)
    nodes = verbatim_nodes(document, chunk_characters=10, overlap_characters=2)
    table = next(node for node in nodes if node.metadata["content_type"] == "table")
    assert table.text == "| a | b |\n| c | d |\n"
    assert all(item["kind"] == "heading" for item in table.metadata["context_spans"])
    assert_verbatim_coverage(document, nodes)


def test_nested_list_and_lazy_continuations_stay_in_one_oversized_atom():
    listing = (
        "- first long item " + "words " * 20 + "\n"
        "  continued\n"
        "  - nested entry\n"
        "    ### nested heading text\n"
        "\n"
        "- second item\n"
        "lazy continuation belongs to this item\n"
        "\n"
        "3. last item\n"
        "\n"
    )
    document = Document(id_="list", text="# List\n" + listing + "Following paragraph.")
    nodes = verbatim_nodes(document, chunk_characters=25, overlap_characters=4)
    assert_verbatim_coverage(document, nodes)
    lists = [node for node in nodes if node.metadata["content_type"] == "list"]
    assert len(lists) == 1
    assert lists[0].text == listing
    assert lists[0].metadata["heading_path"] == ["# List\n"]
    assert all("Following paragraph" not in node.text for node in lists)


@pytest.mark.parametrize("marker", ["```", "~~~~"])
def test_fenced_code_is_atomic_and_internal_markdown_cannot_change_heading_path(marker):
    code = marker + "text\n# not a heading\n| fake | table |\n- not a list\n" + marker + "\n"
    document = Document(id_="code", text="# Real\n" + code + "tail")
    nodes = verbatim_nodes(document, chunk_characters=15, overlap_characters=3)
    assert_verbatim_coverage(document, nodes)
    assert [node.text for node in nodes if node.metadata["content_type"] == "code"] == [code]
    assert nodes[-1].metadata["heading_path"] == ["# Real\n"]
    assert len([node for node in nodes if node.metadata["content_type"] == "heading"]) == 1


def test_unclosed_fence_stays_whole_to_end_of_document():
    text = "```\n" + "very long code\n" * 40 + "# still code"
    document = Document(id_="unclosed", text=text)
    nodes = verbatim_nodes(document, chunk_characters=20, overlap_characters=3)
    assert_verbatim_coverage(document, nodes)
    assert len(nodes) == 1
    assert nodes[0].text == text
    assert nodes[0].metadata["content_type"] == "code"


def test_heading_scope_resets_for_same_or_higher_level_without_rewriting_labels():
    text = "# A\n## Same\nfirst\n### Child\ndeep\n## Same\nsecond\n# B\nlast"
    document = Document(id_="headings", text=text)
    nodes = verbatim_nodes(document)
    assert_verbatim_coverage(document, nodes)
    by_text = {node.text: node for node in nodes}
    assert by_text["deep\n"].metadata["heading_path"] == ["# A\n", "## Same\n", "### Child\n"]
    assert by_text["second\n"].metadata["heading_path"] == ["# A\n", "## Same\n"]
    assert by_text["last"].metadata["heading_path"] == ["# B\n"]
    first_section = by_text["first\n"].metadata["context_spans"][-1]
    second_section = by_text["second\n"].metadata["context_spans"][-1]
    assert first_section["text"] == second_section["text"]
    assert first_section["start"] != second_section["start"]


def test_qa_markdown_is_one_exact_node_with_full_answer_beyond_900_characters():
    raw = " \r\n# Question\r\nWhat is recorded?\r\n## Answer\r\n" + "detail\n" * 300 + "TAIL \r\n"
    document = Document(id_="qa", text=raw, metadata={"kind": "qa-markdown"})
    nodes = verbatim_nodes(document, chunk_characters=100, overlap_characters=10)
    assert_verbatim_coverage(document, nodes)
    assert len(nodes) == 1
    assert nodes[0].text == raw
    assert nodes[0].metadata["content_type"] == "qa"
    assert nodes[0].metadata["context_spans"] == []


def test_stable_ids_bind_document_identity_full_body_and_span_not_title():
    first = Document(id_="site:17", text="repeated " * 100, metadata={"title": "Same"})
    other_page = Document(id_="site:18", text=first.text, metadata={"title": "Same"})
    edited = Document(id_="site:17", text=first.text + "!", metadata={"title": "Same"})
    renamed = Document(id_="site:17", text=first.text, metadata={"title": "Renamed"})

    def nodes(document):
        return verbatim_nodes(document, chunk_characters=90, overlap_characters=9)

    original = nodes(first)
    ids = {node.id_ for node in original}
    assert [node.id_ for node in original] == [node.id_ for node in nodes(first)]
    assert [node.id_ for node in original] == [node.id_ for node in nodes(renamed)]
    assert ids.isdisjoint(node.id_ for node in nodes(other_page))
    assert ids.isdisjoint(node.id_ for node in nodes(edited))
    assert len({node.text for node in original}) < len(original)
    assert len(ids) == len(original)
    for document in (first, other_page, edited, renamed):
        assert_verbatim_coverage(document, nodes(document))


def test_document_metadata_and_sibling_nodes_remain_independent():
    document = Document(
        id_="metadata",
        text="# Title\n" + "body " * 60,
        metadata={"title": "Name", "extra": {"values": [1]}, "source_text_sha256": sha(
            "# Title\n" + "body " * 60
        )},
        excluded_embed_metadata_keys=["extra"],
        excluded_llm_metadata_keys=["title"],
    )
    original = copy.deepcopy(document.model_dump())
    nodes = verbatim_nodes(document, chunk_characters=35, overlap_characters=5)
    assert_verbatim_coverage(document, nodes)
    assert document.model_dump() == original
    nodes[1].metadata["extra"]["values"].append(2)
    nodes[1].metadata["context_spans"][0]["text"] = "changed"
    assert document.model_dump() == original
    assert nodes[2].metadata["extra"] == {"values": [1]}
    assert nodes[2].metadata["context_spans"][0]["text"] == "# Title\n"
    assert "extra" in nodes[2].excluded_embed_metadata_keys
    assert "title" in nodes[2].excluded_llm_metadata_keys


def test_declared_whole_source_hash_mismatch_is_rejected():
    document = Document(id_="hash", text=" raw ", metadata={"source_text_sha256": sha("raw")})
    with pytest.raises(ValueError, match="source_text_sha256"):
        verbatim_nodes(document)


@pytest.mark.parametrize("chunk,overlap", [
    (0, 1), (1, 1), (-1, 1), (True, 1), (3.5, 1),
    (20, 0), (20, -1), (20, 20), (20, 21), (20, True), (20, 1.5),
])
def test_invalid_or_nonoverlapping_windows_are_rejected(chunk, overlap):
    with pytest.raises(ValueError):
        verbatim_nodes(Document(id_="invalid", text="hello"), chunk, overlap)


def test_mixed_structure_all_original_characters_are_covered():
    text = (
        "\n  leading\r\n# Header\r\n" + "plain prose " * 30 + "\r\n\r\n"
        "| A | B |\r\n|---|---|\r\n|1|2|\r\n\r\n"
        "- one\r\n- two\r\n\r\n"
        "ordinary text\r\n## Next\r\n~~~\r\ncode\r\n~~~\r\n\r\n trailing \t"
    )
    document = Document(id_="mixed", text=text)
    nodes = verbatim_nodes(document, chunk_characters=30, overlap_characters=8)
    assert_verbatim_coverage(document, nodes)
    assert {node.metadata["content_type"] for node in nodes} == {
        "prose", "heading", "table", "list", "code",
    }
    assert all(len(node.text) <= 30 or node.metadata["atomic"] for node in nodes)
