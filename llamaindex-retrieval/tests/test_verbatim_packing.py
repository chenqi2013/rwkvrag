"""Packing is original-byte organization, not semantic summarization."""
from hashlib import sha256

from llama_index.core import Document
import pytest

from llamaindex_retrieval.unified_materials import EvidenceCatalog
from llamaindex_retrieval.verbatim_chunking import packed_verbatim_nodes, verbatim_nodes
from llamaindex_retrieval.web_materials import ranked_windows
from test_unified_retrieval import material


def digest(text):
    return sha256(text.encode()).hexdigest()


def check(document, window, overlap):
    plain = verbatim_nodes(document, window, overlap)
    before = [node.model_dump() for node in plain]
    packed = packed_verbatim_nodes(document, window, overlap)
    covered = set()
    assert len(packed) <= len(plain)
    for i, node in enumerate(packed):
        span = node.metadata["source_span"]
        start, end = span["start"], span["end"]
        assert node.text == document.text[start:end]
        assert node.start_char_idx == start and node.end_char_idx == end
        assert span["sha256"] == node.metadata["chunk_text_sha256"] == digest(node.text)
        assert node.metadata["source_text_sha256"] == digest(document.text)
        assert node.metadata["chunk_order"] == i
        assert len(node.text) <= window or node.metadata["atomic"]
        covered.update(range(start, end))
        for context in node.metadata["context_spans"]:
            assert document.text[context["start"]:context["end"]] == context["text"]
            assert digest(context["text"]) == context["sha256"]
        for component in node.metadata.get("packed_components", []):
            s = component["span"]
            assert start <= s["start"] < s["end"] <= end
            assert digest(document.text[s["start"]:s["end"]]) == s["sha256"]
    assert covered == set(range(len(document.text)))
    for original in plain:
        # No original atom or overlapping prose window is truncated by packing.
        assert any(p.start_char_idx <= original.start_char_idx and p.end_char_idx >= original.end_char_idx
                   for p in packed)
        for context in original.metadata["context_spans"]:
            hosts = [p for p in packed if p.start_char_idx <= original.start_char_idx and p.end_char_idx >= original.end_char_idx]
            assert any(context in p.metadata["context_spans"] or
                       p.start_char_idx <= context["start"] < context["end"] <= p.end_char_idx for p in hosts)
    assert [n.model_dump() for n in plain] == before
    assert [n.id_ for n in packed] == [n.id_ for n in packed_verbatim_nodes(document, window, overlap)]
    return plain, packed


@pytest.mark.parametrize("text", [
    "", " \t\r\n", "# 标题🙂\r\n\r\n说明。\r\n```py\r\na()\r\n```\r\n",
    "# A\nintro\n## B\n| key | value |\n| --- | --- |\n| zero | 0 |\n\n- one\n  - nested\n\n# C\nother\n",
    "# A\n" + "正文🙂 x " * 150 + "\n# B\n```py\ncall()\n```\n",
    "# Table\n| x | y |\n| --- | --- |\n" + "| item | 0 |\n" * 100 + "\n# Tail\nend",
    "# List\n" + "- item\n  continuation\n" * 50 + "\n# Tail\nend",
    "# Code\n```text\n" + "value\n" * 70 + "```\n\n# Next\ntext\n",
])
@pytest.mark.parametrize("window,overlap", [(64, 16), (256, 64), (2400, 180)])
def test_exact_coverage_atomic_boundaries_context_and_overlap(text, window, overlap):
    check(Document(text=text, id_="source", metadata={"owner": "fixture"}), window, overlap)


def test_small_explanation_and_code_are_one_contiguous_unit_not_concatenated_atoms():
    text = "# Example\n\nExplanation\r\n\r\n```text\napi_a()\n```\n\nRetrieval\n```text\napi_b()\n```\n"
    plain, packed = check(Document(text=text, id_="source"), 2400, 180)
    assert len(plain) > 1 and len(packed) == 1
    assert packed[0].text == text
    assert packed[0].metadata["chunker_version"] == "verbatim-packed-v1"
    assert len(packed[0].metadata["packed_components"]) == len(plain)


def test_oversized_qa_is_never_split_or_misrepresented_as_within_hard_budget():
    text = "# 问题\n" + "问题内容。" * 90 + "\n# 答案\n回答。"
    plain, packed = check(Document(text=text, id_="qa", metadata={"kind": "qa-markdown"}), 256, 64)
    assert len(plain) == len(packed) == 1 and packed[0].text == text
    assert packed[0].metadata["atomic"] and len(packed[0].text) > 256


def test_packing_preserves_long_prose_overlap_and_single_node_identity():
    text = "普通连续文本" * 200
    plain, packed = check(Document(text=text, id_="prose"), 256, 64)
    assert [p.id_ for p in plain] == [p.id_ for p in packed]
    for left, right in zip(packed, packed[1:]):
        assert left.end_char_idx - right.start_char_idx == 64
        assert left.text[-64:] == right.text[:64]


def test_ranking_is_after_packing_and_query_does_not_change_unit_ids():
    text = "# Alpha\n\nText\n```py\nalpha()\n```\n\n# Beta\n" + "other " * 100
    a, trace = ranked_windows(text, "Alpha", "doc", window=256, limit=100, pack_adjacent=True)
    b, _ = ranked_windows(text, "Beta", "doc", window=256, limit=100, pack_adjacent=True)
    assert {node.id_ for node, _ in a} == {node.id_ for node, _ in b}
    assert trace["packing_policy"] == "adjacent_verbatim_v1"
    _, old_trace = ranked_windows(text, "Alpha", "doc", window=256, limit=100)
    assert "packing_policy" not in old_trace


def test_catalog_packing_is_explicit_and_keeps_parent_metadata_and_request_dedup():
    source = material(text="# A\n\nExplanation\n```py\ncall()\n```\n")
    parent = {"text": "parent context", "sha256": digest("parent context")}
    source.metadata["context_spans"] = [parent]
    old, packed = EvidenceCatalog(2400), EvidenceCatalog(2400, pack_adjacent=True)
    assert len(old.add(source, "A", "A1")) > 1
    first = packed.add(source, "A", "A1")
    assert first == packed.add(source, "other", "A2") == ["E1"]
    assert packed.units["E1"].snippet == source.snippet
    assert packed.material("E1")["parent_context_spans"] == [parent]
    assert source.metadata["context_spans"] == [parent]
    assert packed.add(material(1, source.snippet), "A", "A3") == ["E2"]
    assert packed.units["E1"].metadata["snapshot_id"] != packed.units["E2"].metadata["snapshot_id"]
