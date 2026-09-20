from copy import deepcopy

import pytest
from llama_index.core import Document

from llamaindex_retrieval.atomic_evidence import AtomicRequest, candidates, digest, selection_prompt
from llamaindex_retrieval.reader_prompt import binary_query_prompt
from llamaindex_retrieval.schemas import SourceItem
from llamaindex_retrieval.verbatim_chunking import verbatim_nodes


def table_source(columns=2):
    names = ["续航（小时）"] + [f"辅助字段{i:02d}" for i in range(1, columns - 1)]
    text = ("# 设备😀\n## 模式记录\n| 模式 | " + " | ".join(names) + " |\n| --- | "
            + " | ".join("---" for _ in names) + " |\n| 省电 | "
            + " | ".join(["48"] + ["0"] * (columns - 2)) + " |\n")
    node = next(n for n in verbatim_nodes(Document(text=text, id_="doc"))
                if n.metadata["content_type"] == "table")
    return SourceItem(id=node.id_, document_id="doc", source="synthetic", title="设备😀",
                      score=1, snippet=node.text, metadata=node.metadata)


REQUEST = AtomicRequest(object="设备😀", attribute="续航", conditions="省电模式")


def answer_row(source):
    spans, stats = candidates(source, REQUEST)
    return next(s for s in spans if "| 省电" in s["text"]), stats


def test_header_attached_once_with_exact_offsets_and_saved_audit():
    source = table_source()
    original = deepcopy(source.model_dump())
    row, stats = answer_row(source)
    context = "".join(c["text"] for c in row["context"])
    assert context.count("| 模式 | 续航（小时） |") == 1
    assert "# 设备😀\n" in context and "## 模式记录\n" in context
    assert source.snippet[row["start"]:row["end"]] == row["text"]
    for c in row["context"]:
        parent = source.metadata["context_spans"][c["context_index"]]
        assert c["text"] == parent["text"] and c["sha256"] == digest(parent["text"])
    removed = row["context_deduplication"]["removed_local_ranges"]
    assert len(removed) == 2
    for c in removed:
        assert source.snippet[c["start"]:c["end"]] == c["text"]
    assert stats["oversized_spans"] == 0
    assert original == source.model_dump()


def test_wide_answer_survives_and_both_prompts_use_same_context():
    source = table_source(30)
    row, stats = answer_row(source)
    assert sum(len(c["text"]) for c in row["context"]) <= 600
    assert stats["oversized_spans"] == 0
    assert len(row["text"]) <= 600
    selector = selection_prompt(REQUEST, [row])
    reader = binary_query_prompt(["续航？"], {"id": source.id},
                                 [{"text": c["text"]} for c in row["context"]], row["text"])
    assert selector.count("| 模式 | 续航（小时） |") == 1
    assert reader.count("| 模式 | 续航（小时） |") == 1
    assert "| 省电 | 48" in selector and "| 省电 | 48" in reader


@pytest.mark.parametrize("mutation", ["missing_binding", "wrong_chunk_hash", "wrong_header_hash",
    "different_document", "different_source", "different_revision", "wrong_parent_range",
    "wrong_chunk_unit", "wrong_parent_unit"])
def test_unproven_identity_never_removes_local_context(mutation):
    source = table_source()
    parent = source.metadata["context_spans"][-1]
    if mutation == "missing_binding":
        source.metadata.pop("source_span")
    elif mutation == "wrong_chunk_hash":
        source.metadata["source_span"]["sha256"] = "bad"
    elif mutation == "wrong_header_hash":
        parent["sha256"] = "bad"
    elif mutation == "different_document":
        parent["document_id"] = "another-document"
    elif mutation == "different_source":
        parent["source_id"] = "another-source"
    elif mutation == "different_revision":
        parent["source_text_sha256"] = "another-revision"
    elif mutation == "wrong_parent_range":
        parent["start"] += 1
        parent["end"] += 1
    elif mutation == "wrong_parent_unit":
        parent["unit"] = "utf8_bytes"
    else:
        source.metadata["source_span"]["unit"] = "utf8_bytes"
    row, _ = answer_row(source)
    assert "context_deduplication" not in row
    assert len([c for c in row["context"] if c.get("origin") != "saved_source_metadata"]) == 2


def test_identical_text_at_another_document_range_is_not_deduplicated():
    source = table_source()
    header = source.metadata["context_spans"][-1]
    other = deepcopy(header)
    other["start"] += 1000
    other["end"] += 1000
    source.metadata["context_spans"].append(other)
    row, _ = answer_row(source)
    assert "".join(c["text"] for c in row["context"]).count(header["text"]) == 2
    assert {c["context_index"] for c in row["context"]} == set(range(4))


def test_real_oversized_context_still_excludes_row_without_truncation():
    source = table_source(45)
    original = deepcopy(source.model_dump())
    spans, stats = candidates(source, REQUEST)
    assert sum(len(c["text"]) for c in source.metadata["context_spans"]) > 600
    assert not any("| 省电" in s["text"] for s in spans)
    assert stats["oversized_spans"] > 0
    assert source.model_dump() == original
