import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

from llamaindex_retrieval.ingest import (
    FINEWIKI_METADATA_COLUMNS,
    TEXT_COLUMNS,
    document_id,
    first_value,
    iter_finewiki_documents,
    iter_markdown_documents,
)


def test_iter_finewiki_documents_filters_titles(tmp_path: Path) -> None:
    path = tmp_path / "sample.parquet"
    table = pa.table(
        {
            "id": ["1", "2"],
            "title": ["首都", "光棍之家"],
            "url": ["https://example.test/capital", "https://example.test/movie"],
            "text": ["中华人民共和国的首都是北京。", "这是一部电影。"],
        }
    )
    parquet.write_table(table, path)
    documents = list(iter_finewiki_documents(path, titles={"首都"}))
    assert len(documents) == 1
    assert documents[0].metadata["title"] == "首都"
    assert documents[0].metadata["source"] == "finewiki-zh"
    assert documents[0].metadata["knowledge_base_id"] == "default"


def test_iter_markdown_documents_reads_file_and_directory(tmp_path: Path) -> None:
    path = tmp_path / "docs"
    path.mkdir()
    (path / "policy.md").write_text("# 报销政策\n\n需要提交发票。", encoding="utf-8")
    documents = list(iter_markdown_documents(path))
    assert len(documents) == 1
    assert documents[0].metadata["title"] == "报销政策"
    assert documents[0].metadata["kind"] == "markdown"
    assert documents[0].metadata["knowledge_base_id"] == "default"
    assert documents[0].metadata["uri"].startswith("file://")


def test_finewiki_real_schema_preserves_identity_revision_and_text(tmp_path: Path) -> None:
    path = tmp_path / "official-schema.parquet"
    # The extracted text's exact Unicode sequence matters: CRLF, combining marks,
    # astral characters and boundary whitespace must survive Parquet ingestion.
    content = " \t\r\n維基百科 e\u0301 / é / 😀\r\n正文。\n\t "
    row = {
        "page_id": 42,
        "title": "示例文章",
        "wikiname": "zhwiki",
        "in_language": "zh",
        "version": "123456789",
        "date_modified": "2026-01-02T03:04:05Z",
        "wikidata_id": "Q42",
        "url": "https://zh.wikipedia.org/wiki/示例文章",
        "text": content,
    }
    parquet.write_table(pa.Table.from_pylist([row]), path)

    document = next(
        iter_finewiki_documents(path, knowledge_base_id="wiki-kb", import_job_id="import-1")
    )

    assert document.text == content
    assert (
        document.metadata["source_text_sha256"]
        == hashlib.sha256(content.encode("utf-8")).hexdigest()
    )
    assert document.metadata["source_text_encoding"] == "utf-8"
    assert document.metadata["external_id"] == "zhwiki/42"
    assert document.id_ == document_id("finewiki-zh", "zhwiki/42", "示例文章")
    assert document.metadata["document_id"] == document.id_
    assert document.metadata["file_id"] == document.id_
    assert document.metadata["knowledge_base_id"] == "wiki-kb"
    assert document.metadata["import_job_id"] == "import-1"
    assert document.metadata["uri"] == row["url"]
    assert document.metadata["source_row_index"] == 0
    for key in FINEWIKI_METADATA_COLUMNS:
        if key in row:
            assert document.metadata[key] == row[key]
    for key in (
        *FINEWIKI_METADATA_COLUMNS,
        "source_text_sha256",
        "source_text_encoding",
        "external_id",
        "source_row_index",
    ):
        assert key in document.excluded_embed_metadata_keys
        assert key in document.excluded_llm_metadata_keys
    assert "title" not in document.excluded_embed_metadata_keys


@pytest.mark.parametrize("explicit_id", ["old-id", " old-id ", 0])
def test_finewiki_explicit_id_retains_existing_hash_rule(tmp_path: Path, explicit_id) -> None:
    path = tmp_path / "explicit.parquet"
    row = {
        "id": explicit_id,
        "page_id": 42,
        "wikiname": "zhwiki",
        "title": "原标题",
        "version": "99",
        "text": "原文",
    }
    parquet.write_table(pa.Table.from_pylist([row]), path)

    document = next(iter_finewiki_documents(path, source="original-source"))

    assert document.id_ == document_id("original-source", str(explicit_id), "原标题")
    assert document.metadata["external_id"] == str(explicit_id)
    assert document.metadata["page_id"] == 42


def test_finewiki_page_identity_survives_shard_move_and_revision_update(tmp_path: Path) -> None:
    first = tmp_path / "old-shard.parquet"
    second = tmp_path / "new-shard.parquet"
    row = {"page_id": 42, "wikiname": "zhwiki", "title": "同页", "version": 10, "text": "旧文"}
    parquet.write_table(pa.Table.from_pylist([row]), first)
    parquet.write_table(pa.Table.from_pylist([{**row, "version": 11, "text": "新文"}]), second)

    old = next(iter_finewiki_documents(first))
    new = next(iter_finewiki_documents(second))

    assert old.id_ == new.id_
    assert old.metadata["version"] == 10
    assert new.metadata["version"] == 11
    assert old.metadata["source_text_sha256"] != new.metadata["source_text_sha256"]


@pytest.mark.parametrize("namespace_column", ["wikiname", "in_language", "language"])
def test_finewiki_page_ids_are_namespaced_by_wiki_or_language(
    tmp_path: Path, namespace_column: str
) -> None:
    path = tmp_path / "languages.parquet"
    namespaces = ["zhwiki", "enwiki"] if namespace_column == "wikiname" else ["zh", "en"]
    rows = [
        {"page_id": 42, namespace_column: language, "title": "同名", "text": "正文"}
        for language in namespaces
    ]
    parquet.write_table(pa.Table.from_pylist(rows), path)

    documents = list(iter_finewiki_documents(path))

    assert len({document.id_ for document in documents}) == 2
    assert [document.metadata[namespace_column] for document in documents] == namespaces


def test_finewiki_fallback_ids_do_not_repeat_across_batches(tmp_path: Path) -> None:
    path = tmp_path / "many.parquet"
    # All titles are intentionally equal so the title hash cannot conceal a
    # repeated fallback ID at the 256-row batch boundary.
    parquet.write_table(
        pa.table({"title": ["同名"] * 520, "text": [f"正文 {i}" for i in range(520)]}), path
    )

    documents = list(iter_finewiki_documents(path))

    assert len(documents) == len({document.id_ for document in documents}) == 520
    assert [document.metadata["source_row_index"] for document in documents] == list(range(520))
    for i in [0, 255, 256, 511, 512, 519]:
        assert documents[i].id_ == document_id("finewiki-zh", f"many.parquet:{i}", "同名")
        assert documents[i].metadata["external_id"] == f"many.parquet:{i}"


def test_finewiki_filtered_and_blank_rows_keep_original_row_positions(tmp_path: Path) -> None:
    path = tmp_path / "filtered.parquet"
    selected = {0, 256, 257, 519}
    rows = [
        {"title": "保留" if i in selected else "跳过", "text": " \t" if i == 257 else f"正文 {i}"}
        for i in range(520)
    ]
    parquet.write_table(pa.Table.from_pylist(rows), path)

    full = {
        document.metadata["source_row_index"]: document
        for document in iter_finewiki_documents(path)
    }
    filtered = list(iter_finewiki_documents(path, titles={"保留"}))
    limited = list(iter_finewiki_documents(path, titles={"保留"}, limit=2))

    assert [document.metadata["source_row_index"] for document in filtered] == [0, 256, 519]
    assert [document.id_ for document in filtered] == [full[i].id_ for i in [0, 256, 519]]
    assert [document.id_ for document in limited] == [document.id_ for document in filtered[:2]]


@pytest.mark.parametrize("text_column", TEXT_COLUMNS)
def test_finewiki_all_supported_text_columns_preserve_boundary_whitespace(
    tmp_path: Path, text_column: str
) -> None:
    path = tmp_path / "text-column.parquet"
    content = " \n逐字正文\r\n "
    parquet.write_table(pa.table({"title": ["文章"], text_column: [content]}), path)

    document = next(iter_finewiki_documents(path))

    assert document.text == content
    assert document.metadata["source_text_sha256"] == hashlib.sha256(content.encode()).hexdigest()


def test_finewiki_blank_text_falls_back_without_changing_shared_first_value(tmp_path: Path) -> None:
    path = tmp_path / "alternate.parquet"
    content = " \n备用正文\r\n "
    row = {"title": "文章", "text": " \t", "markdown": content}
    parquet.write_table(pa.Table.from_pylist([row]), path)

    document = next(iter_finewiki_documents(path))

    assert document.text == content
    # Other callers retain the old trim semantics; only FineWiki is verbatim.
    assert first_value(row, TEXT_COLUMNS) == "备用正文"
