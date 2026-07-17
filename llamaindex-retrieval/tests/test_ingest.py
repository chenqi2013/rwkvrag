from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet

from llamaindex_retrieval.ingest import iter_finewiki_documents, iter_markdown_documents


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
