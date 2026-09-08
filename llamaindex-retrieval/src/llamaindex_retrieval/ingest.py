import hashlib
from collections.abc import Callable, Iterator
from pathlib import Path

import pyarrow.parquet as parquet
from llama_index.core import Document
from .components import create_splitter
from .config import Settings
from .lexical_index import LexicalIndex
from .structured_chunking import structure_aware_nodes

TEXT_COLUMNS = ("text", "markdown", "content", "article", "plain_text", "body")
FINEWIKI_METADATA_COLUMNS = (
    "page_id",
    "wikiname",
    "in_language",
    "version",
    "date_modified",
    "wikidata_id",
    "language",
)


def parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.parquet"))


def markdown_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in {".md", ".markdown"}:
            raise ValueError(f"Markdown path must end with .md or .markdown: {path}")
        return [path]
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in {".md", ".markdown"}
    )


def first_value(row: dict, names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def document_id(source: str, external_id: str, title: str) -> str:
    value = f"{source}\0{external_id}\0{title}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:24]


def finewiki_text(row: dict) -> str:
    """Choose a populated text column without changing its Unicode contents."""
    for name in TEXT_COLUMNS:
        value = row.get(name)
        if value is not None:
            content = str(value)
            if content.strip():
                return content
    return ""


def finewiki_external_id(row: dict, file_name: str, row_number: int) -> str:
    explicit_id = row.get("id")
    if explicit_id is not None and str(explicit_id).strip():
        # Preserve the existing explicit-ID hash input, including its whitespace.
        return str(explicit_id)
    page_id = row.get("page_id")
    if page_id is not None and str(page_id).strip():
        wiki = first_value(row, ("wikiname",))
        if wiki:
            return f"{wiki}/{page_id}"
        language = first_value(row, ("in_language", "language"))
        if language:
            return f"language:{language}/{page_id}"
        return f"page_id:{page_id}"
    return f"{file_name}:{row_number}"


def iter_finewiki_documents(
    path: Path,
    source: str = "finewiki-zh",
    titles: set[str] | None = None,
    limit: int = 0,
    knowledge_base_id: str = "default",
    import_job_id: str | None = None,
) -> Iterator[Document]:
    emitted = 0
    for file_path in parquet_files(path):
        parquet_file = parquet.ParquetFile(file_path)
        available = set(parquet_file.schema_arrow.names)
        selected = [
            name
            for name in ("id", "title", "url", "uri", *FINEWIKI_METADATA_COLUMNS, *TEXT_COLUMNS)
            if name in available
        ]
        row_offset = 0
        for batch in parquet_file.iter_batches(batch_size=256, columns=selected):
            for row_number, row in enumerate(batch.to_pylist(), start=row_offset):
                title = str(row.get("title") or "").strip()
                if titles and title not in titles:
                    continue
                content = finewiki_text(row)
                if not content:
                    continue
                external_id = finewiki_external_id(row, file_path.name, row_number)
                stable_id = document_id(source, external_id, title)
                uri = str(row.get("url") or row.get("uri") or "").strip()
                metadata = {
                    "document_id": stable_id,
                    "file_id": stable_id,
                    "knowledge_base_id": knowledge_base_id,
                    "source": source,
                    "title": title,
                    "uri": uri,
                    "file": file_path.name,
                    "kind": "finewiki",
                    "external_id": external_id,
                    # Hash the complete extracted text encoded as UTF-8, not the
                    # Parquet storage bytes. Do not normalize Unicode or newlines.
                    "source_text_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "source_text_encoding": "utf-8",
                    "source_row_index": row_number,
                }
                metadata.update(
                    {key: row[key] for key in FINEWIKI_METADATA_COLUMNS if row.get(key) is not None}
                )
                if import_job_id:
                    metadata["import_job_id"] = import_job_id
                excluded = [
                    "document_id",
                    "file_id",
                    "knowledge_base_id",
                    "source",
                    "uri",
                    "file",
                    "kind",
                    "import_job_id",
                    "external_id",
                    "source_text_sha256",
                    "source_text_encoding",
                    "source_row_index",
                    *FINEWIKI_METADATA_COLUMNS,
                ]
                yield Document(
                    id_=stable_id,
                    text=content,
                    metadata=metadata,
                    excluded_embed_metadata_keys=excluded,
                    excluded_llm_metadata_keys=excluded,
                )
                emitted += 1
                if limit and emitted >= limit:
                    return
            # Source row numbers count skipped/filtered rows and cross batches.
            row_offset += batch.num_rows


def batched(documents: Iterator[Document], batch_size: int) -> Iterator[list[Document]]:
    batch: list[Document] = []
    for document in documents:
        batch.append(document)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_markdown_documents(
    path: Path,
    source: str = "local-markdown",
    limit: int = 0,
    knowledge_base_id: str = "default",
) -> Iterator[Document]:
    emitted = 0
    for file_path in markdown_files(path):
        content = file_path.read_text(encoding="utf-8")
        if not content:
            continue
        title = next(
            (
                line.removeprefix("# ").strip()
                for line in content.splitlines()
                if line.strip().startswith("# ")
            ),
            file_path.stem,
        )
        stable_id = document_id(source, str(file_path.resolve()), title)
        excluded = [
            "document_id",
            "file_id",
            "knowledge_base_id",
            "source",
            "uri",
            "file",
            "kind",
            "path",
        ]
        yield Document(
            id_=stable_id,
            text=content,
            metadata={
                "document_id": stable_id,
                "file_id": stable_id,
                "knowledge_base_id": knowledge_base_id,
                "source": source,
                "title": title,
                "uri": file_path.resolve().as_uri(),
                "file": file_path.name,
                "path": str(file_path.resolve()),
                "kind": "markdown",
            },
            excluded_embed_metadata_keys=excluded,
            excluded_llm_metadata_keys=excluded,
        )
        emitted += 1
        if limit and emitted >= limit:
            return


def ingest_documents(
    settings: Settings,
    documents: Iterator[Document],
    batch_size: int,
    recreate: bool,
    progress_callback: Callable[[int, int], None] | None = None,
    lexical_index: LexicalIndex | None = None,
) -> dict[str, int]:
    index = lexical_index or LexicalIndex(settings)
    if recreate:
        index.recreate()
    splitter = create_splitter(settings)
    documents_count = 0
    nodes_count = 0
    for document_batch in batched(documents, batch_size):
        if settings.rag_pipeline == "rwkv":
            from .verbatim_chunking import verbatim_nodes

            nodes = [node for document in document_batch for node in verbatim_nodes(
                document, chunk_characters=settings.native_ingest_chunk_characters,
                overlap_characters=settings.native_ingest_overlap_characters,
            )]
        else:
            nodes = [
                node
                for document in document_batch
                for node in structure_aware_nodes(document, splitter)
            ]
        index.upsert_nodes(nodes)
        documents_count += len(document_batch)
        nodes_count += len(nodes)
        if progress_callback is not None:
            progress_callback(documents_count, nodes_count)
    return {"documents": documents_count, "nodes": nodes_count}


def ingest_finewiki(
    settings: Settings,
    path: Path,
    source: str,
    titles: set[str] | None,
    limit: int,
    batch_size: int,
    recreate: bool,
    knowledge_base_id: str = "default",
    import_job_id: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    lexical_index: LexicalIndex | None = None,
) -> dict[str, int]:
    documents = iter_finewiki_documents(
        path,
        source=source,
        titles=titles,
        limit=limit,
        knowledge_base_id=knowledge_base_id,
        import_job_id=import_job_id,
    )
    return ingest_documents(
        settings,
        documents,
        batch_size,
        recreate,
        progress_callback=progress_callback,
        lexical_index=lexical_index,
    )


def ingest_markdown(
    settings: Settings,
    path: Path,
    source: str,
    limit: int,
    batch_size: int,
    recreate: bool,
    knowledge_base_id: str = "default",
    progress_callback: Callable[[int, int], None] | None = None,
    lexical_index: LexicalIndex | None = None,
) -> dict[str, int]:
    documents = iter_markdown_documents(
        path,
        source=source,
        limit=limit,
        knowledge_base_id=knowledge_base_id,
    )
    return ingest_documents(
        settings,
        documents,
        batch_size,
        recreate,
        progress_callback=progress_callback,
        lexical_index=lexical_index,
    )


def ingest_uploaded_documents(
    settings: Settings,
    documents: list[Document],
    batch_size: int = 8,
    progress_callback: Callable[[int, int], None] | None = None,
    lexical_index: LexicalIndex | None = None,
) -> dict[str, int]:
    return ingest_documents(
        settings,
        iter(documents),
        batch_size,
        recreate=False,
        progress_callback=progress_callback,
        lexical_index=lexical_index,
    )
