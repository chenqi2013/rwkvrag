import hashlib
from collections.abc import Callable, Iterator
from pathlib import Path

import pyarrow.parquet as parquet
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline

from .components import create_splitter
from .config import Settings
from .lexical_index import LexicalIndex

TEXT_COLUMNS = ("text", "markdown", "content", "article", "plain_text", "body")


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
            for name in ("id", "title", "url", "uri", "language", *TEXT_COLUMNS)
            if name in available
        ]
        for batch in parquet_file.iter_batches(batch_size=256, columns=selected):
            for row_number, row in enumerate(batch.to_pylist()):
                title = str(row.get("title") or "").strip()
                if titles and title not in titles:
                    continue
                content = first_value(row, TEXT_COLUMNS)
                if not content:
                    continue
                external_id = str(row.get("id") or f"{file_path.name}:{row_number}")
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
                }
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
        content = file_path.read_text(encoding="utf-8").strip()
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
    index = lexical_index or LexicalIndex(settings.lexical_index_path)
    if recreate:
        index.recreate()
    pipeline = IngestionPipeline(
        transformations=[create_splitter(settings)],
        disable_cache=True,
    )
    documents_count = 0
    nodes_count = 0
    for document_batch in batched(documents, batch_size):
        nodes = pipeline.run(documents=document_batch, show_progress=True)
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
