import json
import sqlite3
from collections.abc import Callable
from pathlib import Path

from llama_index.core.schema import TextNode

from .config import Settings
from .lexical_index import LexicalIndex


def migrate_from_sqlite(
    settings: Settings,
    *,
    path: Path | None = None,
    batch_size: int = 500,
    recreate: bool = False,
    progress_callback: Callable[[int], None] | None = None,
) -> dict[str, int]:
    sqlite_path = (path or settings.sqlite_migration_path).expanduser().resolve()
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"SQLite 索引不存在：{sqlite_path}")
    index = LexicalIndex(settings)
    if recreate:
        index.recreate()
    index.client.indices.put_settings(
        index=index.index_name,
        body={"index": {"refresh_interval": "-1"}},
    )
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    processed = 0
    indexed = 0
    try:
        cursor = connection.execute(
            """
            SELECT node_id, document_id, file_id, knowledge_base_id, source,
                   title, uri, text, metadata_json, full_answer
            FROM lexical_documents
            ORDER BY row_id
            """
        )
        while rows := cursor.fetchmany(batch_size):
            nodes = []
            for row in rows:
                metadata = json.loads(str(row["metadata_json"]))
                for key in (
                    "document_id",
                    "file_id",
                    "knowledge_base_id",
                    "source",
                    "title",
                    "uri",
                ):
                    if row[key] is not None:
                        metadata[key] = row[key]
                if row["full_answer"]:
                    metadata["full_answer"] = row["full_answer"]
                nodes.append(
                    TextNode(
                        id_=str(row["node_id"]),
                        text=str(row["text"]),
                        metadata=metadata,
                    )
                )
            indexed += index.upsert_nodes(nodes)
            processed += len(rows)
            if progress_callback is not None:
                progress_callback(processed)
    finally:
        connection.close()
        index.client.indices.put_settings(
            index=index.index_name,
            body={"index": {"refresh_interval": settings.opensearch_refresh_interval}},
        )
        index.refresh()
        index.close()
    return {"processed": processed, "indexed": indexed}
