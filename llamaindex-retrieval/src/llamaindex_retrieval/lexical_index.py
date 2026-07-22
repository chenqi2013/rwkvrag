import json
import logging
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import jieba
from llama_index.core.schema import BaseNode
from opencc import OpenCC


_ASCII_TOKEN = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_OPENCC = OpenCC("t2s")
jieba.setLogLevel(logging.WARNING)
_STOP_WORDS = {
    "的",
    "了",
    "吗",
    "呢",
    "是",
    "有",
    "和",
    "与",
    "哪个",
    "哪些",
    "哪里",
    "什么",
    "怎么",
    "怎么样",
    "多少",
}
_QUERY_EXPANSIONS = {
    "最多": ("第一", "最大", "人口大省"),
    "首都": ("国都", "政治中心"),
    "功绩": ("功业", "贡献", "成就"),
    "首播": ("播出", "上档"),
    "试播": ("开始试播", "开播"),
}
_PROXIMITY_EXPANSIONS = {"中国": ("中华人民共和国",)}


@dataclass(frozen=True)
class LexicalResult:
    node_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    score: float


def lexical_tokens(text: str) -> list[str]:
    """Normalize Chinese and produce classic search-engine tokens."""
    normalized = _OPENCC.convert(text.lower())
    tokens: list[str] = _ASCII_TOKEN.findall(normalized)
    for run in _CJK_RUN.findall(normalized):
        tokens.extend(token.strip() for token in jieba.cut_for_search(run))
    return [token for token in tokens if token and token not in _STOP_WORDS]


def query_tokens(text: str) -> list[str]:
    tokens = lexical_tokens(text)
    expanded = list(tokens)
    for token in tokens:
        for synonym in _QUERY_EXPANSIONS.get(token, ()):
            expanded.extend(lexical_tokens(synonym))
    return list(dict.fromkeys(expanded))


def proximity_token_sets(text: str) -> list[set[str]]:
    base = set(lexical_tokens(text))
    variants = [base]
    for token in base:
        for synonym in _PROXIMITY_EXPANSIONS.get(token, ()):
            variants.append((base - {token}) | set(lexical_tokens(synonym)))
    return variants


def _fts_query(text: str) -> str:
    tokens = query_tokens(text)
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def _normalized_text(text: str) -> str:
    normalized = _OPENCC.convert(text.lower())
    return "".join(character for character in normalized if character.isalnum())


def _metadata_tags(metadata: dict[str, Any]) -> str:
    values: list[str] = []
    for key in ("question", "tags", "keywords", "kind", "source", "section", "title"):
        value = metadata.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value is not None:
            values.append(str(value))
    return " ".join(values)


def _proximity_bonus(text: str, tokens: set[str]) -> float:
    """Reward chunks where the original query terms occur close together."""
    if len(tokens) < 2:
        return 0.0
    normalized = _OPENCC.convert(text.lower())
    positions = {
        token: [match.start() for match in re.finditer(re.escape(token), normalized)]
        for token in tokens
    }
    if any(not values for values in positions.values()):
        return 0.0
    anchor = next(iter(tokens))
    spans = []
    for position in positions[anchor]:
        selected = [position]
        for token in tokens - {anchor}:
            selected.append(min(positions[token], key=lambda other: abs(position - other)))
        spans.append(max(selected) - min(selected))
    nearest = min(spans)
    if nearest <= 24:
        return 2.0
    if nearest <= 80:
        return 1.0
    if nearest <= 180:
        return 1.2
    return 0.0


class LexicalIndex:
    """Local BM25 index backed by SQLite FTS5.

    SQLite keeps the search index independent from Qdrant, so documents can be
    searched without generating or storing embeddings.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._initialize()

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS lexical_documents (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL UNIQUE,
                    document_id TEXT NOT NULL,
                    file_id TEXT NOT NULL DEFAULT '',
                    knowledge_base_id TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    uri TEXT,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    full_answer TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS lexical_documents_kb_idx
                    ON lexical_documents(knowledge_base_id);
                CREATE INDEX IF NOT EXISTS lexical_documents_file_idx
                    ON lexical_documents(file_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS lexical_fts USING fts5(
                    node_id UNINDEXED,
                    document_id UNINDEXED,
                    knowledge_base_id UNINDEXED,
                    body,
                    title,
                    tags,
                    tokenize = 'unicode61 remove_diacritics 2'
                );
                """
            )

    def recreate(self) -> None:
        with self._lock, self._connection() as connection:
            connection.execute("DELETE FROM lexical_fts")
            connection.execute("DELETE FROM lexical_documents")

    def upsert_nodes(self, nodes: Iterable[BaseNode]) -> int:
        records = []
        for node in nodes:
            metadata = dict(node.metadata)
            text = node.get_content().strip()
            full_answer = str(metadata.get("full_answer") or "").strip()
            body = " ".join(part for part in (text, full_answer) if part)
            records.append(
                {
                    "node_id": str(node.node_id),
                    "document_id": str(metadata.get("document_id") or node.ref_doc_id or node.node_id),
                    "file_id": str(metadata.get("file_id") or ""),
                    "knowledge_base_id": str(metadata.get("knowledge_base_id") or ""),
                    "source": str(metadata.get("source") or ""),
                    "title": str(metadata.get("title") or ""),
                    "uri": str(metadata.get("uri")) if metadata.get("uri") else None,
                    "text": text,
                    "metadata_json": json.dumps(metadata, ensure_ascii=False, default=str),
                    "full_answer": full_answer,
                    "body_tokens": " ".join(lexical_tokens(body)),
                    "title_tokens": " ".join(lexical_tokens(str(metadata.get("title") or ""))),
                    "tags_tokens": " ".join(lexical_tokens(_metadata_tags(metadata))),
                }
            )
        if not records:
            return 0
        with self._lock, self._connection() as connection:
            for record in records:
                old = connection.execute(
                    "SELECT row_id FROM lexical_documents WHERE node_id = ?",
                    (record["node_id"],),
                ).fetchone()
                if old is not None:
                    connection.execute("DELETE FROM lexical_fts WHERE rowid = ?", (old["row_id"],))
                    connection.execute("DELETE FROM lexical_documents WHERE row_id = ?", (old["row_id"],))
                cursor = connection.execute(
                    """
                    INSERT INTO lexical_documents
                        (node_id, document_id, file_id, knowledge_base_id, source, title,
                         uri, text, metadata_json, full_answer)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    tuple(record[key] for key in (
                        "node_id", "document_id", "file_id", "knowledge_base_id", "source",
                        "title", "uri", "text", "metadata_json", "full_answer",
                    )),
                )
                row_id = cursor.lastrowid
                connection.execute(
                    """
                    INSERT INTO lexical_fts(rowid, node_id, document_id, knowledge_base_id,
                                            body, title, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row_id,
                        record["node_id"],
                        record["document_id"],
                        record["knowledge_base_id"],
                        record["body_tokens"],
                        record["title_tokens"],
                        record["tags_tokens"],
                    ),
                )
        return len(records)

    def search(
        self,
        question: str,
        *,
        candidate_k: int,
        knowledge_base_id: str | None = None,
    ) -> list[LexicalResult]:
        query = _fts_query(question)
        if not query:
            return []
        clauses = ["lexical_fts MATCH ?"]
        parameters: list[Any] = [query]
        if knowledge_base_id:
            clauses.append("d.knowledge_base_id = ?")
            parameters.append(knowledge_base_id)
        parameters.append(max(candidate_k, 1))
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT d.node_id, d.document_id, d.text, d.metadata_json,
                       d.title, d.text, bm25(lexical_fts, 1.0, 4.0, 2.0) AS bm25_rank
                FROM lexical_fts
                JOIN lexical_documents AS d ON d.row_id = lexical_fts.rowid
                WHERE {' AND '.join(clauses)}
                ORDER BY bm25_rank ASC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        if not rows:
            return []
        ranked: list[tuple[float, sqlite3.Row]] = []
        normalized_question = _normalized_text(question)
        question_token_set = set(query_tokens(question))
        for row in rows:
            title = _normalized_text(str(row["title"]))
            raw = max(0.0, -float(row["bm25_rank"]))
            if normalized_question and normalized_question in title:
                raw += 1.5
            title_tokens = set(lexical_tokens(str(row["title"])))
            if title_tokens:
                raw += 1.25 * len(question_token_set & title_tokens) / len(title_tokens)
            raw += max(
                _proximity_bonus(str(row["text"]), tokens)
                for tokens in proximity_token_sets(question)
            )
            ranked.append((raw, row))
        ranked.sort(key=lambda item: item[0], reverse=True)
        top_raw = max(raw for raw, _ in ranked) or 1.0
        results = []
        for raw, row in ranked:
            results.append(
                LexicalResult(
                    node_id=str(row["node_id"]),
                    document_id=str(row["document_id"]),
                    text=str(row["text"]),
                    metadata=json.loads(row["metadata_json"]),
                    score=min(1.0, max(0.0, raw / top_raw)),
                )
            )
        return results

    def delete_by_field(self, key: str, value: str) -> int:
        allowed = {"node_id", "document_id", "file_id", "knowledge_base_id", "source"}
        if key not in allowed:
            raise ValueError(f"不支持的索引字段：{key}")
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                f"SELECT row_id FROM lexical_documents WHERE {key} = ?", (value,)
            ).fetchall()
            for row in rows:
                connection.execute("DELETE FROM lexical_fts WHERE rowid = ?", (row["row_id"],))
            connection.execute(f"DELETE FROM lexical_documents WHERE {key} = ?", (value,))
        return len(rows)

    def list_chunks(
        self,
        *,
        knowledge_base_id: str | None,
        file_id: str | None,
        limit: int,
        offset: str | None,
    ) -> dict[str, Any]:
        clauses = []
        parameters: list[Any] = []
        if knowledge_base_id:
            clauses.append("knowledge_base_id = ?")
            parameters.append(knowledge_base_id)
        if file_id:
            clauses.append("file_id = ?")
            parameters.append(file_id)
        if offset:
            clauses.append("row_id > ?")
            parameters.append(int(offset))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit + 1)
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                f"SELECT row_id, node_id, document_id, text, metadata_json FROM lexical_documents "
                f"{where} ORDER BY row_id LIMIT ?",
                parameters,
            ).fetchall()
        has_more = len(rows) > limit
        rows = rows[:limit]
        return {
            "items": [
                {
                    "id": str(row["node_id"]),
                    "document_id": str(row["document_id"]),
                    "text": str(row["text"]),
                    "metadata": json.loads(row["metadata_json"]),
                }
                for row in rows
            ],
            "next_offset": str(rows[-1]["row_id"]) if has_more and rows else None,
        }

    def health(self) -> dict[str, Any]:
        with self._lock, self._connection() as connection:
            count = int(connection.execute("SELECT COUNT(*) FROM lexical_documents").fetchone()[0])
        return {
            "ok": True,
            "algorithm": "BM25",
            "documents": count,
            "path": str(self.path),
        }
