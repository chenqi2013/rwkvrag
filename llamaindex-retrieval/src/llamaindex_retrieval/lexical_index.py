import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import jieba
from llama_index.core.schema import BaseNode
from opencc import OpenCC
from opensearchpy import OpenSearch, helpers

from .config import Settings


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
_RELATION_BOOSTS = (
    ({"中国", "首都"}, "中华人民共和国 首都", 20.0),
    ({"中国", "人口", "最多", "省"}, "第一 人口 大省", 4.0),
    ({"中国", "人口", "最多", "省"}, "目前 人口", 5.0),
)


@dataclass(frozen=True)
class LexicalResult:
    node_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    score: float


def lexical_tokens(text: str) -> list[str]:
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


def relation_boosts(text: str) -> list[tuple[str, float]]:
    tokens = set(lexical_tokens(text))
    return [
        (phrase, boost)
        for required, phrase, boost in _RELATION_BOOSTS
        if required <= tokens
    ]


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
    def __init__(self, settings: Settings, client: OpenSearch | None = None) -> None:
        self.settings = settings
        self.index_name = settings.opensearch_index
        self.client = client or self._create_client()
        self.ensure_index()

    def _create_client(self) -> OpenSearch:
        authentication = None
        if self.settings.opensearch_username:
            authentication = (
                self.settings.opensearch_username,
                self.settings.opensearch_password or "",
            )
        return OpenSearch(
            hosts=[self.settings.opensearch_url],
            http_auth=authentication,
            verify_certs=self.settings.opensearch_verify_certs,
            http_compress=True,
            timeout=self.settings.opensearch_timeout,
            max_retries=3,
            retry_on_timeout=True,
        )

    def index_definition(self) -> dict[str, Any]:
        return {
            "settings": {
                "index": {
                    "number_of_shards": self.settings.opensearch_shards,
                    "number_of_replicas": self.settings.opensearch_replicas,
                    "refresh_interval": self.settings.opensearch_refresh_interval,
                },
                "analysis": {
                    "analyzer": {
                        "rwkvrag_tokens": {
                            "type": "custom",
                            "tokenizer": "whitespace",
                            "filter": ["lowercase"],
                        }
                    }
                },
            },
            "mappings": {
                "dynamic": False,
                "properties": {
                    "node_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "file_id": {"type": "keyword"},
                    "knowledge_base_id": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "title": {"type": "keyword", "ignore_above": 2048},
                    "uri": {"type": "keyword", "index": False},
                    "text": {"type": "text", "index": False},
                    "full_answer": {"type": "text", "index": False},
                    "metadata": {"type": "object", "enabled": False},
                    "body_tokens": {"type": "text", "analyzer": "rwkvrag_tokens"},
                    "title_tokens": {"type": "text", "analyzer": "rwkvrag_tokens"},
                    "tags_tokens": {"type": "text", "analyzer": "rwkvrag_tokens"},
                },
            },
        }

    def ensure_index(self) -> None:
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(index=self.index_name, body=self.index_definition())

    def recreate(self) -> None:
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
        self.client.indices.create(index=self.index_name, body=self.index_definition())

    def _record(self, node: BaseNode) -> dict[str, Any]:
        metadata = dict(node.metadata)
        text = node.get_content().strip()
        full_answer = str(metadata.get("full_answer") or "").strip()
        body = " ".join(part for part in (text, full_answer) if part)
        return {
            "node_id": str(node.node_id),
            "document_id": str(metadata.get("document_id") or node.ref_doc_id or node.node_id),
            "file_id": str(metadata.get("file_id") or ""),
            "knowledge_base_id": str(metadata.get("knowledge_base_id") or ""),
            "source": str(metadata.get("source") or ""),
            "title": str(metadata.get("title") or ""),
            "uri": str(metadata.get("uri")) if metadata.get("uri") else None,
            "text": text,
            "metadata": metadata,
            "full_answer": full_answer,
            "body_tokens": " ".join(lexical_tokens(body)),
            "title_tokens": " ".join(lexical_tokens(str(metadata.get("title") or ""))),
            "tags_tokens": " ".join(lexical_tokens(_metadata_tags(metadata))),
        }

    def upsert_nodes(self, nodes: Iterable[BaseNode]) -> int:
        actions = []
        for node in nodes:
            record = self._record(node)
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self.index_name,
                    "_id": record["node_id"],
                    "_source": record,
                }
            )
        if not actions:
            return 0
        success, _ = helpers.bulk(
            self.client,
            actions,
            chunk_size=self.settings.opensearch_bulk_size,
            request_timeout=self.settings.opensearch_bulk_timeout,
            raise_on_error=True,
            stats_only=True,
        )
        return int(success)

    def search(
        self,
        question: str,
        *,
        candidate_k: int,
        knowledge_base_id: str | None = None,
    ) -> list[LexicalResult]:
        tokens = query_tokens(question)
        if not tokens:
            return []
        filters = []
        if knowledge_base_id:
            filters.append({"term": {"knowledge_base_id": knowledge_base_id}})
        should = [
            {
                "match_phrase": {
                    "body_tokens": {
                        "query": phrase,
                        "slop": 4,
                        "boost": boost,
                    }
                }
            }
            for phrase, boost in relation_boosts(question)
        ]
        body = {
            "size": max(candidate_k, 1),
            "track_total_hits": False,
            "_source": ["node_id", "document_id", "text", "metadata", "title"],
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": " ".join(tokens),
                                "fields": ["body_tokens", "title_tokens^2", "tags_tokens^1.5"],
                                "type": "best_fields",
                                "operator": "or",
                            }
                        }
                    ],
                    "should": should,
                    "filter": filters,
                }
            },
        }
        response = self.client.search(index=self.index_name, body=body)
        hits = response.get("hits", {}).get("hits", [])
        ranked: list[tuple[float, dict[str, Any]]] = []
        normalized_question = _normalized_text(question)
        question_token_set = set(tokens)
        proximity_sets = proximity_token_sets(question)
        for hit in hits:
            source = dict(hit.get("_source") or {})
            title = str(source.get("title") or "")
            raw = max(0.0, float(hit.get("_score") or 0))
            if normalized_question and normalized_question in _normalized_text(title):
                raw += 1.5
            title_tokens = set(lexical_tokens(title))
            if title_tokens:
                raw += 1.25 * len(question_token_set & title_tokens) / len(title_tokens)
            raw += max(_proximity_bonus(str(source.get("text") or ""), item) for item in proximity_sets)
            ranked.append((raw, source))
        ranked.sort(key=lambda item: item[0], reverse=True)
        top_raw = max((raw for raw, _ in ranked), default=1.0) or 1.0
        return [
            LexicalResult(
                node_id=str(source.get("node_id") or ""),
                document_id=str(source.get("document_id") or source.get("node_id") or ""),
                text=str(source.get("text") or ""),
                metadata=dict(source.get("metadata") or {}),
                score=min(1.0, max(0.0, raw / top_raw)),
            )
            for raw, source in ranked
        ]

    def delete_by_field(self, key: str, value: str) -> int:
        allowed = {"node_id", "document_id", "file_id", "knowledge_base_id", "source"}
        if key not in allowed:
            raise ValueError(f"不支持的索引字段：{key}")
        response = self.client.delete_by_query(
            index=self.index_name,
            body={"query": {"term": {key: value}}},
            conflicts="proceed",
            refresh=True,
        )
        return int(response.get("deleted") or 0)

    def list_chunks(
        self,
        *,
        knowledge_base_id: str | None,
        file_id: str | None,
        limit: int,
        offset: str | None,
    ) -> dict[str, Any]:
        filters = []
        if knowledge_base_id:
            filters.append({"term": {"knowledge_base_id": knowledge_base_id}})
        if file_id:
            filters.append({"term": {"file_id": file_id}})
        body: dict[str, Any] = {
            "size": limit + 1,
            "sort": [{"node_id": "asc"}],
            "_source": ["node_id", "document_id", "text", "metadata"],
            "query": {"bool": {"filter": filters}} if filters else {"match_all": {}},
        }
        if offset:
            body["search_after"] = [offset]
        response = self.client.search(index=self.index_name, body=body)
        hits = response.get("hits", {}).get("hits", [])
        has_more = len(hits) > limit
        hits = hits[:limit]
        items = []
        for hit in hits:
            source = dict(hit.get("_source") or {})
            items.append(
                {
                    "id": str(source.get("node_id") or ""),
                    "document_id": str(source.get("document_id") or ""),
                    "text": str(source.get("text") or ""),
                    "metadata": dict(source.get("metadata") or {}),
                }
            )
        return {
            "items": items,
            "next_offset": items[-1]["id"] if has_more and items else None,
        }

    def refresh(self) -> None:
        self.client.indices.refresh(index=self.index_name)

    def health(self) -> dict[str, Any]:
        info = self.client.info()
        cluster = self.client.cluster.health(index=self.index_name)
        count = int(self.client.count(index=self.index_name).get("count") or 0)
        return {
            "ok": cluster.get("status") in {"green", "yellow"},
            "algorithm": "OpenSearch BM25",
            "documents": count,
            "url": self.settings.opensearch_url,
            "index": self.index_name,
            "cluster_status": cluster.get("status"),
            "version": (info.get("version") or {}).get("number"),
        }

    def close(self) -> None:
        self.client.close()
