"""Generic BM25 storage and chunk retrieval; no question-type answer rules."""

import logging
from copy import deepcopy
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import jieba
from llama_index.core.schema import BaseNode
from opencc import OpenCC
from opensearchpy import OpenSearch, helpers

from .index_versions import IndexVersions, serialized_write
from .config import Settings


_ASCII_TOKEN = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_KANA_RUN = re.compile(r"[\u3040-\u30ff]+")
_OPENCC = OpenCC("t2s")
jieba.setLogLevel(logging.WARNING)


@dataclass(frozen=True)
class LexicalResult:
    node_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    score: float


def normalize_search_text(text: str) -> str:
    return _OPENCC.convert(text.lower())


def _tokens_from_normalized_text(normalized: str) -> list[str]:
    tokens: list[str] = _ASCII_TOKEN.findall(normalized)
    tokens.extend(_KANA_RUN.findall(normalized))
    for run in _CJK_RUN.findall(normalized):
        tokens.extend(token.strip() for token in jieba.cut_for_search(run))
    return [token for token in tokens if token]


def lexical_tokens(text: str) -> list[str]:
    return _tokens_from_normalized_text(normalize_search_text(text))


def entity_bigram_tokens(text: str) -> list[str]:
    normalized = normalize_search_text(text)
    tokens = _ASCII_TOKEN.findall(normalized)
    for run in _CJK_RUN.findall(normalized):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return list(dict.fromkeys(tokens))


def query_tokens(text: str) -> list[str]:
    return list(dict.fromkeys(lexical_tokens(text)))


def _metadata_tags(metadata: dict[str, Any]) -> str:
    values: list[str] = []
    for key in ("question", "tags", "keywords", "kind", "source", "section", "title"):
        value = metadata.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value is not None:
            values.append(str(value))
    return " ".join(values)


def _metadata_aliases(metadata: dict[str, Any]) -> list[str]:
    value = metadata.get("aliases")
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value).strip()] if str(value).strip() else []


class LexicalIndex:
    def __init__(self, settings: Settings, client: OpenSearch | None = None,
                 *, initialize: bool = True) -> None:
        self.settings = settings
        self.index_name = settings.opensearch_index
        self.client = client or self._create_client()
        self.versions = IndexVersions(self)
        if initialize:
            self.versions.initialize()
        self.index_name = self.versions.alias
        if initialize:
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
                    "parent_id": {"type": "keyword"},
                    "content_type": {"type": "keyword"},
                    "chunk_order": {"type": "integer"},
                    "uri": {"type": "keyword", "index": False},
                    "text": {"type": "text", "index": False},
                    "full_answer": {"type": "text", "index": False},
                    "metadata": {"type": "object", "enabled": False},
                    "body_tokens": {"type": "text", "analyzer": "rwkvrag_tokens"},
                    "title_tokens": {"type": "text", "analyzer": "rwkvrag_tokens"},
                    "tags_tokens": {"type": "text", "analyzer": "rwkvrag_tokens"},
                    "section_tokens": {"type": "text", "analyzer": "rwkvrag_tokens"},
                    "structure_tokens": {"type": "text", "analyzer": "rwkvrag_tokens"},
                    "alias_tokens": {"type": "text", "analyzer": "rwkvrag_tokens"},
                    "entity_bigram_tokens": {"type": "text", "analyzer": "rwkvrag_tokens"},
                },
            },
        }

    def ensure_index(self) -> None:
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(index=self.index_name, body=self.index_definition())
            return
        put_mapping = getattr(self.client.indices, "put_mapping", None)
        if put_mapping is not None:
            properties = self.index_definition()["mappings"]["properties"]
            put_mapping(index=self.index_name, body={"properties": properties})

    def recreate(self) -> None:
        raise RuntimeError("禁止先删除活动索引；请使用 ingest_documents(recreate=True) 构建并发布新版本")

    def _record(self, node: BaseNode) -> dict[str, Any]:
        metadata = dict(node.metadata)
        # Native evidence offsets address this exact string, including whitespace.
        text = node.get_content()
        full_answer = str(metadata.get("full_answer") or "").strip()
        body = " ".join(part for part in (text, full_answer) if part)
        aliases = _metadata_aliases(metadata)
        entity_text = " ".join((str(metadata.get("title") or ""), *aliases))
        return {
            "node_id": str(node.node_id),
            "document_id": str(metadata.get("document_id") or node.ref_doc_id or node.node_id),
            "file_id": str(metadata.get("file_id") or ""),
            "knowledge_base_id": str(metadata.get("knowledge_base_id") or ""),
            "source": str(metadata.get("source") or ""),
            "title": str(metadata.get("title") or ""),
            "parent_id": str(metadata.get("parent_id") or ""),
            "content_type": str(metadata.get("content_type") or "prose"),
            "chunk_order": int(metadata.get("chunk_order") or 0),
            "uri": str(metadata.get("uri")) if metadata.get("uri") else None,
            "text": text,
            "metadata": metadata,
            "full_answer": full_answer,
            "body_tokens": " ".join(lexical_tokens(body)),
            "title_tokens": " ".join(lexical_tokens(str(metadata.get("title") or ""))),
            "tags_tokens": " ".join(lexical_tokens(_metadata_tags(metadata))),
            "section_tokens": " ".join(lexical_tokens(str(metadata.get("section") or ""))),
            "structure_tokens": " ".join(
                lexical_tokens(
                    " ".join(
                        (
                            str(metadata.get("content_type") or ""),
                            str(metadata.get("section") or ""),
                            _metadata_tags({"keywords": metadata.get("keywords", [])}),
                        )
                    )
                )
            ),
            "alias_tokens": " ".join(lexical_tokens(" ".join(aliases))),
            "entity_bigram_tokens": " ".join(entity_bigram_tokens(entity_text)),
        }

    @serialized_write
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
        if int(success) != len(actions):
            raise RuntimeError("索引写入数量不完整，新版本不能发布")
        return int(success)

    def _bounded_bm25(self, body: dict[str, Any]) -> dict[str, Any]:
        """Keep all terms using 128-token searches and RRF for long queries.

        Six whitespace-analyzed fields stay below the 1024-clause limit.
        Ordinary queries retain their BM25 scores and one-request behavior.
        """
        match = body["query"]["bool"]["must"][0]["multi_match"]
        tokens = match["query"].split()
        if len(tokens) <= 128:
            return self.client.search(index=self.index_name, body=body)
        ranks: dict[str, float] = {}
        hits: dict[str, dict[str, Any]] = {}
        nodes: dict[str, dict[str, Any]] = {}
        identity = "document_id" if body.get("collapse") else "node_id"
        for start in range(0, len(tokens), 128):
            batch = deepcopy(body)
            batch["query"]["bool"]["must"][0]["multi_match"]["query"] = " ".join(tokens[start:start + 128])
            response = self.client.search(index=self.index_name, body=batch)
            seen = set()
            for rank, hit in enumerate(response.get("hits", {}).get("hits", []), 1):
                source = hit["_source"]
                node = str(source["node_id"])
                if node in nodes and nodes[node] != source:
                    raise ValueError(f"conflicting source identity across query batches: {node}")
                nodes[node] = source
                key = str(hit["_source"][identity])
                if key in seen:
                    continue
                seen.add(key)
                ranks[key] = ranks.get(key, 0.0) + 1 / (60 + rank)
                hits.setdefault(key, hit)
        ordered = sorted(ranks, key=lambda key: (-ranks[key], key))[:body["size"]]
        return {"hits": {"hits": [{**hits[key], "_score": ranks[key]} for key in ordered]}}

    def search_chunks(
        self,
        query: str,
        *,
        candidate_k: int,
        knowledge_base_id: str | None = None,
        document_ids: list[str] | None = None,
        collapse_documents: bool = False,
    ) -> list[LexicalResult]:
        """BM25 chunk ranking for the native pipeline, without document quotas.

        Return complete indexed text. Semantic selection belongs to the resolver;
        this method does not substitute lead chunks or apply entity-specific rules.
        """
        tokens = query_tokens(query)
        if not tokens:
            return []
        filters = (
            [{"term": {"knowledge_base_id": knowledge_base_id}}]
            if knowledge_base_id else []
        )
        if document_ids is not None:
            if not document_ids:
                return []
            filters.append({"terms": {"document_id": document_ids}})
        body = {
            "size": candidate_k,
            "track_total_hits": False,
            "_source": ["node_id", "document_id", "text", "metadata"],
            "query": {"bool": {
                "must": [{"multi_match": {
                    "query": " ".join(tokens),
                    "fields": ["body_tokens", "title_tokens^3", "alias_tokens^3",
                               "tags_tokens^1.5", "section_tokens^2", "structure_tokens^1.5"],
                    "type": "most_fields", "operator": "or",
                }}],
                "filter": filters,
            }},
        }
        if collapse_documents:
            body["collapse"] = {"field": "document_id"}
        response = self._bounded_bm25(body)
        return [LexicalResult(
            node_id=str(hit["_source"]["node_id"]),
            document_id=str(hit["_source"]["document_id"]),
            text=str(hit["_source"]["text"]),
            metadata=dict(hit["_source"].get("metadata") or {}),
            score=float(hit.get("_score") or 0),
        ) for hit in response.get("hits", {}).get("hits", [])]

    @serialized_write
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
            "configured_index": self.settings.opensearch_index,
            "index_version": self.versions.current(),
            "cluster_status": cluster.get("status"),
            "version": (info.get("version") or {}).get("number"),
        }

    def close(self) -> None:
        self.client.close()
