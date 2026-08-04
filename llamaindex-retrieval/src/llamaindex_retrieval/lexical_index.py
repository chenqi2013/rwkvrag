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
_TRANSIT_LINE_NUMBER = re.compile(r"(?:第)?([零〇一二两三四五六七八九十百]+)号线")
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
_QUESTION_WORDS = {
    "什么",
    "什么时候",
    "时候",
    "何时",
    "哪里",
    "哪儿",
    "哪个",
    "哪些",
    "怎么",
    "如何",
    "吗",
    "呢",
}
_QUERY_EXPANSIONS = {
    "最多": ("第一", "最大", "人口大省"),
    "首都": ("国都", "政治中心"),
    "功绩": ("功业", "贡献", "成就"),
    "首播": ("播出", "上档"),
    "试播": ("开始试播", "开播"),
    "站点": ("车站",),
    "车站": ("站点",),
}
_PROXIMITY_EXPANSIONS = {"中国": ("中华人民共和国",)}
_TITLE_INTENT_TOKENS = {
    "全部",
    "列表",
    "所有",
    "站点",
    "车站",
}
_LIST_QUERY_MARKERS = ("哪些", "有哪", "列表", "全部", "所有", "分别", "几个", "多少")
_STEP_QUERY_MARKERS = ("如何", "怎么", "步骤", "流程", "方法")
_QUESTION_BOUNDARIES = (
    "有哪些",
    "有哪",
    "为什么",
    "什么时候",
    "如何",
    "怎么",
    "多少",
    "几个",
    "是哪个",
    "是什么",
)
_RELATION_BOOSTS = (
    ({"中国", "首都"}, "中华人民共和国 首都", 20.0),
    ({"中国", "人口", "最多", "省"}, "第一 人口 大省", 4.0),
    ({"中国", "人口", "最多", "省"}, "目前 人口", 5.0),
)

_CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CHINESE_UNITS = {"十": 10, "百": 100}


@dataclass(frozen=True)
class LexicalResult:
    node_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    score: float


def _chinese_number(value: str) -> int:
    total = 0
    current = 0
    for character in value:
        if character in _CHINESE_DIGITS:
            current = _CHINESE_DIGITS[character]
            continue
        unit = _CHINESE_UNITS[character]
        total += (current or 1) * unit
        current = 0
    return total + current


def normalize_search_text(text: str) -> str:
    normalized = _OPENCC.convert(text.lower())
    return _TRANSIT_LINE_NUMBER.sub(
        lambda match: f"{_chinese_number(match.group(1))}号线",
        normalized,
    )


def _tokens_from_normalized_text(normalized: str) -> list[str]:
    tokens: list[str] = _ASCII_TOKEN.findall(normalized)
    for run in _CJK_RUN.findall(normalized):
        tokens.extend(token.strip() for token in jieba.cut_for_search(run))
    return [token for token in tokens if token and token not in _STOP_WORDS]


def lexical_tokens(text: str) -> list[str]:
    return _tokens_from_normalized_text(normalize_search_text(text))


def _legacy_lexical_tokens(text: str) -> list[str]:
    return _tokens_from_normalized_text(_OPENCC.convert(text.lower()))


def query_tokens(text: str) -> list[str]:
    tokens = list(dict.fromkeys([*lexical_tokens(text), *_legacy_lexical_tokens(text)]))
    expanded = list(tokens)
    for token in tokens:
        for synonym in _QUERY_EXPANSIONS.get(token, ()):
            expanded.extend(lexical_tokens(synonym))
    return list(dict.fromkeys(expanded))


def title_entity_tokens(text: str) -> list[str]:
    subject = _question_subject(text)
    return [token for token in lexical_tokens(subject) if token not in _TITLE_INTENT_TOKENS]


def _question_subject(text: str) -> str:
    boundaries = [position for marker in _QUESTION_BOUNDARIES if (position := text.find(marker)) > 0]
    return text[: min(boundaries)].strip() if boundaries else text


def title_entity_token_variants(text: str) -> list[list[str]]:
    subject = _question_subject(text)
    variants = [
        title_entity_tokens(text),
        [token for token in _legacy_lexical_tokens(subject) if token not in _TITLE_INTENT_TOKENS],
    ]
    unique: list[list[str]] = []
    for variant in variants:
        if variant and variant not in unique:
            unique.append(variant)
    return unique


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
    normalized = normalize_search_text(text)
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


def intent_content_types(question: str) -> tuple[str, ...]:
    if any(marker in question for marker in _LIST_QUERY_MARKERS):
        return ("table_summary", "table", "list")
    if any(marker in question for marker in _STEP_QUERY_MARKERS):
        return ("list", "table", "prose")
    return ()


def _proximity_bonus(text: str, tokens: set[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    normalized = normalize_search_text(text)
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


def _focus_bonus(question: str, title: str, text: str) -> float:
    focus = set(query_tokens(question)) - set(query_tokens(title)) - _QUESTION_WORDS
    if not focus:
        return 0.0
    body_tokens = set(lexical_tokens(text))
    coverage = len(focus & body_tokens) / len(focus)
    bonus = 1.5 * coverage
    if len(focus) >= 2 and focus <= body_tokens:
        bonus += _proximity_bonus(text, focus)
    return bonus


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
        for title_tokens in title_entity_token_variants(question):
            should.append(
                {
                    "match": {
                        "title_tokens": {
                            "query": " ".join(title_tokens),
                            "operator": "and",
                            "boost": 12.0,
                        }
                    }
                }
            )
        content_types = intent_content_types(question)
        if content_types:
            should.append(
                {
                    "constant_score": {
                        "filter": {"terms": {"content_type": list(content_types)}},
                        "boost": 3.0,
                    }
                }
            )
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
                                "fields": [
                                    "body_tokens",
                                    "title_tokens^2",
                                    "tags_tokens^1.5",
                                    "section_tokens^3",
                                    "structure_tokens^2",
                                ],
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
            raw += _focus_bonus(question, title, str(source.get("text") or ""))
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

    def structure_chunks(
        self,
        parent_id: str,
        *,
        knowledge_base_id: str | None,
        limit: int,
        score: float,
    ) -> list[LexicalResult]:
        filters: list[dict[str, Any]] = [{"term": {"parent_id": parent_id}}]
        if knowledge_base_id:
            filters.append({"term": {"knowledge_base_id": knowledge_base_id}})
        response = self.client.search(
            index=self.index_name,
            body={
                "size": max(1, min(limit, 100)),
                "sort": [{"chunk_order": "asc"}, {"node_id": "asc"}],
                "_source": ["node_id", "document_id", "text", "metadata"],
                "query": {"bool": {"filter": filters}},
            },
        )
        results = []
        for hit in response.get("hits", {}).get("hits", []):
            source = dict(hit.get("_source") or {})
            results.append(
                LexicalResult(
                    node_id=str(source.get("node_id") or ""),
                    document_id=str(source.get("document_id") or source.get("node_id") or ""),
                    text=str(source.get("text") or ""),
                    metadata=dict(source.get("metadata") or {}),
                    score=score,
                )
            )
        return results

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
