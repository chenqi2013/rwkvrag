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
from .evidence_quality import is_repetitive_garbage


_ASCII_TOKEN = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_KANA_RUN = re.compile(r"[\u3040-\u30ff]+")
_OPENCC = OpenCC("t2s")
jieba.setLogLevel(logging.WARNING)
_STOP_WORDS: set[str] = set()


@dataclass(frozen=True)
class LexicalResult:
    node_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    score: float


@dataclass(frozen=True)
class _RankedChunk:
    source: dict[str, Any]
    document_id: str
    page_score: float
    passage_score: float


def normalize_search_text(text: str) -> str:
    normalized = _OPENCC.convert(text.lower())
    return normalized


def normalize_query_text(text: str) -> str:
    return normalize_search_text(text)


def _tokens_from_normalized_text(normalized: str) -> list[str]:
    tokens: list[str] = _ASCII_TOKEN.findall(normalized)
    tokens.extend(_KANA_RUN.findall(normalized))
    for run in _CJK_RUN.findall(normalized):
        tokens.extend(token.strip() for token in jieba.cut_for_search(run))
    return [token for token in tokens if token and token not in _STOP_WORDS]


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


def _legacy_lexical_tokens(text: str) -> list[str]:
    return _tokens_from_normalized_text(_OPENCC.convert(text.lower()))


def query_tokens(text: str) -> list[str]:
    return list(dict.fromkeys(_tokens_from_normalized_text(normalize_query_text(text))))


def _model_query_tokens(text: str) -> list[str]:
    """Tokenize a planner-produced query without semantic rule expansion.

    In the model pipeline, query variants are supplied by the planner.  The
    index must not add answer- or topic-specific synonyms a second time.
    Legacy callers keep ``query_tokens`` for backwards compatibility.
    """
    return list(dict.fromkeys(_tokens_from_normalized_text(normalize_search_text(text))))


def title_entity_tokens(text: str) -> list[str]:
    return lexical_tokens(text)


def _question_subject(text: str) -> str:
    return text.strip(" ？?，,。；;")


def title_entity_token_variants(text: str) -> list[list[str]]:
    normalized = normalize_query_text(text)
    legacy_subject = _question_subject(text)
    variants = [
        title_entity_tokens(normalized),
        [
            token
            for token in _legacy_lexical_tokens(legacy_subject)
            if token
        ],
    ]
    unique: list[list[str]] = []
    for variant in variants:
        if variant and variant not in unique:
            unique.append(variant)
    return unique


def proximity_token_sets(text: str) -> list[set[str]]:
    return [set(query_tokens(text))]


def relation_boosts(text: str) -> list[tuple[str, float]]:
    return []


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


def _metadata_aliases(metadata: dict[str, Any]) -> list[str]:
    value = metadata.get("aliases")
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value).strip()] if str(value).strip() else []


def intent_content_types(question: str) -> tuple[str, ...]:
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
    focus = set(query_tokens(question)) - set(query_tokens(title))
    if not focus:
        return 0.0
    body_tokens = set(lexical_tokens(text))
    coverage = len(focus & body_tokens) / len(focus)
    bonus = 1.5 * coverage
    if len(focus) >= 2 and focus <= body_tokens:
        bonus += _proximity_bonus(text, focus)
    return bonus


def _chunk_order(source: dict[str, Any]) -> int:
    metadata = dict(source.get("metadata") or {})
    try:
        return int(metadata.get("chunk_order") or source.get("chunk_order") or 0)
    except (TypeError, ValueError):
        return 0


def _passage_score(
    question: str,
    source: dict[str, Any],
    *,
    page_score: float,
    proximity_sets: list[set[str]],
) -> float:
    metadata = dict(source.get("metadata") or {})
    title = str(source.get("title") or metadata.get("title") or "")
    text = str(source.get("text") or "")
    content_type = str(metadata.get("content_type") or source.get("content_type") or "prose")
    chunk_order = max(0, _chunk_order(source))
    structured_bonus = {
        "table_summary": 0.8,
        "table": 0.6,
        "list": 0.2,
        "key_value": 0.1,
    }.get(content_type, 0.0)
    lead_bonus = 0.08 / (1.0 + chunk_order)
    focus = _focus_bonus(question, title, text)
    proximity = max(_proximity_bonus(text, item) for item in proximity_sets)
    return page_score * 0.35 + focus * 1.6 + proximity + structured_bonus + lead_bonus


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
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
        self.client.indices.create(index=self.index_name, body=self.index_definition())

    def _record(self, node: BaseNode) -> dict[str, Any]:
        metadata = dict(node.metadata)
        text = node.get_content().strip()
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
        model_pipeline = self.settings.semantic_pipeline_enabled
        tokens = _model_query_tokens(question) if model_pipeline else query_tokens(question)
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
            for phrase, boost in (() if model_pipeline else relation_boosts(question))
        ]
        title_variants = (
            [_model_query_tokens(question)]
            if model_pipeline
            else title_entity_token_variants(question)
        )
        for title_tokens in title_variants:
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
        exact_titles = [] if model_pipeline else list(dict.fromkeys((
            _question_subject(question),
            _question_subject(normalize_query_text(question)),
        )))
        phrase_tokens = [token for token in tokens if len(token) >= 2]
        significant_query = " ".join(phrase_tokens)
        entity_bigram_query = " ".join(entity_bigram_tokens(_question_subject(question)))
        topic_tokens = phrase_tokens
        topic_titles = list(dict.fromkeys([
            *topic_tokens,
            *(f"{left}{right}" for left, right in zip(phrase_tokens, phrase_tokens[1:])),
        ]))
        content_types = () if model_pipeline else intent_content_types(question)
        topic_title_boost = 80.0 if content_types else 10.0
        def exact_title_boost(title: str) -> float:
            normalized = normalize_search_text(title).replace(" ", "")
            if content_types and len(normalized) >= 4:
                return 160.0
            return 40.0

        def topic_boost(title: str) -> float:
            # Prefer meaningful multi-token phrases such as “舱外活动” over
            # a broad subject token such as “深圳地铁”. A broad token must not
            # overpower the more specific page merely because the question is
            # asking for a list.
            return (
                min(4.0, topic_title_boost)
                if title in topic_tokens
                else topic_title_boost * 2.0
            )

        for title in topic_titles:
            # Topic overview pages are often short and otherwise lose to highly
            # repetitive detail pages (for example an individual mountain pass).
            should.append({"term": {"title": {"value": title, "boost": topic_boost(title)}}})
        for title in exact_titles:
            if title:
                should.extend([
                    {"term": {"title": {"value": f"{title} (消歧义)", "boost": 18.0}}},
                    {"term": {"title": {"value": f"{title} (消歧義)", "boost": 18.0}}},
                ])
        if content_types:
            should.append(
                {
                    "constant_score": {
                        "filter": {"terms": {"content_type": list(content_types)}},
                        "boost": 3.0,
                    }
                }
            )
        # A title match can cover many chunks from one page. Pull a wider raw
        # window so those chunks cannot crowd every other relevant page out.
        raw_candidate_k = max(candidate_k, min(candidate_k * 12, 200), 1)
        lexical_fields = [
            "body_tokens",
            "title_tokens^3",
            "alias_tokens^4",
            "tags_tokens^1.5",
            "section_tokens^3",
            "structure_tokens^2",
            "entity_bigram_tokens^0.6",
        ]
        recall_queries: list[dict[str, Any]] = [
            {
                "multi_match": {
                    "query": " ".join(tokens),
                    "fields": lexical_fields,
                    "type": "best_fields",
                    "operator": "or",
                }
            }
        ]
        if significant_query:
            recall_queries.extend([
                {
                    "multi_match": {
                        "query": significant_query,
                        "fields": [
                            "title_tokens^8",
                            "alias_tokens^7",
                            "section_tokens^4",
                            "body_tokens^2",
                        ],
                        "type": "phrase",
                        "slop": 2,
                        "boost": 2.0,
                    }
                },
                {
                    "multi_match": {
                        "query": significant_query,
                        "fields": [
                            "title_tokens^5",
                            "alias_tokens^5",
                            "section_tokens^3",
                            "body_tokens",
                        ],
                        "type": "cross_fields",
                        "operator": "and",
                        "boost": 1.5,
                    }
                },
            ])
        if entity_bigram_query:
            recall_queries.append(
                {
                    "match": {
                        "entity_bigram_tokens": {
                            "query": entity_bigram_query,
                            "operator": "and",
                            "boost": 1.2,
                        }
                    }
                }
            )
        body = {
            "size": raw_candidate_k,
            "track_total_hits": False,
            "_source": ["node_id", "document_id", "text", "metadata", "title"],
            "query": {
                "bool": {
                    "must": [{
                        "bool": {
                            "should": [
                                *recall_queries,
                                *(
                                    {"term": {"title": {"value": title, "boost": exact_title_boost(title)}}}
                                    for title in exact_titles
                                    if title
                                ),
                                *(
                                    {"term": {"title": {"value": title, "boost": topic_boost(title)}}}
                                    for title in topic_titles
                                    if title
                                ),
                            ],
                            "minimum_should_match": 1,
                        }
                    }],
                    "should": should,
                    "filter": filters,
                }
            },
        }
        response = self.client.search(index=self.index_name, body=body)
        hits = response.get("hits", {}).get("hits", [])
        ranked: list[_RankedChunk] = []
        normalized_question = _normalized_text(question)
        question_token_set = set(tokens)
        proximity_sets = [{*tokens}] if model_pipeline else proximity_token_sets(question)
        specific_exact_titles = {
            normalize_search_text(exact).replace(" ", "")
            for exact in exact_titles
            if content_types
            and exact
            and len(normalize_search_text(exact).replace(" ", "")) >= 4
        }
        for hit in hits:
            source = dict(hit.get("_source") or {})
            title = str(source.get("title") or "")
            metadata = dict(source.get("metadata") or {})
            if is_repetitive_garbage(str(source.get("text") or "")):
                continue
            raw = max(0.0, float(hit.get("_score") or 0))
            if normalized_question and normalized_question in _normalized_text(title):
                raw += 1.5
            title_tokens = set(lexical_tokens(title))
            if title_tokens:
                raw += 1.25 * len(question_token_set & title_tokens) / len(title_tokens)
            alias_tokens = set(lexical_tokens(" ".join(_metadata_aliases(metadata))))
            if alias_tokens:
                raw += 1.5 * len(question_token_set & alias_tokens) / len(alias_tokens)
            if not model_pipeline:
                raw += _focus_bonus(question, title, str(source.get("text") or ""))
            raw += max(_proximity_bonus(str(source.get("text") or ""), item) for item in proximity_sets)
            document_id = str(source.get("document_id") or source.get("node_id") or "")
            ranked.append(
                _RankedChunk(
                    source=source,
                    document_id=document_id,
                    page_score=raw,
                    passage_score=_passage_score(
                        question,
                        source,
                        page_score=raw,
                        proximity_sets=proximity_sets,
                    ),
                )
            )
        page_best: dict[str, _RankedChunk] = {}
        page_support: dict[str, float] = {}
        page_chunk_counts: dict[str, int] = {}
        for chunk in ranked:
            page_chunk_counts[chunk.document_id] = page_chunk_counts.get(chunk.document_id, 0) + 1
            page_support[chunk.document_id] = page_support.get(chunk.document_id, 0.0) + (
                chunk.page_score / (60.0 + page_chunk_counts[chunk.document_id])
            )
            current = page_best.get(chunk.document_id)
            if current is None or (
                chunk.passage_score,
                -_chunk_order(chunk.source),
                chunk.source.get("node_id") or "",
            ) > (
                current.passage_score,
                -_chunk_order(current.source),
                current.source.get("node_id") or "",
            ):
                page_best[chunk.document_id] = chunk

        merged = [
            (
                chunk.page_score + min(0.25, page_support.get(document_id, 0.0)),
                chunk,
            )
            for document_id, chunk in page_best.items()
        ]
        highest_page_score = max((raw for raw, _chunk in merged), default=0.0)
        merged = [
            (
                raw + highest_page_score + 1.0
                if normalize_search_text(str(chunk.source.get("title") or "")).replace(" ", "")
                in specific_exact_titles
                else raw,
                chunk,
            )
            for raw, chunk in merged
        ]
        merged.sort(key=lambda item: item[0], reverse=True)
        top_raw = max((raw for raw, _chunk in merged), default=1.0) or 1.0
        return [
            LexicalResult(
                node_id=str(chunk.source.get("node_id") or ""),
                document_id=str(
                    chunk.source.get("document_id") or chunk.source.get("node_id") or ""
                ),
                text=str(chunk.source.get("text") or ""),
                metadata=dict(chunk.source.get("metadata") or {}),
                score=min(1.0, max(0.0, raw / top_raw)),
            )
            for raw, chunk in merged
        ]

    def search_plain(
        self,
        query: str,
        *,
        candidate_k: int,
        knowledge_base_id: str | None = None,
    ) -> list[LexicalResult]:
        tokens = query_tokens(query)
        if not tokens:
            return []
        filters: list[dict[str, Any]] = []
        if knowledge_base_id:
            filters.append({"term": {"knowledge_base_id": knowledge_base_id}})
        response = self.client.search(
            index=self.index_name,
            body={
                "size": max(candidate_k, min(candidate_k * 8, 200)),
                "track_total_hits": False,
                "_source": ["node_id", "document_id", "text", "metadata"],
                "query": {
                    "bool": {
                        "must": [{
                            "multi_match": {
                                "query": " ".join(tokens),
                                "fields": [
                                    "body_tokens",
                                    "title_tokens^3",
                                    "alias_tokens^3",
                                    "tags_tokens^1.5",
                                    "section_tokens^2",
                                    "structure_tokens^1.5",
                                ],
                                "type": "best_fields",
                                "operator": "or",
                            }
                        }],
                        "filter": filters,
                    }
                },
            },
        )
        query_subject = _question_subject(normalize_query_text(query))
        subject_tokens = {
            token for token in lexical_tokens(query_subject)
            if token
            and token not in {"历史", "歷史", "上", "第一个", "第一位", "首位", "最早", "个"}
            and len(token) >= 2
        }
        normalized_subject = normalize_query_text(query_subject).replace(" ", "")

        def identity(value: str) -> str:
            compact = normalize_query_text(value).replace(" ", "")
            compact = compact.translate(str.maketrans({
                "〇": "0", "一": "1", "二": "2", "两": "2", "三": "3",
                "四": "4", "五": "5", "六": "6", "七": "7", "八": "8",
                "九": "9", "十": "10",
            }))
            for suffix in ("事变", "之变", "政变"):
                if compact.endswith(suffix):
                    compact = compact[: -len(suffix)]
                    break
            return compact

        best_by_document: dict[str, tuple[float, dict[str, Any]]] = {}
        for hit in response.get("hits", {}).get("hits", []):
            source = dict(hit.get("_source") or {})
            text = str(source.get("text") or "")
            if is_repetitive_garbage(text):
                continue
            document_id = str(
                source.get("document_id") or source.get("node_id") or ""
            )
            score = max(0.0, float(hit.get("_score") or 0.0))
            title = str(source.get("metadata", {}).get("title") or "")
            title_tokens = set(lexical_tokens(title))
            score += 4.0 * len(subject_tokens & title_tokens)
            if subject_tokens and subject_tokens <= title_tokens:
                score += 4.0
            if normalized_subject and identity(title) == identity(query_subject):
                score += 12.0
            current = best_by_document.get(document_id)
            if current is None or score > current[0]:
                best_by_document[document_id] = (score, source)
        ranked = sorted(best_by_document.values(), key=lambda item: item[0], reverse=True)
        top_score = max((score for score, _ in ranked), default=1.0) or 1.0
        return [
            LexicalResult(
                node_id=str(source.get("node_id") or ""),
                document_id=str(source.get("document_id") or source.get("node_id") or ""),
                text=str(source.get("text") or ""),
                metadata=dict(source.get("metadata") or {}),
                score=min(1.0, score / top_score),
            )
            for score, source in ranked[:candidate_k]
        ]

    def document_passage_candidates(
        self,
        query: str,
        *,
        document_id: str,
        knowledge_base_id: str | None,
        limit: int,
    ) -> list[LexicalResult]:
        """Retrieve relevant passages after document-level ranking.

        ``search_plain`` intentionally keeps one best passage per document so
        that long documents cannot dominate RRF.  Once a document has been
        selected, this method reopens that document and ranks its passages
        independently.  The operation is purely lexical and therefore works
        for prose, lists, and tables without question-specific rules.
        """

        # Reuse the index-wide query normalization (including generic
        # synonyms), so equivalent terms can rank the matching passage.
        tokens = query_tokens(query)
        if not tokens or not document_id:
            return []
        filters: list[dict[str, Any]] = [{"term": {"document_id": document_id}}]
        if knowledge_base_id:
            filters.append({"term": {"knowledge_base_id": knowledge_base_id}})
        response = self.client.search(
            index=self.index_name,
            body={
                "size": max(1, min(int(limit), 100)),
                "track_total_hits": False,
                "_source": ["node_id", "document_id", "text", "metadata"],
                "query": {
                    "bool": {
                        "filter": filters,
                        "must_not": [{"term": {"content_type": "key_value"}}],
                        "must": [{
                            "multi_match": {
                                "query": " ".join(tokens),
                                "fields": [
                                    "body_tokens",
                                    "section_tokens^3",
                                    "structure_tokens^2",
                                    "title_tokens",
                                ],
                                "type": "best_fields",
                                "operator": "or",
                                "minimum_should_match": 1,
                            }
                        }],
                    }
                },
                "sort": [{"_score": "desc"}, {"chunk_order": "asc"}, {"node_id": "asc"}],
            },
        )
        hits = [
            hit
            for hit in response.get("hits", {}).get("hits", [])
            if not is_repetitive_garbage(
                str((hit.get("_source") or {}).get("text") or "")
            )
        ]
        def passage_rank(hit: dict[str, Any]) -> tuple[float, int, str]:
            source = dict(hit.get("_source") or {})
            metadata = dict(source.get("metadata") or {})
            focus = _focus_bonus(
                query,
                str(metadata.get("title") or ""),
                str(source.get("text") or ""),
            )
            return (
                float(hit.get("_score") or 0.0) + 2.0 * focus,
                -int(metadata.get("chunk_order") or 0),
                str(source.get("node_id") or ""),
            )

        hits.sort(key=passage_rank, reverse=True)
        top_raw = max((passage_rank(hit)[0] for hit in hits), default=1.0) or 1.0
        results: list[LexicalResult] = []
        seen: set[str] = set()
        for hit in hits:
            source = dict(hit.get("_source") or {})
            node_id = str(source.get("node_id") or "")
            if not node_id or node_id in seen:
                continue
            seen.add(node_id)
            results.append(
                LexicalResult(
                    node_id=node_id,
                    document_id=str(source.get("document_id") or document_id),
                    text=str(source.get("text") or ""),
                    metadata=dict(source.get("metadata") or {}),
                    score=min(1.0, max(0.0, passage_rank(hit)[0] / top_raw)),
                )
            )
        return results

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

    def document_lead_chunk(
        self,
        document_id: str,
        *,
        knowledge_base_id: str | None,
        score: float,
    ) -> LexicalResult | None:
        """Return the earliest indexed chunk, which normally contains the article definition."""

        filters: list[dict[str, Any]] = [{"term": {"document_id": document_id}}]
        if knowledge_base_id:
            filters.append({"term": {"knowledge_base_id": knowledge_base_id}})
        response = self.client.search(
            index=self.index_name,
            body={
                "size": 100,
                "sort": [{"chunk_order": "asc"}, {"_seq_no": "asc"}],
                "_source": ["node_id", "document_id", "text", "metadata"],
                "query": {"bool": {"filter": filters}},
            },
        )
        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            return None
        candidates = [
            dict(hit.get("_source") or {})
            for hit in hits
            if not is_repetitive_garbage(str((hit.get("_source") or {}).get("text") or ""))
        ]
        def lead_priority(candidate: dict[str, Any]) -> tuple[int, int, int]:
            metadata = dict(candidate.get("metadata") or {})
            title = normalize_search_text(str(metadata.get("title") or "")).replace(" ", "")
            section = normalize_search_text(str(metadata.get("section") or "")).replace(" ", "")
            exact_root = section == title or not section
            section_depth = section.count(">")
            return (0 if exact_root else 1, section_depth, int(metadata.get("chunk_order") or 0))

        source = min(candidates, key=lead_priority, default=None)
        if source is None:
            return None
        return LexicalResult(
            node_id=str(source.get("node_id") or ""),
            document_id=str(source.get("document_id") or source.get("node_id") or ""),
            text=str(source.get("text") or ""),
            metadata=dict(source.get("metadata") or {}),
            score=score,
        )

    def document_structure_candidates(
        self,
        question: str,
        *,
        document_id: str,
        content_types: Iterable[str],
        knowledge_base_id: str | None,
        limit: int,
    ) -> list[LexicalResult]:
        filters: list[dict[str, Any]] = [
            {"term": {"document_id": document_id}},
            {"terms": {"content_type": list(dict.fromkeys(content_types))}},
        ]
        if knowledge_base_id:
            filters.append({"term": {"knowledge_base_id": knowledge_base_id}})
        tokens = query_tokens(question)
        response = self.client.search(
            index=self.index_name,
            body={
                "size": max(1, min(int(limit), 100)),
                "_source": ["node_id", "document_id", "text", "metadata"],
                "query": {
                    "bool": {
                        "filter": filters,
                        "should": [
                            {
                                "multi_match": {
                                    "query": " ".join(tokens),
                                    "fields": [
                                        "body_tokens",
                                        "section_tokens^3",
                                        "structure_tokens^3",
                                    ],
                                    "type": "best_fields",
                                    "operator": "or",
                                }
                            }
                        ],
                    }
                },
                "sort": [{"_score": "desc"}, {"chunk_order": "asc"}, {"node_id": "asc"}],
            },
        )
        hits = response.get("hits", {}).get("hits", [])
        top_raw = max((float(hit.get("_score") or 0.0) for hit in hits), default=1.0) or 1.0
        results = []
        for hit in hits:
            source = dict(hit.get("_source") or {})
            results.append(
                LexicalResult(
                    node_id=str(source.get("node_id") or ""),
                    document_id=str(source.get("document_id") or source.get("node_id") or ""),
                    text=str(source.get("text") or ""),
                    metadata=dict(source.get("metadata") or {}),
                    score=min(1.0, max(0.0, float(hit.get("_score") or 0.0) / top_raw)),
                )
            )
        return results

    def document_relation_candidates(
        self,
        question: str,
        *,
        document_id: str,
        relations: Iterable[str],
        knowledge_base_id: str | None,
        limit: int,
    ) -> list[LexicalResult]:
        """Find passages for a requested relation inside an already matched article."""

        filters: list[dict[str, Any]] = [{"term": {"document_id": document_id}}]
        if knowledge_base_id:
            filters.append({"term": {"knowledge_base_id": knowledge_base_id}})
        relation_values = tuple(dict.fromkeys(value.strip() for value in relations if value.strip()))
        relation_tokens = list(
            dict.fromkeys(
                token
                for value in relation_values
                for token in query_tokens(value)
            )
        )
        if not relation_tokens:
            return []
        question_tokens = query_tokens(question)
        response = self.client.search(
            index=self.index_name,
            body={
                "size": max(1, min(int(limit), 50)),
                "_source": ["node_id", "document_id", "text", "metadata", "title"],
                "query": {
                    "bool": {
                        "filter": filters,
                        "must": [
                            {
                                "multi_match": {
                                    "query": " ".join(relation_tokens),
                                    "fields": [
                                        "section_tokens^6",
                                        "body_tokens^3",
                                        "structure_tokens^2",
                                    ],
                                    "type": "best_fields",
                                    "operator": "or",
                                    "minimum_should_match": 1,
                                }
                            }
                        ],
                        "should": [
                            {
                                "multi_match": {
                                    "query": " ".join(question_tokens),
                                    "fields": ["body_tokens", "section_tokens^3"],
                                    "type": "best_fields",
                                    "operator": "or",
                                }
                            },
                            {
                                "constant_score": {
                                    "filter": {"term": {"content_type": "prose"}},
                                    "boost": 1.5,
                                }
                            },
                        ],
                    }
                },
            },
        )
        hits = [
            hit
            for hit in response.get("hits", {}).get("hits", [])
            if not is_repetitive_garbage(
                str((hit.get("_source") or {}).get("text") or "")
            )
        ]
        relation_token_set = set(relation_tokens)

        def relation_priority(hit: dict[str, Any]) -> tuple[float, float, int]:
            source = dict(hit.get("_source") or {})
            metadata = dict(source.get("metadata") or {})
            section_tokens = set(query_tokens(str(metadata.get("section") or "")))
            body_tokens = set(query_tokens(str(source.get("text") or "")))
            section_overlap = len(relation_token_set & section_tokens)
            body_overlap = len(relation_token_set & body_tokens)
            return (
                section_overlap * 4.0 + body_overlap,
                float(hit.get("_score") or 0.0),
                -_chunk_order(source),
            )

        hits.sort(key=relation_priority, reverse=True)
        top_raw = max(
            (float(hit.get("_score") or 0.0) for hit in hits),
            default=1.0,
        ) or 1.0
        return [
            LexicalResult(
                node_id=str((hit.get("_source") or {}).get("node_id") or ""),
                document_id=str(
                    (hit.get("_source") or {}).get("document_id")
                    or (hit.get("_source") or {}).get("node_id")
                    or ""
                ),
                text=str((hit.get("_source") or {}).get("text") or ""),
                metadata=dict((hit.get("_source") or {}).get("metadata") or {}),
                score=min(
                    1.0,
                    max(0.0, float(hit.get("_score") or 0.0) / top_raw),
                ),
            )
            for hit in hits
        ]

    def document_prose_candidates(
        self,
        question: str,
        *,
        document_id: str,
        knowledge_base_id: str | None,
        limit: int,
    ) -> list[LexicalResult]:
        """Find prose fallbacks when a chronological list structure is malformed."""

        filters: list[dict[str, Any]] = [
            {"term": {"document_id": document_id}},
            {
                "bool": {
                    "should": [
                        {"term": {"content_type": "prose"}},
                        {"bool": {"must_not": [{"exists": {"field": "content_type"}}]}},
                    ],
                    "minimum_should_match": 1,
                }
            },
        ]
        if knowledge_base_id:
            filters.append({"term": {"knowledge_base_id": knowledge_base_id}})
        response = self.client.search(
            index=self.index_name,
            body={
                "size": max(1, min(int(limit), 20)),
                "_source": ["node_id", "document_id", "text", "metadata", "title"],
                "query": {
                    "bool": {
                        "filter": filters,
                        "must": [{
                            "multi_match": {
                                "query": " ".join(query_tokens(question)),
                                "fields": ["body_tokens", "section_tokens^2", "title_tokens^2"],
                                "operator": "or",
                            }
                        }],
                    }
                },
            },
        )
        hits = [
            hit
            for hit in response.get("hits", {}).get("hits", [])
            if not is_repetitive_garbage(str((hit.get("_source") or {}).get("text") or ""))
        ]
        top_raw = max((float(hit.get("_score") or 0.0) for hit in hits), default=1.0) or 1.0
        return [
            LexicalResult(
                node_id=str((hit.get("_source") or {}).get("node_id") or ""),
                document_id=str(
                    (hit.get("_source") or {}).get("document_id")
                    or (hit.get("_source") or {}).get("node_id")
                    or ""
                ),
                text=str((hit.get("_source") or {}).get("text") or ""),
                metadata=dict((hit.get("_source") or {}).get("metadata") or {}),
                score=min(1.0, max(0.0, float(hit.get("_score") or 0.0) / top_raw)),
            )
            for hit in hits
        ]

    def document_line_section_chunks(
        self,
        line: str,
        *,
        document_id: str,
        knowledge_base_id: str | None,
        limit: int = 3,
        score: float = 1.0,
    ) -> list[LexicalResult]:
        """Return consecutive chunks for a numbered line in a flattened list page."""

        filters: list[dict[str, Any]] = [{"term": {"document_id": document_id}}]
        if knowledge_base_id:
            filters.append({"term": {"knowledge_base_id": knowledge_base_id}})
        response = self.client.search(
            index=self.index_name,
            body={
                "size": 100,
                "sort": [{"chunk_order": "asc"}, {"node_id": "asc"}],
                "_source": ["node_id", "document_id", "text", "metadata"],
                "query": {"bool": {"filter": filters}},
            },
        )
        sources = [dict(hit.get("_source") or {}) for hit in response.get("hits", {}).get("hits", [])]
        normalized_line = normalize_search_text(line).strip()
        heading = re.compile(rf"^\s*{re.escape(normalized_line)}(?:\s|前称|曾称|在|沿途)")
        other_heading = re.compile(r"^\s*\d+号线")
        anchor_index = next(
            (
                index
                for index, source in enumerate(sources)
                if heading.search(normalize_search_text(str(source.get("text") or "")))
            ),
            None,
        )
        if anchor_index is None:
            return []
        selected = []
        for source in sources[anchor_index:]:
            normalized_text = normalize_search_text(str(source.get("text") or ""))
            if selected and other_heading.search(normalized_text):
                break
            selected.append(source)
            if len(selected) >= max(1, min(limit, 10)):
                break
        return [
            LexicalResult(
                node_id=str(source.get("node_id") or ""),
                document_id=str(source.get("document_id") or source.get("node_id") or ""),
                text=str(source.get("text") or ""),
                metadata=dict(source.get("metadata") or {}),
                score=score,
            )
            for source in selected
        ]

    def document_term_candidates(
        self,
        term: str,
        *,
        document_id: str,
        knowledge_base_id: str | None,
        limit: int = 3,
    ) -> list[LexicalResult]:
        """Find same-document prose that can repair a truncated structured entity."""

        filters: list[dict[str, Any]] = [{"term": {"document_id": document_id}}]
        if knowledge_base_id:
            filters.append({"term": {"knowledge_base_id": knowledge_base_id}})
        response = self.client.search(
            index=self.index_name,
            body={
                "size": max(1, min(int(limit), 100)),
                "_source": ["node_id", "document_id", "text", "metadata"],
                "query": {
                    "bool": {
                        "filter": filters,
                        "must": [{
                            "wildcard": {
                                "body_tokens": {
                                    "value": f"*{term}站*",
                                }
                            }
                        }],
                    }
                },
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
                    score=1.0,
                )
            )
        return results

    def station_title_candidates(
        self,
        suffix: str,
        *,
        question: str,
        knowledge_base_id: str | None,
        limit: int = 10,
    ) -> list[LexicalResult]:
        """Resolve a truncated station name from dedicated station articles."""

        filters: list[dict[str, Any]] = []
        if knowledge_base_id:
            filters.append({"term": {"knowledge_base_id": knowledge_base_id}})
        response = self.client.search(
            index=self.index_name,
            body={
                "size": max(1, min(int(limit), 50)),
                "_source": ["node_id", "document_id", "text", "metadata", "title"],
                "query": {
                    "bool": {
                        "must": [{"wildcard": {"title": {"value": f"*{suffix}站"}}}],
                        "should": [{
                            "multi_match": {
                                "query": " ".join(query_tokens(question)),
                                "fields": ["body_tokens", "title_tokens^2", "tags_tokens"],
                                "operator": "or",
                            }
                        }],
                        "filter": filters,
                    }
                },
            },
        )
        results = []
        for hit in response.get("hits", {}).get("hits", []):
            source = dict(hit.get("_source") or {})
            title = str(source.get("title") or "")
            text = str(source.get("text") or "")
            if not title.endswith(f"{suffix}站"):
                continue
            results.append(
                LexicalResult(
                    node_id=str(source.get("node_id") or ""),
                    document_id=str(source.get("document_id") or source.get("node_id") or ""),
                    text=f"{title}\n{text}",
                    metadata=dict(source.get("metadata") or {}),
                    score=1.0,
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
