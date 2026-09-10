from hashlib import sha256
from typing import Any, cast

import pytest

from llama_index.core.schema import TextNode

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.evidence_extraction import (
    EvidenceExtractionResult,
    EvidenceSpan,
)
from llamaindex_retrieval.lexical_index import (
    LexicalIndex,
    LexicalResult,
    entity_bigram_tokens,
)
from llamaindex_retrieval.schemas import SearchRequest, SourceItem
from llamaindex_retrieval.service import SearchService


def result(document_id: str, score: float) -> LexicalResult:
    return LexicalResult(
        node_id=f"node-{document_id}",
        document_id=document_id,
        text=document_id,
        metadata={"document_id": document_id},
        score=score,
    )


class FakeIndices:
    @staticmethod
    def exists(*, index: str) -> bool:
        return True


class FakeOpenSearch:
    indices = FakeIndices()

    def __init__(self) -> None:
        self.search_body: dict[str, Any] = {}
        self.search_bodies: list[dict[str, Any]] = []

    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.search_body = body
        self.search_bodies.append(body)
        return {
            "hits": {
                "hits": [
                    {
                        "_score": 3.0,
                        "_source": {
                            "node_id": "capital-node",
                            "document_id": "capital",
                            "title": "首都",
                            "text": "中华人民共和国的首都是北京。",
                            "metadata": {
                                "document_id": "capital",
                                "title": "首都",
                                "source": "wiki",
                            },
                        },
                    }
                ]
            }
        }


class FakeCapitalIndex:
    def search(
        self,
        question: str,
        *,
        candidate_k: int,
        knowledge_base_id: str | None = None,
    ) -> list[LexicalResult]:
        return [
            LexicalResult(
                node_id="capital-node",
                document_id="capital",
                text="中华人民共和国的首都是北京。",
                metadata={
                    "document_id": "capital",
                    "title": "首都",
                    "source": "finewiki-zh",
                },
                score=1.0,
            )
        ]


def test_select_results_deduplicates_documents_and_filters_low_scores() -> None:
    service = SearchService(
        Settings(relative_score_threshold=0.55),
        cast(Any, None),
    )
    selected = service._select_results(
        [
            result("capital", 1.0),
            result("capital", 0.8),
            result("country", 0.7),
            result("irrelevant", 0.2),
        ],
        top_k=5,
        min_score=0,
    )
    assert [item.document_id for item in selected] == ["capital", "country"]


def test_lexical_index_searches_chinese_and_titles() -> None:
    client = FakeOpenSearch()
    index = LexicalIndex(Settings(), client=cast(Any, client))
    results = index.search("中国首都在哪里", candidate_k=5)
    assert results
    assert results[0].document_id == "capital"
    query = client.search_body["query"]["bool"]["must"][0]["multi_match"]
    fields = {field.split("^")[0] for field in query["fields"]}
    assert {"body_tokens", "title_tokens", "alias_tokens", "tags_tokens"} <= fields
    assert {"section_tokens", "structure_tokens", "entity_bigram_tokens"} <= fields
    assert "首都" in query["query"]
    assert client.search_body["collapse"] == {"field": "document_id"}
    assert client.search_body["size"] == 5


def test_entity_bigrams_preserve_chinese_entity_boundaries() -> None:
    assert entity_bigram_tokens("深圳地铁1号线") == [
        "1",
        "深圳",
        "圳地",
        "地铁",
        "号线",
    ]


def test_index_mapping_contains_alias_and_entity_fallback_fields() -> None:
    index = LexicalIndex(Settings(), client=cast(Any, FakeOpenSearch()))
    properties = index.index_definition()["mappings"]["properties"]

    assert properties["alias_tokens"]["type"] == "text"
    assert properties["entity_bigram_tokens"]["type"] == "text"


def test_opensearch_record_contains_pretokenized_fields() -> None:
    index = LexicalIndex(Settings(), client=cast(Any, FakeOpenSearch()))
    record = index._record(
        TextNode(
            text="中华人民共和国的首都是北京。",
            metadata={"document_id": "capital", "title": "首都", "source": "wiki"},
        )
    )
    assert "北京" in record["body_tokens"]
    assert record["title_tokens"] == "首都"


def test_opensearch_record_contains_structure_fields() -> None:
    index = LexicalIndex(Settings(), client=cast(Any, FakeOpenSearch()))
    record = index._record(
        TextNode(
            text="型号：A；容量：10",
            metadata={
                "document_id": "spec",
                "title": "产品",
                "section": "规格参数",
                "content_type": "key_value",
                "parent_id": "parent-1",
                "chunk_order": 2,
                "keywords": ["参数", "属性"],
            },
        )
    )
    assert record["parent_id"] == "parent-1"
    assert record["content_type"] == "key_value"
    assert record["chunk_order"] == 2
    assert "规格" in record["section_tokens"]
    assert "参数" in record["structure_tokens"]


def test_source_item_preserves_retrieved_body_over_metadata_answer() -> None:
    full_answer = "完整答案" * 500
    item = SearchService._source_item(
        LexicalResult(
            node_id="qa-node",
            document_id="qa-1",
            text="只命中了答案的一小部分",
            metadata={
                "document_id": "qa-1",
                "source": "uploaded-document",
                "title": "问题标题",
                "full_answer": full_answer,
                "question_id": "1",
            },
            score=0.9,
        )
    )

    assert item.snippet == "只命中了答案的一小部分"
    assert item.metadata["question_id"] == "1"
    assert "full_answer" not in item.metadata


def test_source_item_preserves_wiki_reference_marks() -> None:
    item = SearchService._source_item(
        LexicalResult(
            node_id="metro-node",
            document_id="metro",
            text="站名列表：罗湖、后瑞[4]、机场东。[12]",
            metadata={
                "document_id": "metro",
                "source": "finewiki-zh",
                "title": "深圳地铁1号线",
                "content_type": "table_summary",
            },
            score=0.9,
        )
    )

    assert item.snippet == "站名列表：罗湖、后瑞[4]、机场东。[12]"


@pytest.mark.asyncio
async def test_immutable_pipeline_writer_only_receives_resolver_spans() -> None:
    question = "中国的首都是哪个城市？"
    selected_span = "中华人民共和国的首都是北京。"

    class Resolver:
        async def extract(
            self,
            value: str,
            plan: Any,
            sources: list[SourceItem],
        ) -> EvidenceExtractionResult:
            assert value == question
            return EvidenceExtractionResult(
                candidates=(
                    EvidenceSpan(
                        field_id="f1",
                        source_index=0,
                        span=selected_span,
                        content_hash=sha256(selected_span.encode("utf-8")).hexdigest(),
                    ),
                ),
                attempted_sources=1,
                completed_sources=1,
                source_signatures=(
                    (
                        sources[0].id,
                        sha256(sources[0].snippet.encode("utf-8")).hexdigest(),
                    ),
                ),
            )

    class Writer:
        received: list[SourceItem] = []

        async def generate(self, value: str, sources: list[SourceItem]) -> str:
            self.received = sources
            return "北京。[资料 1]"

    writer = Writer()
    service = SearchService(
        Settings(generation_output_mode="immutable"),
        cast(Any, FakeCapitalIndex()),
        generator=cast(Any, writer),
        evidence_extractor=cast(Any, Resolver()),
    )

    response = await service.ask(SearchRequest(question=question, top_k=1))

    assert response.answer == "北京。[资料 1]"
    assert [source.snippet for source in writer.received] == [selected_span]
    assert response.generation["trace_stages"][-1]["answer_modified"] is False
    assert response.generation["writer_trace"]["output_modified"] is False


@pytest.mark.asyncio
async def test_immutable_answer_point_fanout_records_branch_trace() -> None:
    question = "中国的首都是哪个城市？"
    selected_span = "中华人民共和国的首都是北京。"

    class Resolver:
        async def extract(
            self,
            value: str,
            plan: Any,
            sources: list[SourceItem],
        ) -> EvidenceExtractionResult:
            return EvidenceExtractionResult(
                candidates=(
                    EvidenceSpan(
                        field_id=plan.fields[0].field_id,
                        source_index=0,
                        span=selected_span,
                        content_hash=sha256(selected_span.encode("utf-8")).hexdigest(),
                    ),
                ),
                attempted_sources=len(sources),
                completed_sources=len(sources),
                source_signatures=(
                    (
                        sources[0].id,
                        sha256(sources[0].snippet.encode("utf-8")).hexdigest(),
                    ),
                ),
            )

    class Writer:
        async def generate(self, value: str, sources: list[SourceItem]) -> str:
            return "北京。[资料 1]"

    service = SearchService(
        Settings(
            generation_output_mode="immutable",
            answer_point_fanout_enabled=True,
            model_query_planning_enabled=False,
        ),
        cast(Any, FakeCapitalIndex()),
        generator=cast(Any, Writer()),
        evidence_extractor=cast(Any, Resolver()),
    )

    response = await service.ask(SearchRequest(question=question, top_k=1))

    assert response.answer == "北京。[资料 1]"
    assert response.retrieval["answer_point_fanout"] is True
    assert response.retrieval["mode"] == "answer-point-fanout+bm25+document-rrf"
    assert [branch["field"] for branch in response.retrieval["branches"]] == ["f1"]


@pytest.mark.asyncio
async def test_immutable_pipeline_reopens_selected_documents_for_passages() -> None:
    question = "深圳地铁1号线有哪些站点？"
    station_span = "车站列表：罗湖、国贸、老街、大剧院、科学馆、华强路。"

    class ExpandedIndex:
        def search(self, *args: Any, **kwargs: Any) -> list[LexicalResult]:
            return [
                LexicalResult(
                    node_id="overview",
                    document_id="metro",
                    text="深圳地铁1号线全长41.04公里，共设30个车站。",
                    metadata={"title": "深圳地铁1号线", "source": "wiki"},
                    score=1.0,
                )
            ]

        def document_passage_candidates(self, *args: Any, **kwargs: Any) -> list[LexicalResult]:
            return [
                LexicalResult(
                    node_id="overview",
                    document_id="metro",
                    text="深圳地铁1号线全长41.04公里，共设30个车站。",
                    metadata={"title": "深圳地铁1号线", "source": "wiki"},
                    score=1.0,
                ),
                LexicalResult(
                    node_id="stations",
                    document_id="metro",
                    text=station_span,
                    metadata={"title": "深圳地铁1号线", "source": "wiki"},
                    score=0.95,
                ),
            ]

    class Resolver:
        async def extract(
            self, value: str, plan: Any, sources: list[SourceItem]
        ) -> EvidenceExtractionResult:
            station_index = next(
                index for index, source in enumerate(sources) if "车站列表" in source.snippet
            )
            return EvidenceExtractionResult(
                candidates=(
                    EvidenceSpan(
                        field_id="f1",
                        source_index=station_index,
                        span=station_span,
                        content_hash=sha256(station_span.encode("utf-8")).hexdigest(),
                    ),
                ),
                attempted_sources=len(sources),
                completed_sources=len(sources),
                source_signatures=tuple(
                    (
                        source.id,
                        sha256(source.snippet.encode("utf-8")).hexdigest(),
                    )
                    for source in sources
                ),
            )

    class Writer:
        async def generate(self, value: str, sources: list[SourceItem]) -> str:
            assert sources[0].snippet == station_span
            return "罗湖、国贸、老街、大剧院、科学馆、华强路。[资料 1]"

    service = SearchService(
        Settings(
            generation_output_mode="immutable",
            model_query_planning_enabled=False,
        ),
        cast(Any, ExpandedIndex()),
        generator=cast(Any, Writer()),
        evidence_extractor=cast(Any, Resolver()),
    )

    response = await service.ask(SearchRequest(question=question, top_k=1))

    expansion = response.retrieval["document_passage_expansion"]
    assert expansion["enabled"] is True
    assert expansion["documents"][0]["expanded_count"] == 2
    assert response.answer.startswith("罗湖、国贸、老街")
