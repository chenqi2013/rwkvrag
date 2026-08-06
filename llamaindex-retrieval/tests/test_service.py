from typing import Any, cast

import pytest
from llama_index.core.schema import TextNode

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.lexical_index import (
    LexicalIndex,
    LexicalResult,
    _focus_bonus,
    lexical_tokens,
    normalize_query_text,
    query_tokens,
)
from llamaindex_retrieval.schemas import SearchRequest
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

    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.search_body = body
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
    fields = client.search_body["query"]["bool"]["must"][0]["multi_match"]["fields"]
    assert fields == [
        "body_tokens",
        "title_tokens^2",
        "tags_tokens^1.5",
        "section_tokens^3",
        "structure_tokens^2",
    ]


def test_transit_line_numbers_are_normalized() -> None:
    assert lexical_tokens("深圳地铁一号线有哪些站点") == lexical_tokens(
        "深圳地铁1号线有哪些站点"
    )
    assert lexical_tokens("广州地铁二十一号线") == lexical_tokens("广州地铁21号线")


def test_query_normalization_corrects_common_chinese_typos() -> None:
    assert normalize_query_text("中国有多少个名族") == "中国有多少个民族"
    assert "民族" in query_tokens("中国有多少个名族")


@pytest.mark.asyncio
async def test_service_searches_with_normalized_question() -> None:
    client = FakeOpenSearch()
    service = SearchService(Settings(), LexicalIndex(Settings(), client=cast(Any, client)))

    response = await service.search(SearchRequest(question="中国有多少个名族", top_k=1))

    query = client.search_body["query"]["bool"]["must"][0]["multi_match"]["query"]
    assert "民族" in query
    assert "名族" not in query
    assert response.retrieval["normalized_question"] == "中国有多少个民族"
    assert response.retrieval["query_normalized"] is True


def test_focus_bonus_reranks_chunks_by_non_title_query_terms() -> None:
    question = "深圳地铁1号线什么时候全线开通"
    title = "深圳地铁1号线"
    assert _focus_bonus(question, title, "该线路于2011年6月15日全线开通。") > _focus_bonus(
        question,
        title,
        "该线路大致呈东西走向。",
    )


def test_search_strongly_boosts_title_entities() -> None:
    client = FakeOpenSearch()
    index = LexicalIndex(Settings(), client=cast(Any, client))
    index.search("深圳地铁一号线有哪些站点", candidate_k=5)

    title_boosts = [
        item["match"]["title_tokens"]
        for item in client.search_body["query"]["bool"]["should"]
        if "match" in item
    ]
    assert {item["query"] for item in title_boosts} == {
        "1 深圳 地铁 号线",
        "深圳 地铁 一号 一号线",
    }
    assert all(item["operator"] == "and" for item in title_boosts)
    assert all(item["boost"] == 12.0 for item in title_boosts)


def test_title_entity_extraction_stops_before_question_predicate() -> None:
    client = FakeOpenSearch()
    index = LexicalIndex(Settings(), client=cast(Any, client))
    index.search("秦始皇有哪些丰功伟绩", candidate_k=5)

    title_boosts = [
        item["match"]["title_tokens"]
        for item in client.search_body["query"]["bool"]["should"]
        if "match" in item
    ]
    assert [item["query"] for item in title_boosts] == ["始皇 秦始皇"]


def test_list_queries_allow_multiple_chunks_from_one_document() -> None:
    service = SearchService(
        Settings(max_chunks_per_document=1, list_query_max_chunks_per_document=3),
        cast(Any, None),
    )
    document_limit = service._max_chunks_per_document("深圳地铁一号线有哪些站点")
    selected = service._select_results(
        [
            result("metro", 1.0),
            result("metro", 0.9),
            result("metro", 0.8),
            result("metro", 0.7),
            result("other", 0.6),
        ],
        top_k=5,
        min_score=0,
        max_chunks_per_document=document_limit,
    )

    assert document_limit == 3
    assert [item.document_id for item in selected] == ["metro", "metro", "metro", "other"]


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


def test_source_item_returns_complete_qa_answer() -> None:
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

    assert item.snippet == full_answer
    assert item.metadata["question_id"] == "1"
    assert "full_answer" not in item.metadata


@pytest.mark.asyncio
async def test_list_query_expands_all_chunks_from_the_matching_structure() -> None:
    anchor = LexicalResult(
        node_id="table-2",
        document_id="metro",
        text="部分车站",
        metadata={
            "document_id": "metro",
            "title": "测试线路",
            "section": "测试线路 > 车站",
            "parent_id": "station-table",
            "content_type": "table",
            "chunk_order": 2,
        },
        score=0.9,
    )
    misleading = LexicalResult(
        node_id="transfer-list",
        document_id="metro",
        text="换乘车站",
        metadata={
            "document_id": "metro",
            "title": "测试线路",
            "section": "测试线路 > 车站 > 换乘说明",
            "parent_id": "transfer-list",
            "content_type": "list",
            "chunk_order": 0,
        },
        score=0.95,
    )
    siblings = [
        LexicalResult(
            node_id=f"table-{index}",
            document_id="metro",
            text=f"表格片段{index}",
            metadata={
                "document_id": "metro",
                "parent_id": "station-table",
                "content_type": "table_summary" if index == 0 else "table",
                "chunk_order": index,
            },
            score=0.9,
        )
        for index in range(4)
    ]

    class FakeStructureIndex:
        def structure_chunks(self, parent_id: str, **kwargs: Any) -> list[LexicalResult]:
            assert parent_id == "station-table"
            return siblings

    service = SearchService(Settings(), cast(Any, FakeStructureIndex()))
    expanded, did_expand = await service._expand_structured_results(
        "测试线路有哪些车站",
        [result("metro", 1.0), misleading, anchor, result("other", 0.5)],
        knowledge_base_id=None,
        top_k=5,
    )

    assert did_expand is True
    assert [item.node_id for item in expanded[:4]] == [
        "table-0",
        "table-1",
        "table-2",
        "table-3",
    ]
