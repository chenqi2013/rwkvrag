from typing import Any, cast

import pytest
from llama_index.core.schema import TextNode

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.generation import EvidenceAnswerGenerator
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


class FakeOpenSearchWithPassages:
    indices = FakeIndices()

    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "hits": {
                "hits": [
                    {
                        "_score": 8.0,
                        "_source": {
                            "node_id": "metro-overview",
                            "document_id": "metro-line-1",
                            "title": "深圳地铁1号线",
                            "text": "深圳地铁1号线是深圳市的一条城市轨道交通线路。",
                            "metadata": {
                                "document_id": "metro-line-1",
                                "title": "深圳地铁1号线",
                                "source": "wiki",
                                "content_type": "prose",
                                "chunk_order": 0,
                            },
                        },
                    },
                    {
                        "_score": 6.8,
                        "_source": {
                            "node_id": "metro-stations",
                            "document_id": "metro-line-1",
                            "title": "深圳地铁1号线",
                            "text": "车站列表：罗湖、国贸、老街、大剧院、科学馆、华强路。",
                            "metadata": {
                                "document_id": "metro-line-1",
                                "title": "深圳地铁1号线",
                                "source": "wiki",
                                "content_type": "table_summary",
                                "chunk_order": 4,
                            },
                        },
                    },
                    {
                        "_score": 7.0,
                        "_source": {
                            "node_id": "metro-line-2",
                            "document_id": "metro-line-2",
                            "title": "深圳地铁2号线",
                            "text": "深圳地铁2号线连接赤湾和莲塘。",
                            "metadata": {
                                "document_id": "metro-line-2",
                                "title": "深圳地铁2号线",
                                "source": "wiki",
                            },
                        },
                    },
                ]
            }
        }


class FakeMismatchIndex:
    def search(
        self,
        question: str,
        *,
        candidate_k: int,
        knowledge_base_id: str | None = None,
    ) -> list[LexicalResult]:
        return [
            LexicalResult(
                node_id="henan-node",
                document_id="henan",
                text="河南省拥有龙门石窟、殷墟等历史文化遗产。",
                metadata={
                    "document_id": "henan",
                    "title": "河南省",
                    "source": "finewiki-zh",
                },
                score=1.0,
            )
        ]


class FakeInsufficientGenerator:
    assess_evidence = staticmethod(EvidenceAnswerGenerator.assess_evidence)

    async def generate(self, question: str, sources: Any) -> str:
        return "根据检索到的资料，无法确定。"

    async def current_model(self) -> str:
        return "test-model"


class FakeFailingGenerator:
    assess_evidence = staticmethod(EvidenceAnswerGenerator.assess_evidence)

    async def generate(self, question: str, sources: Any) -> str:
        raise AssertionError("structured list answers should not call the model")

    async def current_model(self) -> str:
        return "test-model"


class FakeStructuredListIndex:
    def search(
        self,
        question: str,
        *,
        candidate_k: int,
        knowledge_base_id: str | None = None,
    ) -> list[LexicalResult]:
        return [
            LexicalResult(
                node_id="metro-stations",
                document_id="metro",
                text=(
                    "深圳地铁1号线 > 车站\n"
                    "站名/1号线列表：罗湖、国贸、老街、大剧院、科学馆、华强路、后瑞[4]、机场东。"
                ),
                metadata={
                    "document_id": "metro",
                    "title": "深圳地铁1号线",
                    "source": "finewiki-zh",
                    "content_type": "table_summary",
                },
                score=1.0,
            ),
            LexicalResult(
                node_id="metro-history",
                document_id="metro",
                text="2011年10月14日，一趟列车在后瑞站发生故障停运。",
                metadata={
                    "document_id": "metro",
                    "title": "深圳地铁1号线",
                    "source": "finewiki-zh",
                    "content_type": "prose",
                },
                score=0.8,
            )
        ]


class FakeStructuredRowsIndex:
    def search(
        self,
        question: str,
        *,
        candidate_k: int,
        knowledge_base_id: str | None = None,
    ) -> list[LexicalResult]:
        return [
            LexicalResult(
                node_id="metro-transfer",
                document_id="metro",
                text="- 机场东站：轉乘20号线。\n- 老街站：轉乘17号线。",
                metadata={
                    "document_id": "metro",
                    "title": "深圳地铁1号线",
                    "source": "finewiki-zh",
                    "content_type": "list",
                },
                score=1.0,
            ),
            LexicalResult(
                node_id="metro-station-rows",
                document_id="metro",
                text=(
                    "站名/1号线：罗湖；所在地/1号线：罗湖区\n"
                    "站名/1号线：国贸；所在地/1号线：罗湖区\n"
                    "站名/1号线：老街；所在地/1号线：罗湖区\n"
                    "站名/1号线：大剧院；所在地/1号线：罗湖区"
                ),
                metadata={
                    "document_id": "metro",
                    "title": "深圳地铁1号线",
                    "source": "finewiki-zh",
                    "content_type": "table",
                },
                score=0.9,
            ),
        ]


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


class FakeGreatWallPassIndex:
    def search(
        self,
        question: str,
        *,
        candidate_k: int,
        knowledge_base_id: str | None = None,
    ) -> list[LexicalResult]:
        return [
            LexicalResult(
                node_id="china-experts",
                document_id="china-experts",
                text="中國通 > 著名日本中國通\n- 原日本驻华大使阿南惟茂\n- 宫本雄二\n- 二阶俊博",
                metadata={
                    "document_id": "china-experts",
                    "title": "中國通",
                    "source": "finewiki-zh",
                    "content_type": "list",
                },
                score=1.0,
            ),
            LexicalResult(
                node_id="great-wall-passes",
                document_id="great-wall",
                text=(
                    "长城 > 形制与体系 > 关隘 > 著名关城\n"
                    "- 虎山长城：明长城的东端起点。\n"
                    "- 山海关：有“天下第一关”之称。\n"
                    "- 嘉峪关：有“天下第一雄关”之称。\n"
                    "- 玉门关：丝绸之路上的重要关隘。\n"
                    "- 萧关：中国古代西北著名关口。\n"
                    "- 阳关"
                ),
                metadata={
                    "document_id": "great-wall",
                    "title": "长城",
                    "source": "finewiki-zh",
                    "content_type": "list",
                },
                score=0.9,
            ),
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
    assert {"车站", "站点", "站名"} <= set(query_tokens("深圳地铁1号线有哪些站"))


def test_query_normalization_corrects_common_chinese_typos() -> None:
    assert normalize_query_text("中国有多少个名族") == "中国有多少个民族"
    assert "民族" in query_tokens("中国有多少个名族")
    assert {"关城", "关口"} <= set(query_tokens("中国有哪些著名的长城关隘"))


@pytest.mark.asyncio
async def test_service_searches_with_normalized_question() -> None:
    client = FakeOpenSearch()
    service = SearchService(Settings(), LexicalIndex(Settings(), client=cast(Any, client)))

    response = await service.search(SearchRequest(question="中国有多少个名族", top_k=1))

    query = client.search_bodies[0]["query"]["bool"]["must"][0]["multi_match"]["query"]
    assert "民族" in query
    assert "名族" not in query
    assert response.retrieval["normalized_question"] == "中国有多少个民族"
    assert response.retrieval["query_normalized"] is True


@pytest.mark.asyncio
async def test_ask_reports_subject_anchor_mismatch() -> None:
    service = SearchService(
        Settings(),
        cast(Any, FakeMismatchIndex()),
        generator=cast(Any, FakeInsufficientGenerator()),
    )

    response = await service.ask(SearchRequest(question="秦始皇有哪些丰功伟绩？", top_k=1))

    assert response.answer == "根据检索到的资料，无法确定。"
    assert response.generation["evidence_grounded"] is False
    assert response.generation["blocked_reason"] == "insufficient_evidence"
    assert "秦始皇" in response.generation["evidence_anchors"]
    assert response.generation["matched_evidence_anchors"] == []


def test_focus_bonus_reranks_chunks_by_non_title_query_terms() -> None:
    question = "深圳地铁1号线什么时候全线开通"
    title = "深圳地铁1号线"
    assert _focus_bonus(question, title, "该线路于2011年6月15日全线开通。") > _focus_bonus(
        question,
        title,
        "该线路大致呈东西走向。",
    )


def test_search_merges_page_and_selects_best_passage() -> None:
    index = LexicalIndex(Settings(), client=cast(Any, FakeOpenSearchWithPassages()))

    results = index.search("深圳地铁一号线有哪些站点", candidate_k=2)

    assert [item.document_id for item in results] == ["metro-line-1", "metro-line-2"]
    assert results[0].node_id == "metro-stations"
    assert "车站列表" in results[0].text


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


def test_source_item_cleans_wiki_reference_marks() -> None:
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

    assert item.snippet == "站名列表：罗湖、后瑞、机场东。"


@pytest.mark.asyncio
async def test_ask_extracts_complete_structured_list_without_model_call() -> None:
    service = SearchService(
        Settings(),
        cast(Any, FakeStructuredListIndex()),
        generator=cast(Any, FakeFailingGenerator()),
    )

    response = await service.ask(SearchRequest(question="深圳地铁1号线有哪些站点", top_k=2))

    assert response.answer == (
        "深圳地铁1号线的站名包括：罗湖、国贸、老街、大剧院、科学馆、华强路、后瑞、机场东。[资料 1]"
    )
    assert response.generation["answer_strategy"] == "direct_extract"
    assert response.generation["evidence_grounded"] is True


@pytest.mark.asyncio
async def test_ask_extracts_station_rows_for_short_station_wording() -> None:
    service = SearchService(
        Settings(),
        cast(Any, FakeStructuredRowsIndex()),
        generator=cast(Any, FakeFailingGenerator()),
    )

    response = await service.ask(SearchRequest(question="深圳地铁1号线有哪些站？", top_k=2))

    assert response.answer == "深圳地铁1号线的站名包括：罗湖、国贸、老街、大剧院。[资料 2]"
    assert response.generation["answer_strategy"] == "direct_extract"


@pytest.mark.asyncio
async def test_ask_extracts_capital_answer_without_model_call() -> None:
    service = SearchService(
        Settings(),
        cast(Any, FakeCapitalIndex()),
        generator=cast(Any, FakeFailingGenerator()),
    )

    response = await service.ask(SearchRequest(question="中国的首都是哪个城市？", top_k=1))

    assert response.answer == "中国的首都是北京。[资料 1]"
    assert response.generation["answer_strategy"] == "direct_extract"


@pytest.mark.asyncio
async def test_ask_extracts_bullet_list_with_specific_context() -> None:
    service = SearchService(
        Settings(),
        cast(Any, FakeGreatWallPassIndex()),
        generator=cast(Any, FakeFailingGenerator()),
    )

    response = await service.ask(SearchRequest(question="中国有哪些著名的长城关隘?", top_k=2))

    assert response.answer == "长城的著名关城包括：虎山长城、山海关、嘉峪关、玉门关、萧关、阳关。[资料 2]"
    assert response.generation["answer_strategy"] == "direct_extract"


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


@pytest.mark.asyncio
async def test_list_query_prefers_station_list_over_station_name_issue() -> None:
    station_issue = LexicalResult(
        node_id="station-name-issue",
        document_id="metro",
        text="站名問題：續建工程機場段其中兩個車站曾同時出現兩個站名。",
        metadata={
            "document_id": "metro",
            "title": "深圳地铁1号线",
            "section": "深圳地铁1号线 > 历史 > 續建工程",
            "parent_id": "station-name-issue",
            "content_type": "list",
            "chunk_order": 0,
        },
        score=0.98,
    )
    station_list = LexicalResult(
        node_id="station-list-summary",
        document_id="metro",
        text="车站列表：罗湖、国贸、老街、大剧院、科学馆、华强路。",
        metadata={
            "document_id": "metro",
            "title": "深圳地铁1号线",
            "section": "深圳地铁1号线 > 车站列表",
            "parent_id": "station-list",
            "content_type": "table_summary",
            "chunk_order": 0,
            "keywords": ["表格", "列表", "全部"],
        },
        score=0.82,
    )
    station_siblings = [
        LexicalResult(
            node_id="station-list-expanded",
            document_id="metro",
            text="车站列表：罗湖、国贸、老街、大剧院、科学馆、华强路。",
            metadata={
                "document_id": "metro",
                "parent_id": "station-list",
                "content_type": "table_summary",
                "chunk_order": 0,
            },
            score=0.82,
        )
    ]

    class FakeStructureIndex:
        def document_structure_candidates(
            self,
            question: str,
            *,
            document_id: str,
            content_types: Any,
            knowledge_base_id: str | None,
            limit: int,
        ) -> list[LexicalResult]:
            assert document_id == "metro"
            return [station_list]

        def structure_chunks(self, parent_id: str, **kwargs: Any) -> list[LexicalResult]:
            assert parent_id == "station-list"
            return station_siblings

    service = SearchService(Settings(), cast(Any, FakeStructureIndex()))
    expanded, did_expand = await service._expand_structured_results(
        "深圳地铁1号线有哪些站点",
        [result("metro", 1.0), station_issue],
        knowledge_base_id=None,
        top_k=3,
    )

    assert did_expand is True
    assert expanded[0].node_id == "station-list-expanded"
