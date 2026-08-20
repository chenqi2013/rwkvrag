from dataclasses import replace
from typing import Any, cast

import pytest
from llama_index.core.schema import TextNode

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.generation import EvidenceAnswerGenerator
from llamaindex_retrieval.lexical_index import (
    LexicalIndex,
    LexicalResult,
    _focus_bonus,
    entity_bigram_tokens,
    lexical_tokens,
    normalize_query_text,
    query_tokens,
    title_entity_tokens,
)
from llamaindex_retrieval.schemas import SearchRequest, SourceItem
from llamaindex_retrieval.service import SearchService
from llamaindex_retrieval.query_planning import build_query_plan
from llamaindex_retrieval.semantic_query_planning import QueryPlanningResult


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


class FakeCauseGenerator:
    assess_evidence = staticmethod(EvidenceAnswerGenerator.assess_evidence)

    async def generate(self, question: str, sources: Any) -> str:
        assert len(sources) == 3
        return "明朝灭亡与政治失序、天灾饥荒和农民起义有关。[资料 1][资料 2][资料 3]"

    async def current_model(self) -> str:
        return "test-model"


class FakeRefusingCauseGenerator(FakeCauseGenerator):
    async def generate(self, question: str, sources: Any) -> str:
        return "根据检索到的资料，无法确定。"


class FakeEmptyCauseGenerator(FakeCauseGenerator):
    async def generate(self, question: str, sources: Any) -> str:
        return "根据资料，导致明朝灭亡的原因有：[资料 1]"


class FakeCauseIndex:
    def __init__(self) -> None:
        self.structure_calls = 0

    def search(
        self,
        question: str,
        *,
        candidate_k: int,
        knowledge_base_id: str | None = None,
    ) -> list[LexicalResult]:
        return [
            LexicalResult(
                node_id="ming-fall-2",
                document_id="ming",
                text="李自成攻克北京，崇祯帝自缢，明亡。",
                metadata={
                    "document_id": "ming",
                    "title": "明朝",
                    "source": "finewiki-zh",
                    "section": "明朝 > 历史 > 灭亡",
                    "parent_id": "ming-fall",
                    "structure_size": 3,
                    "chunk_order": 2,
                },
                score=1.0,
            )
        ]

    def structure_chunks(self, parent_id: str, **kwargs: Any) -> list[LexicalResult]:
        self.structure_calls += 1
        assert parent_id == "ming-fall"
        texts = (
            "明末党争不断，崇祯帝多疑躁刻，不善用人。",
            "严寒、干旱、饥荒、蝗灾和鼠疫频繁出现，各地相继爆发农民起义。",
            "明政府镇压失败，李自成攻克北京，崇祯帝自缢，明亡。",
        )
        return [
            LexicalResult(
                node_id=f"ming-fall-{index}",
                document_id="ming",
                text=text,
                metadata={
                    "document_id": "ming",
                    "title": "明朝",
                    "source": "finewiki-zh",
                    "section": "明朝 > 历史 > 灭亡",
                    "parent_id": "ming-fall",
                    "structure_size": 3,
                    "chunk_order": index,
                },
                score=1.0,
            )
            for index, text in enumerate(texts)
        ]


class FakeDocumentRelationIndex:
    def __init__(self) -> None:
        self.relation_calls: list[tuple[str, tuple[str, ...]]] = []

    def search(
        self,
        question: str,
        *,
        candidate_k: int,
        knowledge_base_id: str | None = None,
    ) -> list[LexicalResult]:
        return [
            LexicalResult(
                node_id="silk-lead",
                document_id="silk-road",
                text="班超后来再次打通了荒废已久的丝绸之路。",
                metadata={"title": "丝绸之路", "source": "finewiki-zh"},
                score=1.0,
            )
        ]

    def document_relation_candidates(
        self,
        question: str,
        *,
        document_id: str,
        relations: tuple[str, ...],
        knowledge_base_id: str | None,
        limit: int,
    ) -> list[LexicalResult]:
        self.relation_calls.append((document_id, relations))
        return [
            LexicalResult(
                node_id="silk-zhang-qian",
                document_id=document_id,
                text="前139年，张骞带随从从长安出发，史书称其首次西行为凿空。",
                metadata={
                    "title": "丝绸之路",
                    "source": "finewiki-zh",
                    "section": "历史发展 > 张骞的西行",
                },
                score=1.0,
            )
        ]

    def document_lead_chunk(self, *args: Any, **kwargs: Any) -> None:
        return None


class FakeOrdinalIndex:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(
        self,
        question: str,
        *,
        candidate_k: int,
        knowledge_base_id: str | None = None,
    ) -> list[LexicalResult]:
        self.queries.append(question)
        return [
            LexicalResult(
                node_id="first-emperor",
                document_id="emperor",
                text="在中国历史中，嬴政创建皇帝制度，成为中原第一个皇帝，称始皇帝。",
                metadata={
                    "document_id": "emperor",
                    "title": "皇帝",
                    "source": "finewiki-zh",
                },
                score=1.0,
            )
        ]


class FakeOrdinalGenerator:
    assess_evidence = staticmethod(EvidenceAnswerGenerator.assess_evidence)

    async def generate(self, question: str, sources: Any) -> str:
        assert sources[0].title == "皇帝"
        return "中国历史上第一个皇帝是嬴政，即秦始皇。[资料 1]"

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


class FakeDefinitionIndex:
    def search(self, question: str, *, candidate_k: int, knowledge_base_id: str | None = None) -> list[LexicalResult]:
        return [LexicalResult(node_id="later", document_id="entity", text="Category:示例", metadata={"title": "示例"}, score=1.0)]

    def document_lead_chunk(self, document_id: str, *, knowledge_base_id: str | None, score: float) -> LexicalResult:
        return LexicalResult(node_id="lead", document_id=document_id, text="示例是用于测试的条目。", metadata={"title": "示例"}, score=score)


class FakeComparisonIndex:
    def search(self, question: str, *, candidate_k: int, knowledge_base_id: str | None = None) -> list[LexicalResult]:
        return [
            LexicalResult(
                node_id=f"node-{question}",
                document_id=question,
                text=f"{question}是一种测试对象。",
                metadata={"title": question, "source": "finewiki-zh"},
                score=1.0,
            )
        ]


@pytest.mark.asyncio
async def test_definition_question_uses_document_lead_and_direct_answer() -> None:
    service = SearchService(Settings(), cast(Any, FakeDefinitionIndex()), generator=cast(Any, FakeFailingGenerator()))
    response = await service.ask(SearchRequest(question="示例是什么？", top_k=1))
    assert response.answer == "示例是用于测试的条目。 [资料 1]"
    assert response.generation["answer_strategy"] == "direct_extract"


@pytest.mark.asyncio
async def test_comparison_question_decomposes_and_merges_subject_searches() -> None:
    service = SearchService(Settings(), cast(Any, FakeComparisonIndex()))
    response = await service.search(SearchRequest(question="尺八和长笛有什么区别？", top_k=5))

    assert [item.title for item in response.results] == ["尺八", "长笛"]
    assert response.retrieval["intent"] == "comparison"
    assert response.retrieval["query_decomposition"] == ["尺八", "长笛"]


class FakeCapitalDecisionIndex:
    def search(
        self,
        question: str,
        *,
        candidate_k: int,
        knowledge_base_id: str | None = None,
    ) -> list[LexicalResult]:
        return [
            LexicalResult(
                node_id="capital-history",
                document_id="capital-history",
                text="叶剑英已经预料到北平将有可能成为新中国的国都。",
                metadata={
                    "document_id": "capital-history",
                    "title": "中华人民共和国首都",
                    "source": "finewiki-zh",
                },
                score=1.0,
            ),
            LexicalResult(
                node_id="capital-decision",
                document_id="capital-decision",
                text="文件记载：“中华人民共和国的国都定于北平。自即日起，改名北平为北京。”",
                metadata={
                    "document_id": "capital-decision",
                    "title": "关于中华人民共和国国都、纪年、国歌、国旗的决议",
                    "source": "finewiki-zh",
                },
                score=0.95,
            ),
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


def test_rank_fusion_prefers_an_exact_page_alias() -> None:
    plan = build_query_plan("罗宝线有哪些站点？")
    unrelated = LexicalResult(
        node_id="other-node",
        document_id="other",
        text="其他地铁线路也设有多个车站。",
        metadata={"title": "其他地铁线路"},
        score=1.0,
    )
    aliased = LexicalResult(
        node_id="line-1-node",
        document_id="line-1",
        text="深圳地铁1号线设有罗湖站等车站。",
        metadata={"title": "深圳地铁1号线", "aliases": ["罗宝线"]},
        score=0.8,
    )

    merged = SearchService._merge_rank_fusion(
        ([unrelated, aliased], [unrelated, aliased]),
        plan=plan,
    )

    assert merged[0].document_id == "line-1"


def test_rank_fusion_prefers_route_page_for_endpoint_description() -> None:
    plan = build_query_plan("连接罗湖和机场东的深圳地铁线路有哪些车站？")
    station = LexicalResult(
        node_id="station",
        document_id="station",
        text="罗湖站可以换乘多条线路，交通接驳包括机场站。",
        metadata={"title": "罗湖站 (深圳地铁)"},
        score=1.0,
    )
    route = LexicalResult(
        node_id="route",
        document_id="route",
        text="深圳地铁1号线由罗湖站至机场东站，全线共设30座车站。",
        metadata={"title": "深圳地铁1号线"},
        score=0.8,
    )

    merged = SearchService._merge_rank_fusion(
        ([station, route], [station, route]),
        plan=plan,
    )

    assert merged[0].document_id == "route"


def test_lexical_index_searches_chinese_and_titles() -> None:
    client = FakeOpenSearch()
    index = LexicalIndex(Settings(), client=cast(Any, client))
    results = index.search("中国首都在哪里", candidate_k=5)
    assert results
    assert results[0].document_id == "capital"
    fields = client.search_body["query"]["bool"]["must"][0]["bool"]["should"][0]["multi_match"]["fields"]
    assert fields == [
        "body_tokens",
        "title_tokens^3",
        "alias_tokens^4",
        "tags_tokens^1.5",
        "section_tokens^3",
        "structure_tokens^2",
        "entity_bigram_tokens^0.6",
    ]
    recall_queries = client.search_body["query"]["bool"]["must"][0]["bool"]["should"]
    assert any(query.get("multi_match", {}).get("type") == "phrase" for query in recall_queries)
    assert any("entity_bigram_tokens" in query.get("match", {}) for query in recall_queries)


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


def test_transit_line_numbers_are_normalized() -> None:
    assert lexical_tokens("深圳地铁一号线有哪些站点") == lexical_tokens(
        "深圳地铁1号线有哪些站点"
    )
    assert lexical_tokens("广州地铁二十一号线") == lexical_tokens("广州地铁21号线")
    assert {"车站", "站点", "站名"} <= set(query_tokens("深圳地铁1号线有哪些站"))
    assert "センラ" in query_tokens("センラ指的是什么？")
    assert title_entity_tokens("創新方指的是什么？") == lexical_tokens("創新方")


def test_query_normalization_corrects_common_chinese_typos() -> None:
    assert normalize_query_text("中国有多少个名族") == "中国有多少个民族"
    assert "民族" in query_tokens("中国有多少个名族")
    assert {"关城", "关口"} <= set(query_tokens("中国有哪些著名的长城关隘"))
    assert {"国都", "首都"} <= set(query_tokens("中华人民共和国的国都是哪个城市"))
    assert "知道" not in query_tokens("你知道现在中国的首都是在哪个地方吗？")
    assert "地方" not in title_entity_tokens("你知道现在中国的首都是在哪个地方吗？")
    assert title_entity_tokens("YouTube由哪些人共同创立？") == ["youtube"]
    assert title_entity_tokens("YouTube是由哪几个人创立的？") == ["youtube"]


@pytest.mark.asyncio
async def test_service_searches_with_normalized_question() -> None:
    client = FakeOpenSearch()
    service = SearchService(Settings(), LexicalIndex(Settings(), client=cast(Any, client)))

    response = await service.search(SearchRequest(question="中国有多少个名族", top_k=1))

    query = client.search_bodies[0]["query"]["bool"]["must"][0]["bool"]["should"][0]["multi_match"]["query"]
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


@pytest.mark.asyncio
async def test_cause_question_expands_sibling_chunks_from_same_section() -> None:
    index = FakeCauseIndex()
    service = SearchService(
        Settings(max_chunks_per_document=1),
        cast(Any, index),
        generator=cast(Any, FakeCauseGenerator()),
    )

    response = await service.ask(
        SearchRequest(question="明朝是因为什么原因走上了灭亡", top_k=1)
    )

    assert "政治失序、天灾饥荒和农民起义" in response.answer
    assert index.structure_calls == 1
    assert response.retrieval["cause_context_expanded"] is True
    assert response.retrieval["multi_evidence"] is True
    assert response.retrieval["answer_evidence_count"] == 3
    assert response.generation["evidence_grounded"] is True
    assert "明朝" in response.generation["evidence_anchors"]
    assert "是因为" not in response.generation["evidence_anchors"]


@pytest.mark.asyncio
async def test_non_cause_question_does_not_expand_cause_context() -> None:
    index = FakeCauseIndex()
    service = SearchService(Settings(), cast(Any, index))

    response = await service.search(SearchRequest(question="明朝灭亡于哪一年", top_k=5))

    assert index.structure_calls == 0
    assert response.retrieval["intent"] == "time"
    assert response.retrieval["cause_context_expanded"] is False


@pytest.mark.asyncio
async def test_cause_question_falls_back_to_grounded_excerpts_when_model_refuses() -> None:
    service = SearchService(
        Settings(max_chunks_per_document=1),
        cast(Any, FakeCauseIndex()),
        generator=cast(Any, FakeRefusingCauseGenerator()),
    )

    response = await service.ask(
        SearchRequest(question="明朝究竟为何一步步走向覆亡了?", top_k=1)
    )

    assert "党争" in response.answer
    assert "无法确定" not in response.answer
    assert response.generation["answer_strategy"] == "evidence_fallback"


@pytest.mark.asyncio
async def test_cause_question_repairs_empty_model_answer_before_caching() -> None:
    service = SearchService(
        Settings(max_chunks_per_document=1),
        cast(Any, FakeCauseIndex()),
        generator=cast(Any, FakeEmptyCauseGenerator()),
    )

    first = await service.ask(SearchRequest(question="明朝灭亡的原因有哪些？", top_k=1))
    second = await service.ask(SearchRequest(question="明朝灭亡的原因有哪些？", top_k=1))

    assert "党争" in first.answer
    assert first.generation["answer_strategy"] == "evidence_fallback"
    assert second.answer == first.answer
    assert second.generation["cache_hit"] is True


@pytest.mark.asyncio
async def test_cause_question_repairs_legacy_empty_cached_answer() -> None:
    service = SearchService(
        Settings(max_chunks_per_document=1),
        cast(Any, FakeCauseIndex()),
        generator=cast(Any, FakeFailingGenerator()),
    )
    request = SearchRequest(question="明朝灭亡的原因有哪些？", top_k=1)
    evidence = await service.search(request.model_copy(update={"top_k": 5}))
    question = str(evidence.retrieval.get("normalized_question") or request.question)
    key = service._answer_cache_key(question, evidence)
    service._store_cached_answer(key, "根据资料，导致明朝灭亡的原因有：[资料 1]")

    response = await service.ask(request)

    assert "党争" in response.answer
    assert response.generation["cache_hit"] is True
    assert response.generation["answer_strategy"] == "evidence_fallback"
    assert service._get_cached_answer(key) == response.answer


@pytest.mark.asyncio
async def test_relation_question_searches_inside_exact_subject_document() -> None:
    index = FakeDocumentRelationIndex()
    service = SearchService(
        Settings(),
        cast(Any, index),
        generator=cast(Any, FakeFailingGenerator()),
    )

    response = await service.ask(
        SearchRequest(question="中国历史上开启丝绸之路的是谁？", top_k=2)
    )

    assert "张骞" in response.answer
    assert response.sources[0].id == "silk-zhang-qian"
    assert response.retrieval["document_relation_expanded"] is True
    assert index.relation_calls
    assert {"开启", "开辟", "出使", "凿空"} <= set(index.relation_calls[0][1])


@pytest.mark.asyncio
async def test_ordinal_question_searches_equivalent_first_relation_phrases() -> None:
    index = FakeOrdinalIndex()
    service = SearchService(
        Settings(),
        cast(Any, index),
        generator=cast(Any, FakeOrdinalGenerator()),
    )

    response = await service.ask(
        SearchRequest(question="中国历史上第一个皇帝是谁？", top_k=1)
    )

    assert response.answer == (
        "在中国历史中，嬴政创建皇帝制度，成为中原第一个皇帝，称始皇帝。 [资料 1]"
    )
    assert response.generation["answer_strategy"] == "direct_extract"
    assert response.retrieval["intent"] == "ordinal"
    assert set(index.queries) == {
        "中国 第一个 皇帝",
        "中国 第一位 皇帝",
        "中国 首位 皇帝",
        "中国历史上第一个皇帝是谁？",
    }


@pytest.mark.asyncio
async def test_ungrounded_definition_does_not_extract_unrelated_heading() -> None:
    service = SearchService(
        Settings(),
        cast(Any, FakeMismatchIndex()),
        generator=cast(Any, FakeInsufficientGenerator()),
    )

    response = await service.ask(SearchRequest(question="秦始皇是谁？", top_k=1))

    assert response.answer == "根据检索到的资料，无法确定。"
    assert response.generation["answer_strategy"] == "model"
    assert response.generation["evidence_grounded"] is False


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
    requested_limit = service._max_chunks_per_document(
        "中国从古至今总共经历了哪些朝代？", top_k=10
    )
    assert requested_limit == 10
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
async def test_ask_extracts_capital_answer_for_colloquial_wording() -> None:
    service = SearchService(
        Settings(),
        cast(Any, FakeCapitalIndex()),
        generator=cast(Any, FakeFailingGenerator()),
    )

    response = await service.ask(SearchRequest(question="你知道现在中国的首都是在哪个地方吗？", top_k=1))

    assert response.answer == "中国的首都是北京。[资料 1]"
    assert response.generation["answer_strategy"] == "direct_extract"


@pytest.mark.asyncio
async def test_ask_extracts_capital_answer_for_national_capital_wording() -> None:
    service = SearchService(
        Settings(),
        cast(Any, FakeCapitalIndex()),
        generator=cast(Any, FakeFailingGenerator()),
    )

    response = await service.ask(SearchRequest(question="中华人民共和国的国都是哪个城市？", top_k=1))

    assert response.answer == "中华人民共和国的首都是北京。[资料 1]"
    assert response.generation["answer_strategy"] == "direct_extract"


@pytest.mark.asyncio
async def test_ask_extracts_capital_decision_and_ignores_possible_future_capital() -> None:
    service = SearchService(
        Settings(),
        cast(Any, FakeCapitalDecisionIndex()),
        generator=cast(Any, FakeFailingGenerator()),
    )

    response = await service.ask(SearchRequest(question="中华人民共和国的国都是哪个城市？", top_k=2))

    assert response.answer == "中华人民共和国的首都是北京。[资料 2]"
    assert response.generation["answer_strategy"] == "direct_extract"


@pytest.mark.asyncio
async def test_ask_uses_more_internal_evidence_than_displayed_sources() -> None:
    service = SearchService(
        Settings(),
        cast(Any, FakeCapitalDecisionIndex()),
        generator=cast(Any, FakeFailingGenerator()),
    )

    response = await service.ask(SearchRequest(question="中华人民共和国的国都是哪个城市？", top_k=1))

    assert response.answer == "中华人民共和国的首都是北京。[资料 2]"
    assert len(response.sources) == 2
    assert [source.id for source in response.sources] == ["capital-history", "capital-decision"]
    assert response.generation["evidence_count"] == 2
    assert response.generation["displayed_evidence_count"] == 2
    assert response.retrieval["answer_evidence_count"] == 2


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
async def test_ask_extracts_bullet_list_for_list_instruction_wording() -> None:
    service = SearchService(
        Settings(),
        cast(Any, FakeGreatWallPassIndex()),
        generator=cast(Any, FakeFailingGenerator()),
    )

    response = await service.ask(SearchRequest(question="列一下长城的著名关城", top_k=2))

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
async def test_search_uses_normalized_question_for_structured_list_expansion() -> None:
    list_chunk = LexicalResult(
        node_id="dialog-list",
        document_id="flight",
        text="事故 > 与空管的对话\n- 飞行员：失去所有引擎。\n- 空管：请重复。",
        metadata={
            "document_id": "flight",
            "title": "测试班机事故",
            "section": "测试班机事故 > 与空管的对话",
            "parent_id": "dialog-parent",
            "content_type": "list",
            "chunk_order": 0,
        },
        score=1.0,
    )

    class FakeNormalizedListIndex:
        def search(self, question: str, **kwargs: Any) -> list[LexicalResult]:
            return [list_chunk]

        def structure_chunks(self, parent_id: str, **kwargs: Any) -> list[LexicalResult]:
            assert parent_id == "dialog-parent"
            return [list_chunk]

    service = SearchService(Settings(), cast(Any, FakeNormalizedListIndex()))
    response = await service.search(
        SearchRequest(question="测试班机事故在与空管的对话方面都包括什么？", top_k=5)
    )

    assert response.retrieval["normalized_question"] == "测试班机事故有哪些与空管的对话？"
    assert response.retrieval["structure_expanded"] is True
    assert response.retrieval["max_chunks_per_document"] >= 5


@pytest.mark.asyncio
async def test_model_list_relations_do_not_override_structured_evidence() -> None:
    prose = LexicalResult(
        node_id="line-prose",
        document_id="line",
        text="线路建设期间曾调整部分车站名称。",
        metadata={
            "document_id": "line",
            "title": "某线路",
            "content_type": "prose",
        },
        score=1.0,
    )
    structure_anchor = LexicalResult(
        node_id="station-summary",
        document_id="line",
        text="车站列表：甲站、乙站、丙站。",
        metadata={
            "document_id": "line",
            "title": "某线路",
            "section": "某线路 > 车站列表",
            "parent_id": "station-table",
            "content_type": "table_summary",
            "chunk_order": 0,
        },
        score=0.9,
    )

    class FakeListIndex:
        relation_calls = 0

        def search(self, question: str, **kwargs: Any) -> list[LexicalResult]:
            return [prose]

        def document_relation_candidates(self, *args: Any, **kwargs: Any) -> list[LexicalResult]:
            self.relation_calls += 1
            return [prose]

        def document_structure_candidates(self, *args: Any, **kwargs: Any) -> list[LexicalResult]:
            return [structure_anchor]

        def structure_chunks(self, parent_id: str, **kwargs: Any) -> list[LexicalResult]:
            assert parent_id == "station-table"
            return [structure_anchor]

        def document_lead_chunk(self, *args: Any, **kwargs: Any) -> None:
            return None

    class FakeModelPlanner:
        async def plan(self, question: str, fallback: Any) -> QueryPlanningResult:
            return QueryPlanningResult(
                replace(
                    fallback,
                    queries=("某线路全部车站", "某线路站点列表", question),
                    subject="某线路",
                    relations=("所有站点",),
                ),
                "model",
                model_queries=("某线路全部车站", "某线路站点列表"),
            )

    index = FakeListIndex()
    service = SearchService(
        Settings(),
        cast(Any, index),
        query_planner=cast(Any, FakeModelPlanner()),
    )

    response = await service.search(SearchRequest(question="某线路所有站点", top_k=1))

    assert response.results[0].id == "station-summary"
    assert response.retrieval["structure_expanded"] is True
    assert response.retrieval["document_relation_expanded"] is False
    assert index.relation_calls == 0


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


@pytest.mark.asyncio
async def test_station_summary_without_parent_adds_repair_context() -> None:
    summary = LexicalResult(
        node_id="station-summary",
        document_id="metro",
        text="站名/1号线列表：罗湖、固戍、瑞、机场东。",
        metadata={
            "document_id": "metro",
            "title": "深圳地铁1号线",
            "content_type": "table_summary",
        },
        score=1.0,
    )
    context = LexicalResult(
        node_id="station-context",
        document_id="metro",
        text="机场东站和后瑞站为高架车站。",
        metadata={"document_id": "metro", "title": "深圳地铁1号线"},
        score=1.0,
    )

    class FakeStationRepairIndex:
        def document_term_candidates(self, term: str, **kwargs: Any) -> list[LexicalResult]:
            assert term == "瑞"
            return [summary, context]

    service = SearchService(Settings(), cast(Any, FakeStationRepairIndex()))
    expanded, did_expand = await service._expand_structured_results(
        "深圳地铁1号线有哪些站点",
        [summary],
        knowledge_base_id=None,
        top_k=5,
    )

    assert did_expand is True
    assert [item.node_id for item in expanded] == ["station-summary", "station-context"]


def test_cause_evidence_trims_content_after_target_event() -> None:
    sources = [
        SourceItem(
            id="ming-1",
            document_id="ming",
            source="finewiki-zh",
            title="明朝",
            score=1.0,
            snippet="李自成攻克北京，崇祯帝自缢，明亡。明朝灭亡后，南明内部分裂。",
        )
    ]

    trimmed = SearchService._trim_post_event_evidence(
        sources,
        subject="明朝",
        relations=("灭亡", "原因", "导致"),
    )

    assert trimmed[0].snippet == "李自成攻克北京，崇祯帝自缢，明亡"


def test_focus_sources_keeps_only_exact_subject_document() -> None:
    plan = build_query_plan("2030年世界杯由哪些国家主办？")
    sources = [
        SourceItem(
            id="football",
            document_id="football",
            source="wiki",
            title="2030年国际足协世界杯",
            score=1.0,
            snippet="赛事由多个国家共同主办。",
        ),
        SourceItem(
            id="cricket",
            document_id="cricket",
            source="wiki",
            title="男子T20世界杯",
            score=0.9,
            snippet="2030年赛事由多个国家共同主办。",
        ),
    ]

    focused = SearchService._focus_sources_on_subject(plan, sources)

    assert [source.document_id for source in focused] == ["football"]


def test_focus_sources_keeps_only_agent_relation_document() -> None:
    plan = build_query_plan("安全电梯由谁发明？")
    sources = [
        SourceItem(
            id="elevator",
            document_id="elevator",
            source="wiki",
            title="电梯",
            score=1.0,
            snippet="安全电梯使用的安全钳由奥的斯发明。",
        ),
        SourceItem(
            id="invention",
            document_id="invention",
            source="wiki",
            title="发明",
            score=0.8,
            snippet="发明是创造新事物的过程。",
        ),
    ]

    focused = SearchService._focus_sources_on_subject(plan, sources)

    assert [source.document_id for source in focused] == ["elevator"]


def test_focus_sources_keeps_alias_route_and_matching_list_chunks() -> None:
    plan = build_query_plan("罗宝线沿途停靠哪些站？")
    sources = [
        SourceItem(
            id="line-1",
            document_id="line-1",
            source="wiki",
            title="深圳地铁1号线",
            score=1.0,
            snippet="深圳地铁1号线曾称罗宝线，由罗湖站至机场东站。",
            metadata={"aliases": ["罗宝线"], "chunk_order": 0},
        ),
        SourceItem(
            id="stations-1",
            document_id="station-list",
            source="wiki",
            title="深圳地铁车站列表",
            score=0.9,
            snippet="1号线前称罗宝线，沿途共设30个车站。罗湖站、国贸站。",
            metadata={"chunk_order": 1},
        ),
        SourceItem(
            id="stations-2",
            document_id="station-list",
            source="wiki",
            title="深圳地铁车站列表",
            score=0.9,
            snippet="香蜜湖站、车公庙站、机场东站。",
            metadata={"chunk_order": 2},
        ),
        SourceItem(
            id="line-2-stations",
            document_id="station-list",
            source="wiki",
            title="深圳地铁车站列表",
            score=0.8,
            snippet="2号线前称蛇口线，沿途共设32个车站。",
            metadata={"chunk_order": 3},
        ),
    ]

    focused = SearchService._focus_sources_on_subject(plan, sources)

    assert [source.id for source in focused] == ["line-1", "stations-1", "stations-2"]


def test_focus_sources_uses_route_endpoints_instead_of_first_station() -> None:
    plan = build_query_plan("连接罗湖和机场东的深圳地铁线路有哪些车站？")
    sources = [
        SourceItem(
            id="route",
            document_id="line-1",
            source="wiki",
            title="深圳地铁1号线",
            score=1.0,
            snippet="深圳地铁1号线由罗湖站至机场东站，全线共设30座车站。",
            metadata={"aliases": ["罗宝线"], "chunk_order": 0},
        ),
        SourceItem(
            id="station",
            document_id="luohu",
            source="wiki",
            title="罗湖站 (深圳地铁)",
            score=0.9,
            snippet="罗湖站位于罗湖口岸附近。",
        ),
    ]

    focused = SearchService._focus_sources_on_subject(plan, sources)

    assert [source.id for source in focused] == ["route"]
