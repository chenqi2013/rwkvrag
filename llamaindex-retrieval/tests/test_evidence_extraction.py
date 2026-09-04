import json
from dataclasses import replace

import httpx
import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.evidence_extraction import LanguageModelEvidenceExtractor
from llamaindex_retrieval.query_planning import TaskField, build_query_plan
from llamaindex_retrieval.schemas import SourceItem


def stream_response(content: str) -> httpx.Response:
    event = json.dumps(
        {"choices": [{"delta": {"content": content}}]},
        ensure_ascii=False,
    )
    return httpx.Response(
        200,
        text=f"data: {event}\n\ndata: [DONE]\n\n",
        headers={"content-type": "text/event-stream"},
    )


def source(snippet: str) -> SourceItem:
    return SourceItem(
        id="unitree",
        document_id="unitree",
        source="finewiki-zh",
        title="宇树科技",
        score=1.0,
        snippet=snippet,
    )


def test_subject_match_rejects_longer_unrelated_name_prefix() -> None:
    plan = build_query_plan("马斯克创办了哪几家公司？")
    elon = SourceItem(
        id="elon",
        document_id="elon",
        source="finewiki-zh",
        title="埃隆·马斯克",
        score=1.0,
        snippet="马斯克是SpaceX创始人。",
    )
    place = SourceItem(
        id="muskadine",
        document_id="muskadine",
        source="finewiki-zh",
        title="马斯克丁 (阿拉巴马州)",
        score=0.9,
        snippet="马斯克丁是美国的一处非建制地区。",
    )

    assert LanguageModelEvidenceExtractor._source_contains_subject(plan, elon) is True
    assert LanguageModelEvidenceExtractor._source_contains_subject(plan, place) is False


def test_comparison_accepts_each_subject_page_independently() -> None:
    plan = build_query_plan("尺八和长笛有什么区别？")
    shakuhachi = SourceItem(
        id="shakuhachi",
        document_id="shakuhachi",
        source="finewiki-zh",
        title="尺八",
        score=1.0,
        snippet="尺八是竖吹乐器，竹制，音色苍凉。",
    )
    flute = SourceItem(
        id="flute",
        document_id="flute",
        source="finewiki-zh",
        title="长笛",
        score=1.0,
        snippet="长笛是横吹乐器，现代多使用金属材质。",
    )

    assert plan.analysis.intent == "comparison"
    assert LanguageModelEvidenceExtractor._source_contains_subject(plan, shakuhachi)
    assert LanguageModelEvidenceExtractor._source_contains_subject(plan, flute)


def test_single_fact_rejects_title_only_unrelated_chunk() -> None:
    plan = replace(
        build_query_plan("西游记是谁写的？"),
        subject="西游记",
        relations=("作者", "作者姓名"),
        fields=(TaskField("f1", "西游记的作者是谁？", ("作者", "作者姓名")),),
    )
    unrelated = SourceItem(
        id="journey-west-theory",
        document_id="journey-west",
        source="finewiki-zh",
        title="西游记",
        score=1.0,
        snippet="脱冕説",
    )
    direct = SourceItem(
        id="journey-west-lead",
        document_id="journey-west",
        source="finewiki-zh",
        title="西游记",
        score=1.0,
        snippet="成书于16世纪明朝中叶，一般认为作者是明朝的吴承恩。",
    )

    assert LanguageModelEvidenceExtractor._source_contains_subject(plan, unrelated) is False
    assert LanguageModelEvidenceExtractor._source_contains_subject(plan, direct) is True


def test_semantic_parser_accepts_compact_sentence_selection() -> None:
    evidence = SourceItem(
        id="capital",
        document_id="capital",
        source="finewiki-zh",
        title="中国首都",
        score=1.0,
        snippet=(
            "清朝入主中原后将北京定为国都。\n"
            "现时北京自1949年後定为中华人民共和国首都。"
        ),
    )
    plan = replace(
        build_query_plan("中国的首都是哪个城市？"),
        subject="中国",
        relations=("首都", "城市"),
        fields=(TaskField("f1", "中国的首都是哪个城市？", ("首都", "城市")),),
    )
    units = (
        "清朝入主中原后将北京定为国都。",
        "现时北京自1949年後定为中华人民共和国首都。",
    )

    candidates = LanguageModelEvidenceExtractor._parse(
        ">\n[s2]",
        plan,
        0,
        evidence,
        sentence_units=units,
        semantic_mode=True,
    )

    assert [candidate.span for candidate in candidates] == [units[1]]


def test_compact_sentence_selection_rejects_explanatory_prose() -> None:
    with pytest.raises(ValueError, match="supported output contract"):
        LanguageModelEvidenceExtractor._parse_compact_candidates(
            "我认为应该选择 s2",
            {"f1"},
        )


def test_compact_sentence_selection_accepts_shared_field_prefix() -> None:
    assert LanguageModelEvidenceExtractor._parse_compact_candidates(
        "f1: [s1, s2, s3]\n无需解释",
        {"f1"},
    ) == [
        {"field_id": "f1", "sentence_id": "s1"},
        {"field_id": "f1", "sentence_id": "s2"},
        {"field_id": "f1", "sentence_id": "s3"},
    ]


def test_adjudication_parser_accepts_compact_field_evidence_pairs() -> None:
    assert LanguageModelEvidenceExtractor._parse_adjudication(
        ">\nf1:e3,f1:e7\n不参与协议的解释",
        {"f1"},
        8,
    ) == (("f1", 3), ("f1", 7))

    assert LanguageModelEvidenceExtractor._parse_adjudication(
        "f1:e3,e7,e8",
        {"f1"},
        8,
    ) == (("f1", 3), ("f1", 7), ("f1", 8))

    assert LanguageModelEvidenceExtractor._parse_adjudication(
        "e2,e1",
        {"f1"},
        2,
    ) == (("f1", 2), ("f1", 1))


def test_adjudication_parser_ignores_explanatory_prose() -> None:
    assert LanguageModelEvidenceExtractor._parse_adjudication(
        "应该选择 f1:e2",
        {"f1"},
        3,
    ) == (("f1", 2),)


@pytest.mark.asyncio
async def test_semantic_extractor_adjudicates_candidates_across_chunks() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        prompt = json.loads(request.content)["contents"][0]
        if "你是证据裁决器" in prompt:
            assert "政治腐败导致国力衰退" in prompt
            evidence_id = next(
                line.split("]", 1)[0].removeprefix("[")
                for line in prompt.splitlines()
                if "政治腐败导致国力衰退" in line
            )
            return stream_response(f"f1:{evidence_id}")
        return stream_response("f1:s1")

    settings = Settings(
        generation_password="secret",
        semantic_pipeline_enabled=True,
        evidence_extraction_concurrency=2,
    )
    extractor = LanguageModelEvidenceExtractor(
        settings,
        transport=httpx.MockTransport(handler),
    )
    plan = replace(
        build_query_plan("明朝灭亡的原因是什么？"),
        subject="明朝",
        relations=("原因", "灭亡"),
        fields=(TaskField("f1", "明朝灭亡的原因是什么？", ("原因", "灭亡")),),
        answer_shape="summary",
    )
    sources = [
        SourceItem(
            id="early",
            document_id="ming",
            source="finewiki-zh",
            title="明朝",
            score=1.0,
            snippet="靖难之役后朱棣即位。",
        ),
        SourceItem(
            id="late",
            document_id="ming",
            source="finewiki-zh",
            title="明朝",
            score=1.0,
            snippet="政治腐败导致国力衰退，最终爆发大规模民变，明朝灭亡。",
        ),
    ]

    result = await extractor.extract(plan.original_question, plan, sources)

    assert calls == 3
    assert result.strategy == "model_map_reduce"
    assert [candidate.span for candidate in result.candidates] == [sources[1].snippet]


def test_adjudication_prioritizes_causal_relation_over_event_only_sentence() -> None:
    extractor = LanguageModelEvidenceExtractor(Settings(semantic_pipeline_enabled=True))
    plan = replace(
        build_query_plan("某王朝灭亡的原因是什么？"),
        subject="某王朝",
        relations=("原因", "灭亡"),
        fields=(TaskField("f1", "某王朝灭亡的原因是什么？", ("原因", "灭亡")),),
        answer_shape="summary",
    )
    source_item = SourceItem(
        id="dynasty",
        document_id="dynasty",
        source="wiki",
        title="某王朝",
        score=1.0,
        snippet="某王朝灭亡。政治腐败导致国力衰退，最终爆发民变。",
    )
    units = extractor._adjudication_units(plan, [source_item], [])

    assert "导致国力衰退" in units[0].span


@pytest.mark.asyncio
async def test_extractor_keeps_only_verbatim_spans() -> None:
    snippet = "2016年，宇树科技创始人王兴兴开发了XDog，随后创办宇树科技。"

    async def handler(_: httpx.Request) -> httpx.Response:
        return stream_response(json.dumps({
            "candidates": [
                {"field_id": "f1", "sentence_id": "s1"},
                {"field_id": "f1", "sentence_id": "s99"},
            ]
        }, ensure_ascii=False))

    settings = Settings(
        generation_password="secret",
        generation_base_url="https://generation.example/v1",
    )
    extractor = LanguageModelEvidenceExtractor(
        settings,
        transport=httpx.MockTransport(handler),
    )
    plan = replace(
        build_query_plan("宇树科技创始人是谁？"),
        answer_shape="summary",
    )
    result = await extractor.extract(
        "宇树科技创始人是谁？",
        plan,
        [source(snippet)],
    )

    assert result.available is True
    assert [candidate.span for candidate in result.candidates] == [
        snippet,
    ]
    assert result.answer_sources([source(snippet)])[0].snippet == (
        snippet
    )
    assert len(result.candidates[0].content_hash) == 64
    unrelated = SourceItem(
        id="other",
        document_id="other",
        source="finewiki-zh",
        title="其他资料",
        score=0.9,
        snippet="这是一条无关资料。",
    )
    remapped = result.remap_sources([unrelated, source(snippet)])
    assert remapped is not None
    assert {candidate.source_index for candidate in remapped.candidates} == {1}
    assert remapped.strategy == "model_remapped"


@pytest.mark.asyncio
async def test_extractor_distinguishes_no_candidate_from_transport_failure() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return stream_response('{"candidates":[]}')

    settings = Settings(
        generation_password="secret",
        generation_base_url="https://generation.example/v1",
    )
    extractor = LanguageModelEvidenceExtractor(
        settings,
        transport=httpx.MockTransport(handler),
    )
    result = await extractor.extract(
        "阿尔法泽是谁？",
        build_query_plan("阿尔法泽是谁？"),
        [source("阿尔法岛是南极洲的一座岛屿。")],
    )

    assert result.available is True
    assert result.has_candidates is False
    assert result.errors == ()


@pytest.mark.asyncio
async def test_extractor_rejects_verbatim_value_from_unrelated_source() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return stream_response(json.dumps({
            "candidates": [{"field_id": "f1", "span": "北京"}]
        }, ensure_ascii=False))

    settings = Settings(
        generation_password="secret",
        generation_base_url="https://generation.example/v1",
    )
    extractor = LanguageModelEvidenceExtractor(
        settings,
        transport=httpx.MockTransport(handler),
    )
    unrelated = SourceItem(
        id="train",
        document_id="train",
        source="finewiki-zh",
        title="京泰高速动车组列车",
        score=1.0,
        snippet="该列车往返北京至泰州。",
    )
    result = await extractor.extract(
        "中国的首都在哪里",
        build_query_plan("中国的首都在哪里"),
        [unrelated],
    )

    assert result.available is True
    assert result.has_candidates is False


def test_narrative_event_extractor_rejects_value_without_relation_span() -> None:
    plan = build_query_plan("水浒传里赤手空拳打死老虎的是谁？")
    evidence = SourceItem(
        id="water-margin",
        document_id="water-margin",
        source="finewiki-zh",
        title="水浒传",
        score=1.0,
        snippet="宋江代表的动物是老虎；武松在景阳冈打死老虎。",
    )
    value_only = json.dumps({
        "candidates": [{"field_id": "f1", "span": "宋江"}]
    }, ensure_ascii=False)
    direct_fact = json.dumps({
        "candidates": [{"field_id": "f1", "span": "武松在景阳冈打死老虎。"}]
    }, ensure_ascii=False)

    assert LanguageModelEvidenceExtractor._parse(value_only, plan, 0, evidence) == ()
    assert LanguageModelEvidenceExtractor._parse(direct_fact, plan, 0, evidence)[0].span == (
        "武松在景阳冈打死老虎。"
    )


def test_sentence_selection_does_not_require_literal_relation_alias() -> None:
    sentence = "成书于16世纪明朝中叶，一般认为作者是明朝的吴承恩。"
    evidence = SourceItem(
        id="journey-west",
        document_id="journey-west",
        source="finewiki-zh",
        title="西游记",
        score=1.0,
        snippet=sentence,
    )
    plan = replace(
        build_query_plan("西游记是哪个作者写的？"),
        subject="西游记",
        relations=("作者", "作者姓名"),
        fields=(TaskField("f1", "西游记的作者是谁？", ("作者", "作者姓名")),),
    )
    raw = json.dumps({
        "candidates": [{"field_id": "f1", "sentence_id": "s1"}]
    }, ensure_ascii=False)

    result = LanguageModelEvidenceExtractor._parse(
        raw,
        plan,
        0,
        evidence,
        sentence_units=(sentence,),
    )

    assert result[0].span == sentence


def test_extractor_accepts_metadata_fields_alongside_candidates() -> None:
    sentence = "宇树科技创始人王兴兴在2016年创办了宇树科技。"
    evidence = SourceItem(
        id="unitree",
        document_id="unitree",
        source="finewiki-zh",
        title="宇树科技",
        score=1.0,
        snippet=sentence,
    )
    plan = replace(
        build_query_plan("宇树科技老板是谁？"),
        subject="宇树科技",
        relations=("老板",),
        fields=(TaskField("f1", "宇树科技的老板是谁？", ("老板",)),),
    )
    raw = json.dumps({
        "candidates": [{"field_id": "f1", "sentence_id": "s1"}],
        "confidence": 0.98,
    }, ensure_ascii=False)

    result = LanguageModelEvidenceExtractor._parse(
        raw,
        plan,
        0,
        evidence,
        sentence_units=(sentence,),
    )

    assert result[0].span == sentence


def test_sentence_selection_rejects_relationless_heading() -> None:
    snippet = "作者认为传统宗教经书已成为束缚。\n脱冕説"
    evidence = SourceItem(
        id="journey-west-theory",
        document_id="journey-west",
        source="finewiki-zh",
        title="西游记",
        score=1.0,
        snippet=snippet,
    )
    plan = replace(
        build_query_plan("西游记是谁写的？"),
        subject="西游记",
        relations=("作者", "作者姓名"),
        fields=(TaskField("f1", "西游记的作者是谁？", ("作者", "作者姓名")),),
    )
    raw = json.dumps({
        "candidates": [{"field_id": "f1", "sentence_id": "s2"}]
    }, ensure_ascii=False)

    assert LanguageModelEvidenceExtractor._parse(
        raw,
        plan,
        0,
        evidence,
        sentence_units=("作者认为传统宗教经书已成为束缚。", "脱冕説"),
    ) == ()


def test_summary_selection_rejects_unrelated_subject_mention() -> None:
    evidence = SourceItem(
        id="author",
        document_id="author",
        source="finewiki-zh",
        title="罗贯中",
        score=1.0,
        snippet="罗贯中是《三国演义》的编作者。",
    )
    plan = replace(
        build_query_plan("三国演义最后的结局是什么？"),
        subject="三国演义",
        relations=("最终结局", "结局"),
        fields=(TaskField("f1", "最后的结局", ("最终结局", "结局")),),
        answer_shape="summary",
    )
    raw = json.dumps({
        "candidates": [{"field_id": "f1", "sentence_id": "s1"}],
    }, ensure_ascii=False)

    assert LanguageModelEvidenceExtractor._parse(
        raw,
        plan,
        0,
        evidence,
        sentence_units=("罗贯中是《三国演义》的编作者。",),
    ) == ()


def test_list_evidence_falls_back_to_direct_relation_sentence() -> None:
    sentence = "四大名著，即四大小说名著，是指《三国演义》《西游记》《水浒传》《红楼梦》4部小说。"
    evidence = SourceItem(
        id="four-classics",
        document_id="four-classics",
        source="finewiki-zh",
        title="四大名著",
        score=1.0,
        snippet=sentence,
    )
    plan = replace(
        build_query_plan("中国四大名著是哪几个？"),
        subject="中国四大名著",
        relations=("是指", "包括", "分别是", "分别为"),
        fields=(TaskField("f1", "中国四大名著是哪几个？", ("是指", "包括")),),
        answer_shape="list",
        set_semantics="all",
    )

    result = LanguageModelEvidenceExtractor._parse(
        '{"candidates":[]}',
        plan,
        0,
        evidence,
        sentence_units=(sentence,),
    )

    assert [candidate.span for candidate in result] == [sentence]


def test_station_companion_page_is_valid_subject_evidence() -> None:
    plan = build_query_plan("深圳地铁1号线有哪几个站")
    evidence = SourceItem(
        id="station-list",
        document_id="station-list",
        source="finewiki-zh",
        title="深圳地铁车站列表",
        score=1.0,
        snippet="1号线沿途共设30个车站：罗湖站、国贸站、老街站。",
    )

    assert LanguageModelEvidenceExtractor._source_contains_subject(plan, evidence) is True
