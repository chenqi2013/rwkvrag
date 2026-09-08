import json
from dataclasses import replace
from hashlib import sha256

import httpx
import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.evidence_extraction import (
    EvidenceExtractionResult,
    EvidenceSpan,
    LanguageModelEvidenceExtractor,
)
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


def bound_result(sources: list[SourceItem], index: int, span: str) -> EvidenceExtractionResult:
    return EvidenceExtractionResult(
        candidates=(EvidenceSpan("f1", index, span, sha256(span.encode()).hexdigest()),),
        attempted_sources=len(sources),
        completed_sources=len(sources),
        source_signatures=tuple(
            (item.id, sha256(item.snippet.encode()).hexdigest()) for item in sources
        ),
        source_identities=tuple(
            (item.id, item.document_id, item.source, item.title, item.uri) for item in sources
        ),
    )


def test_selected_local_index_is_resolved_before_attaching_source_identity() -> None:
    original = [
        source(f"Article A paragraph {i}.").model_copy(update={"id": f"a{i}"})
        for i in range(4)
    ]
    selected_b = SourceItem(
        id="b0", document_id="article-b", source="wiki", title="Article B",
        uri="https://example.test/B", score=0.7, snippet="B has the answer. Other B text.",
        metadata={"nested": {"revision": 42}},
    )
    original.append(selected_b)
    selected = original[:3] + original[4:]
    extraction = bound_result(selected, 3, "B has the answer.")
    before = [item.model_dump() for item in original]

    assert not extraction.matches_sources(original)
    rendered = extraction.answer_sources(original)
    assert len(rendered) == 1
    assert rendered[0].id == "b0"
    assert rendered[0].document_id == "article-b"
    assert rendered[0].title == "Article B"
    assert rendered[0].uri == "https://example.test/B"
    assert rendered[0].snippet == "B has the answer."
    assert rendered[0].score == 0.7
    rendered[0].metadata["nested"]["revision"] = 99
    assert [item.model_dump() for item in original] == before

    reordered = [original[4], *original[:4]]
    remapped = extraction.remap_sources(reordered)
    assert remapped is not None
    assert remapped.candidates[0].source_index == 0
    assert remapped.matches_sources(reordered)
    assert remapped.answer_sources(reordered)[0].id == "b0"
    assert extraction.candidates[0].source_index == 3


@pytest.mark.parametrize("changed", [
    {"id": "reused-text-other-node"},
    {"document_id": "other-document"},
    {"title": "Other title"},
    {"uri": "https://example.test/other"},
    {"source": "other-collection"},
    {"snippet": "The quoted answer. Changed surrounding context."},
])
def test_binding_rejects_identity_or_complete_source_changes(changed: dict) -> None:
    original = source("The quoted answer. Original surrounding context.")
    extraction = bound_result([original], 0, "The quoted answer.")
    changed_source = original.model_copy(update=changed)

    assert not extraction.matches_sources([changed_source])
    assert extraction.remap_sources([changed_source]) is None
    assert extraction.answer_sources([changed_source]) == []


@pytest.mark.parametrize("index", [-1, True, 1])
def test_invalid_candidate_indexes_cannot_use_python_negative_indexing(index: int) -> None:
    original = source("Exact evidence.")
    extraction = bound_result([original], 0, original.snippet)
    extraction = replace(extraction, candidates=(replace(extraction.candidates[0], source_index=index),))
    assert extraction.answer_sources([original]) == []
    assert extraction.remap_sources([original]) is None


@pytest.mark.parametrize("span, content_hash", [
    ("Exact evidence.", "0" * 64),
    ("Invented evidence.", sha256(b"Invented evidence.").hexdigest()),
    ("", sha256(b"").hexdigest()),
])
def test_binding_checks_exact_quote_and_quote_hash(span: str, content_hash: str) -> None:
    original = source("Exact evidence.")
    extraction = bound_result([original], 0, original.snippet)
    extraction = replace(
        extraction,
        candidates=(replace(extraction.candidates[0], span=span, content_hash=content_hash),),
    )
    assert not extraction.matches_sources([original])
    assert extraction.answer_sources([original]) == []
    assert extraction.remap_sources([original]) is None


def test_duplicate_source_identity_is_rejected_instead_of_picking_last_match() -> None:
    original = source("Exact evidence.")
    extraction = bound_result([original], 0, original.snippet)
    duplicates = [original, original.model_copy(deep=True)]
    assert not extraction.matches_sources(duplicates)
    assert extraction.answer_sources(duplicates) == []
    assert extraction.remap_sources(duplicates) is None


def test_remapping_keeps_only_valid_surviving_sources() -> None:
    first = source("First evidence.").model_copy(update={"id": "first"})
    second = source("Second evidence.").model_copy(update={"id": "second"})
    extraction = bound_result([first, second], 0, first.snippet)
    extraction = replace(extraction, candidates=(
        *extraction.candidates,
        EvidenceSpan("f2", 1, second.snippet, sha256(second.snippet.encode()).hexdigest()),
    ))
    current = [second, first.model_copy(update={"snippet": "Changed first."})]
    remapped = extraction.remap_sources(current)
    assert remapped is not None
    assert [(item.field_id, item.source_index) for item in remapped.candidates] == [("f2", 0)]
    assert [item.id for item in extraction.answer_sources(current)] == ["second"]
    assert [item.id for item in remapped.answer_sources(current)] == ["second"]


def test_legacy_unbound_result_requires_exact_evidence_at_original_position() -> None:
    original = source("Exact evidence.")
    extraction = EvidenceExtractionResult(
        (EvidenceSpan("f1", 0, original.snippet, sha256(original.snippet.encode()).hexdigest()),),
        1, 1,
    )
    assert extraction.answer_sources([original])[0].snippet == original.snippet
    unrelated = source("Other evidence.").model_copy(update={"id": "other"})
    assert extraction.answer_sources([unrelated, original]) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("shape, limit", [("single_fact", 6), ("summary", 6), ("list", 8)])
async def test_extraction_uses_total_budget_without_per_document_quota(
    shape: str, limit: int,
) -> None:
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return stream_response("f1:s1")

    settings = Settings(
        generation_password="secret",
        generation_base_url="https://generation.example/v1",
        evidence_extraction_max_sources=limit,
    )
    extractor = LanguageModelEvidenceExtractor(settings, transport=httpx.MockTransport(handler))
    plan = replace(
        build_query_plan("Atlas的指标是什么？"), subject="Atlas", relations=("指标",),
        fields=(TaskField("f1", "Atlas的指标是什么？", ("指标",)),), answer_shape=shape,
    )
    sources = [
        SourceItem(
            id=f"atlas-{i}", document_id="atlas", title="Atlas", source="wiki",
            score=1, snippet=f"Atlas 的第{i}项指标为编号{i}。",
        )
        for i in range(limit + 1)
    ]
    original = [item.model_dump() for item in sources]
    result = await extractor.extract(plan.original_question, plan, sources)

    assert len(calls) == limit
    assert result.attempted_sources == result.completed_sources == limit
    assert [item.id for item in result.answer_sources(sources)] == [
        item.id for item in sources[:limit]
    ]
    selection = result.trace_events[0]
    assert selection["stage"] == "resolver_selection"
    assert selection["source_budget"] == limit
    assert selection["input_source_count"] == limit + 1
    assert selection["selected_sources"] == [
        {
            "source_id": item.id, "document_id": item.document_id,
            "original_source_index": i, "selected_source_index": i,
            "source_text_sha256": sha256(item.snippet.encode()).hexdigest(),
        }
        for i, item in enumerate(sources[:limit])
    ]
    for event in result.trace_events[1:]:
        i = event["original_source_index"]
        assert event["source_id"] == sources[i].id
        assert event["source_text_sha256"] == sha256(sources[i].snippet.encode()).hexdigest()
    assert [item.model_dump() for item in sources] == original


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
