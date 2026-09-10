import json
from dataclasses import replace
from hashlib import sha256

import httpx
import pytest


from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.evidence_extraction import LanguageModelEvidenceExtractor
from llamaindex_retrieval.query_planning import TaskField, build_query_plan
from llamaindex_retrieval.schemas import SourceItem


@pytest.mark.parametrize(
    "selection",
    [
        {"field_id": "f1", "span": "不存在的引用"},
        {"field_id": "f2", "span": "甲由乙创立。"},
        {"field_id": "f1", "sentence_id": "s99"},
        {"field_id": "f1", "sentence_id": "s0"},
        {"field_id": "f1", "sentence_id": "s-1"},
    ],
)
def test_parser_rejects_invalid_field_sentence_and_nonverbatim_span(selection) -> None:
    evidence = source("甲由乙创立。")
    parsed = LanguageModelEvidenceExtractor._parse(
        json.dumps({"candidates": [selection]}),
        build_query_plan("甲的创建者是谁？"),
        0,
        evidence,
        sentence_units=(evidence.snippet,),
    )
    assert parsed == ()


def test_parser_preserves_model_selection_without_relation_word_filter() -> None:
    evidence = source("代号乙担任最初的召集者。")
    plan = replace(build_query_plan("谁创立了它？"), subject="甲", relations=("创始人",))
    parsed = LanguageModelEvidenceExtractor._parse(
        '{"candidates":[{"field_id":"f1","sentence_id":"s1"}]}',
        plan,
        0,
        evidence,
        sentence_units=(evidence.snippet,),
    )
    assert parsed[0].span == evidence.snippet
    assert parsed[0].content_hash == sha256(evidence.snippet.encode()).hexdigest()


@pytest.mark.asyncio
async def test_resolver_calls_read_sources_independently_and_select_original_spans() -> None:
    evidence = [
        source("甲片段包含干扰项。"),
        source("乙片段给出了创建者。 ").model_copy(update={"id": "second"}),
    ]
    map_prompts = []

    async def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["contents"][0]
        if "你是证据裁决器" in prompt:
            assert "甲片段包含干扰项" in prompt
            assert "乙片段给出了创建者" in prompt
            return stream_response("f1:e2")
        map_prompts.append(prompt)
        assert ("甲片段包含干扰项" in prompt) != ("乙片段给出了创建者" in prompt)
        return stream_response("f1:s1")

    extractor = LanguageModelEvidenceExtractor(
        Settings(generation_password="secret", semantic_pipeline_enabled=True),
        transport=httpx.MockTransport(handler),
    )
    result = await extractor.extract("谁创建了它？", build_query_plan("谁创建了它？"), evidence)

    assert len(map_prompts) == 2
    assert result.errors == ()
    assert result.completed_sources == 2
    assert result.matches_sources(evidence)
    assert result.candidates[0].source_index == 1
    assert result.candidates[0].span == "乙片段给出了创建者。"
    assert result.answer_sources(evidence)[0].id == "second"
    assert result.answer_sources(evidence)[0].snippet == result.candidates[0].span
    assert [event["stage"] for event in result.trace_events] == [
        "resolver_map",
        "resolver_map",
        "resolver_adjudication",
    ]
    for event in result.trace_events:
        assert event["raw_output_sha256"] == sha256(event["raw_output"].encode()).hexdigest()
    changed = [evidence[0], evidence[1].model_copy(update={"snippet": "已修改正文"})]
    assert not result.matches_sources(changed)
    assert result.remap_sources(changed) is None


@pytest.mark.asyncio
async def test_resolver_empty_decision_does_not_create_fallback_evidence() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return stream_response("NONE")

    extractor = LanguageModelEvidenceExtractor(
        Settings(generation_password="secret", semantic_pipeline_enabled=True),
        transport=httpx.MockTransport(handler),
    )
    result = await extractor.extract(
        "创建者是谁？",
        build_query_plan("创建者是谁？"),
        [source("甲由乙创立。")],
    )
    assert result.available
    assert not result.has_candidates
    assert result.errors == ()
    assert result.answer_sources([source("甲由乙创立。")]) == []


@pytest.mark.parametrize("failure", ["http", "timeout", "invalid_output"])
@pytest.mark.asyncio
async def test_extractor_reports_transport_and_protocol_errors(failure) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if failure == "http":
            return httpx.Response(503)
        if failure == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        return stream_response("invalid output")

    extractor = LanguageModelEvidenceExtractor(
        Settings(generation_password="secret"),
        transport=httpx.MockTransport(handler),
    )
    result = await extractor.extract(
        "创建者是谁？", build_query_plan("创建者是谁？"), [source("正文。")]
    )
    assert result.attempted_sources == 1
    assert result.completed_sources == 0
    assert not result.has_candidates
    assert len(result.errors) == 1
    assert "source_1" in result.errors[0]


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


def test_semantic_parser_accepts_compact_sentence_selection() -> None:
    evidence = SourceItem(
        id="capital",
        document_id="capital",
        source="finewiki-zh",
        title="中国首都",
        score=1.0,
        snippet=("清朝入主中原后将北京定为国都。\n现时北京自1949年後定为中华人民共和国首都。"),
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


@pytest.mark.asyncio
async def test_extractor_keeps_only_verbatim_spans() -> None:
    snippet = "2016年，宇树科技创始人王兴兴开发了XDog，随后创办宇树科技。"

    async def handler(_: httpx.Request) -> httpx.Response:
        return stream_response(
            json.dumps(
                {
                    "candidates": [
                        {"field_id": "f1", "sentence_id": "s1"},
                        {"field_id": "f1", "sentence_id": "s99"},
                    ]
                },
                ensure_ascii=False,
            )
        )

    settings = Settings(
        generation_password="secret",
        generation_base_url="https://generation.example/v1",
        semantic_pipeline_enabled=False,
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
    assert result.answer_sources([source(snippet)])[0].snippet == (snippet)
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
        semantic_pipeline_enabled=False,
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
    raw = json.dumps({"candidates": [{"field_id": "f1", "sentence_id": "s1"}]}, ensure_ascii=False)

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
    raw = json.dumps(
        {
            "candidates": [{"field_id": "f1", "sentence_id": "s1"}],
            "confidence": 0.98,
        },
        ensure_ascii=False,
    )

    result = LanguageModelEvidenceExtractor._parse(
        raw,
        plan,
        0,
        evidence,
        sentence_units=(sentence,),
    )

    assert result[0].span == sentence
