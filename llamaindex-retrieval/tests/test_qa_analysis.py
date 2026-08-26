from llamaindex_retrieval.qa_analysis import (
    ambiguity_candidates,
    analyze_question,
    comparison_subjects,
    remove_unsupported_number_sentences,
    validate_grounding,
    validate_list_answer,
)
from llamaindex_retrieval.query_planning import build_query_plan
from llamaindex_retrieval.lexical_index import intent_content_types
from llamaindex_retrieval.schemas import SourceItem


def source(title: str, snippet: str) -> SourceItem:
    return SourceItem(
        id=title,
        document_id=title,
        source="finewiki-zh",
        title=title,
        score=1.0,
        snippet=snippet,
    )


def test_analyzes_comparison_and_list_intents() -> None:
    assert comparison_subjects("CPUID和GPU有什么区别？") == ("CPUID", "GPU")
    comparison = analyze_question("比较尺八和长笛")
    assert comparison.intent == "comparison"
    assert comparison.subjects == ("尺八", "长笛")

    listing = analyze_question("列出深圳地铁1号线所有站点")
    assert listing.intent == "list"
    assert listing.expects_list is True
    assert listing.expects_complete_list is True

    counted_listing = analyze_question("中国四大名著是哪四个？")
    assert counted_listing.intent == "list"
    assert counted_listing.expects_list is True
    assert counted_listing.expects_complete_list is True

    generic_counted_listing = analyze_question("中国四大名著是哪几个？")
    assert generic_counted_listing.intent == "list"
    assert generic_counted_listing.expects_list is True
    assert generic_counted_listing.expects_complete_list is True

    time_question = analyze_question("中国共产党是什么时候成立的")
    assert time_question.intent == "time"
    assert time_question.subjects == ("中国共产党", "中国共产党成立")
    assert analyze_question("香港是哪一年回归的").subjects == ("香港", "香港回归")
    assert analyze_question("赤壁之战是谁发起的").intent == "agent"
    youtube_founders = analyze_question("YouTube是由哪几个人创立的？")
    assert youtube_founders.intent == "agent"
    assert youtube_founders.entity_type == "person"
    assert youtube_founders.expects_list is False
    assert analyze_question("电话是由谁发明的？").intent == "agent"
    assert analyze_question("这座建筑由哪些人共同设计？").intent == "agent"
    assert analyze_question("唐朝的开国皇帝是谁").intent == "agent"
    assert analyze_question("唐朝的开国皇帝是谁").subjects == ("唐朝 开国皇帝",)
    first_emperor = analyze_question("中国历史上第一个皇帝是谁？")
    assert first_emperor.intent == "ordinal"
    assert first_emperor.entity_type == "person"
    assert first_emperor.subjects == (
        "中国 第一个 皇帝",
        "中国 第一位 皇帝",
        "中国 首位 皇帝",
    )
    reverse_first_emperor = analyze_question("谁是中国第一位皇帝？")
    assert reverse_first_emperor.intent == "ordinal"
    assert reverse_first_emperor.subjects == first_emperor.subjects
    bare_first_emperor = analyze_question("中国历史上第一个皇帝")
    assert bare_first_emperor.intent == "ordinal"
    assert bare_first_emperor.subjects == first_emperor.subjects
    assert analyze_question("明朝走向灭亡的主要原因有哪些？").intent == "cause"
    assert intent_content_types("YouTube是由哪几个人创立的？") == ()
    assert intent_content_types("这座建筑由哪些人共同设计？") == ()
    dynasty_list = analyze_question("中国从古至今总共经历了哪些朝代？")
    assert dynasty_list.intent == "list"
    assert dynasty_list.subjects == ("中国 朝代列表", "中国 朝代")
    assert analyze_question("赤壁之战是什么？").intent == "definition"
    assert analyze_question("请简要介绍动画长片列表。").intent == "definition"
    assert analyze_question("某线路的车站列表有哪些？").intent == "list"
    assert analyze_question("昌平区位于哪里？").intent == "location"
    assert analyze_question("普实克出生于哪里？").intent == "birthplace"
    assert analyze_question("深圳地铁1号线有哪些站点").expects_complete_list is True
    assert analyze_question("中国有哪些著名关隘").expects_complete_list is False
    long_wall_passes = analyze_question("中国有哪些著名的长城关隘？")
    assert long_wall_passes.subjects[:2] == ("长城关隘列表", "长城关隘")
    reverse_object = analyze_question("奥的斯发明了什么？")
    assert reverse_object.intent == "agent"
    assert reverse_object.subjects == ("奥的斯 发明", "奥的斯")
    assert analyze_question("中国历史上开启丝绸之路的是谁？").intent == "agent"
    assert analyze_question("现在美国副总统是谁？").intent == "agent"
    assert analyze_question("2025年诺贝尔和平奖得主是谁？").intent == "agent"
    coordinated_time = analyze_question("香港和澳门分别是哪一年回归的？")
    assert coordinated_time.intent == "time"
    assert coordinated_time.subjects == ("香港", "香港回归", "澳门", "澳门回归")
    assert analyze_question("2026年世界杯决赛在哪个球场举行？").intent == "location"
    assert analyze_question("谁执导无无眠？").intent == "agent"


def test_reverse_agent_relations_preserve_full_subject() -> None:
    cases = {
        "说下渭水春風背后的創作者。": "渭水春风",
        "说下W創作社背后的創辦者。": "w创作社",
        "西北岁月的创作者是谁？": "西北岁月",
    }
    for question, expected_subject in cases.items():
        plan = build_query_plan(question)
        assert plan.analysis.intent == "agent"
        assert plan.subject == expected_subject
    assert build_query_plan("说下W創作社背后的創辦者。").relations[0] == "创办"


def test_cleans_colloquial_introduction_shell() -> None:
    plan = build_query_plan("介绍下宇树科技")

    assert plan.analysis.intent == "definition"
    assert plan.subject == "宇树科技"


def test_cause_plan_searches_exact_subject_event_title_first() -> None:
    plan = build_query_plan("美国为什么会走向衰落？")

    assert plan.subject == "美国"
    assert plan.queries[0] == "美国衰落"


def test_detects_only_real_ambiguity_candidates() -> None:
    candidates = ambiguity_candidates(
        "马昂是谁？",
        [source("马昂 (演员)", "演员。"), source("马昂 (明朝人物)", "官员。")],
    )
    assert candidates == ["马昂 (演员)", "马昂 (明朝人物)"]
    assert ambiguity_candidates("马昂是谁？", [source("马昂", "明朝人物。")]) == []
    assert ambiguity_candidates(
        "马昂是谁？",
        [source("马昂", "明朝人物。"), source("马昂 (消歧义)", "可以指多人。")],
    ) == []


def test_validates_citations_and_unsupported_numbers() -> None:
    sources = [source("示例", "示例成立于1994年，有3名成员。")]
    valid = validate_grounding("示例成立于1994年。[资料 1]", sources)
    assert valid.valid is True

    invalid = validate_grounding("示例成立于2001年，共5人。[资料 2]", sources)
    assert set(invalid.issues) == {"invalid_citation", "unsupported_number"}
    assert invalid.unsupported_numbers == ("2001", "5")
    assert remove_unsupported_number_sentences(
        "示例是一个组织。[资料 1]成立于2001年。[资料 1]",
        ("2001",),
    ) == "示例是一个组织。[资料 1]"


def test_detects_incomplete_list_against_explicit_evidence_count() -> None:
    sources = [source("线路", "线路共有5站：甲站、乙站、丙站、丁站、戊站。")]
    result = validate_list_answer("列出线路所有站点", "包括：甲站、乙站、丙站。[资料 1]", sources)
    assert result.complete is False
    assert result.expected_count == 5
    assert result.answer_count == 3
    assert result.issues == ("count_mismatch",)

    result = validate_list_answer("线路有哪些站点", "包括：甲站、乙站、丙站。[资料 1]", sources)
    assert result.complete is False
