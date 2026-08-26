from llamaindex_retrieval.evidence_gate import (
    evaluate_answer_support,
    evaluate_evidence_gate,
    repair_answer_citations,
)
from llamaindex_retrieval.qa_analysis import analyze_question
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


def test_gate_requires_subject_and_explicit_ordinal_relation() -> None:
    question = "中国历史上第一个皇帝是谁？"
    analysis = analyze_question(question)

    passed = evaluate_evidence_gate(
        question,
        analysis,
        [source("皇帝", "在中国历史中，嬴政成为中原第一个皇帝。")],
    )
    failed = evaluate_evidence_gate(
        question,
        analysis,
        [source("中国历史", "明光宗继承皇位，后来由天启帝继承皇位。")],
    )

    assert passed.passed is True
    assert "第一个" in passed.matched_relation_terms
    assert failed.passed is False
    assert "relation_mismatch" in failed.issues


def test_gate_accepts_explicit_person_role_relation() -> None:
    question = "宇树科技创始人是谁？"
    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [source("宇树科技", "2016年，宇树科技创始人王兴兴创办了宇树科技。")],
        subject="宇树科技",
    )

    assert result.passed is True
    assert "创始人" in result.matched_relation_terms


def test_field_evidence_bypasses_brittle_title_string_matching() -> None:
    question = "中国的首都在哪里"
    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [source("中国首都", "北京自1949年后定为中华人民共和国首都。")],
        subject="中国",
        relations=("首都",),
        field_evidence_available=True,
        field_candidate_count=1,
    )

    assert result.passed is True
    assert result.issues == ()


def test_gate_uses_dynamic_planner_relation_outside_builtin_vocabulary() -> None:
    question = "火星计划由谁负责？"
    evidence = [source("火星计划", "火星计划首席科学家张三承担总体研究工作。")]

    without_dynamic_relation = evaluate_evidence_gate(
        question,
        analyze_question(question),
        evidence,
        subject="火星计划",
    )
    with_dynamic_relation = evaluate_evidence_gate(
        question,
        analyze_question(question),
        evidence,
        subject="火星计划",
        relations=("首席科学家",),
    )

    assert without_dynamic_relation.passed is False
    assert "relation_mismatch" in without_dynamic_relation.issues
    assert with_dynamic_relation.passed is True
    assert "首席科学家" in with_dynamic_relation.matched_relation_terms


def test_gate_rejects_partial_definition_entity_match() -> None:
    question = "阿尔法泽是谁？"
    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [source("阿尔法岛", "阿尔法岛是南极洲的岛屿。")],
    )

    assert result.passed is False
    assert "subject_mismatch" in result.issues


def test_gate_uses_structured_anchor_when_rule_subject_is_unavailable() -> None:
    question = "这个对象是谁？"
    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [source("阿尔法岛", "阿尔法岛是南极洲的岛屿。")],
        anchor_subject="阿尔法泽",
    )

    assert result.passed is False
    assert "subject_mismatch" in result.issues


def test_gate_accepts_embedded_definition_on_related_page() -> None:
    question = "037型反潜护卫艇是什么？"
    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [source("206型炮艇", "037型反潜护卫艇由6604型猎潜艇放大而来。")],
        subject="037型反潜护卫艇",
    )

    assert result.passed is True


def test_gate_rejects_bare_field_name_as_embedded_definition() -> None:
    question = "CPUID指的是什么？"
    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [source("处理器", "表格字段：核心代号、CPUID、步进、处理器插座。")],
        subject="CPUID",
    )

    assert result.passed is False


def test_gate_rejects_longer_entity_as_embedded_definition() -> None:
    question = "东平是谁？"
    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [source("东平县", "东平县是中国山东省泰安市下辖的一个县。")],
        subject="东平",
    )

    assert result.passed is False


def test_gate_rejects_wrong_page_for_complete_list() -> None:
    question = "深圳地铁1号线有哪些站点？"
    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [source("深圳地铁17号线", "深圳地铁17号线设有多个车站。")],
        subject="深圳地铁1号线",
    )

    assert result.passed is False
    assert "subject_title_mismatch" in result.issues


def test_gate_accepts_complete_list_route_alias() -> None:
    question = "罗宝线沿途停靠哪些站？"
    aliased_source = source(
        "深圳地铁1号线",
        "深圳地铁1号线曾称罗宝线，沿途共设30个车站。",
    ).model_copy(update={"metadata": {"aliases": ["罗宝线"]}})

    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [aliased_source],
        subject="罗宝线",
    )

    assert result.passed is True


def test_gate_ignores_parenthetical_title_base_as_an_alias() -> None:
    question = "秦始皇有哪些伟大成就？"
    qualified_source = source(
        "秦始皇 (歌剧)",
        "《秦始皇》是一部以秦始皇为原型的英语歌剧。",
    ).model_copy(update={"metadata": {"aliases": ["秦始皇"]}})

    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [qualified_source],
        subject="秦始皇",
    )

    assert result.passed is False
    assert "subject_title_mismatch" in result.issues


def test_gate_accepts_route_identified_by_endpoints() -> None:
    question = "连接罗湖和机场东的深圳地铁线路有哪些车站？"
    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [source("深圳地铁1号线", "深圳地铁1号线由罗湖站至机场东站，共设30个车站。")],
        subject="罗湖",
    )

    assert result.passed is True


def test_gate_allows_partial_list_from_parent_topic_page() -> None:
    question = "中国有哪些著名的长城关隘？"
    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [source("长城", "著名关城包括山海关、嘉峪关、玉门关、萧关和阳关。")],
        subject="中国",
    )

    assert result.passed is True


def test_gate_rejects_partial_location_page_title() -> None:
    question = "昌平区位于哪里？"
    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [source("昌平区中西医结合医院", "医院位于北京市昌平区。")],
        subject="昌平区",
    )

    assert result.passed is False
    assert "subject_title_mismatch" in result.issues


def test_gate_rejects_related_but_different_cause_page() -> None:
    question = "明朝为什么灭亡？"
    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [source("明朝经济", "明朝经济发展经历了不同阶段。")],
        subject="明朝",
    )

    assert result.passed is False
    assert "subject_title_mismatch" in result.issues


def test_gate_accepts_subject_plus_event_cause_page() -> None:
    question = "美国为什么会走向衰落？"
    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [source("美国衰落", "美国衰落是一个讨论国家实力变化的主题。")],
        subject="美国",
    )

    assert result.passed is True


def test_gate_accepts_subject_plus_event_time_and_list_pages() -> None:
    time_result = evaluate_evidence_gate(
        "香港是哪一年回归的？",
        analyze_question("香港是哪一年回归的？"),
        [source("香港回归", "1997年7月1日，中国恢复对香港行使主权。")],
        subject="香港",
    )
    list_result = evaluate_evidence_gate(
        "中国从古至今总共经历了哪些朝代？",
        analyze_question("中国从古至今总共经历了哪些朝代？"),
        [source("中国朝代", "中国主要朝代包括夏、商、周、秦、汉。")],
        subject="中国",
    )

    assert time_result.passed is True
    assert list_result.passed is True


def test_gate_accepts_normalized_world_cup_title() -> None:
    question = "2030年世界杯由哪些国家主办？"
    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [source("2030年国际足协世界杯", "赛事由西班牙、葡萄牙和摩洛哥主办。")],
        subject="2030年世界杯",
    )

    assert result.passed is True


def test_gate_rejects_evidence_without_requested_year() -> None:
    question = "2025年诺贝尔和平奖得主是谁？"
    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [source("诺贝尔和平奖", "截至2024年，共有111位个人获奖。")],
        subject="2025年诺贝尔和平奖",
    )

    assert result.passed is False
    assert "temporal_mismatch" in result.issues


def test_time_gate_accepts_explicit_date_for_end_question() -> None:
    question = "谭德塞第二任期什么时候结束？"
    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [source("谭德塞", "2022年5月24日获得连任，任期5年。")],
        subject="谭德塞第二任期",
    )

    assert result.passed is True


def test_gate_accepts_agent_relation_synonym_on_exact_topic_page() -> None:
    question = "中国历史上开启丝绸之路的是谁？"
    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [source("丝绸之路", "汉武帝派张骞出使西域，史称凿空。")],
        subject="丝绸之路",
    )

    assert result.passed is True
    assert "出使" in result.matched_relation_terms


def test_gate_maps_colloquial_owner_to_creator_relation() -> None:
    question = "宇树科技老板是谁？"
    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [source("宇树科技", "宇树科技创始人王兴兴在2016年创办了宇树科技。")],
        subject="宇树科技",
        relations=("老板",),
    )

    assert result.passed is True
    assert "创始人" in result.matched_relation_terms


def test_gate_allows_broad_scope_prefix_for_exact_page() -> None:
    question = "中国四大名著？"
    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [source("四大名著", "四大名著是指《三国演义》《西游记》《水浒传》《红楼梦》。")],
        subject="中国四大名著",
    )

    assert result.passed is True


def test_gate_accepts_relation_evidence_on_parent_topic_page() -> None:
    question = "现代主义建筑发展所依赖的安全电梯由谁发明？"
    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [source("电梯", "安全电梯使用的安全钳由奥的斯发明。")],
        subject="安全电梯",
    )

    assert result.passed is True


def test_gate_accepts_creator_question_when_evidence_says_created() -> None:
    question = "渭水春风的创作者是谁？"
    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [source("渭水春风", "该音乐剧由音乐时代剧场创作，并由多人共同编剧。")],
        subject="渭水春风",
    )

    assert result.passed is True
    assert "创作" in result.matched_relation_terms


def test_gate_accepts_current_office_on_exact_topic_page() -> None:
    question = "现在美国总统是谁？"
    result = evaluate_evidence_gate(
        question,
        analyze_question(question),
        [source("美国总统", "现任美国总统于2025年1月20日上任。")],
        subject="美国总统",
    )

    assert result.passed is True


def test_answer_support_uses_only_cited_evidence() -> None:
    sources = [
        source("皇帝", "嬴政成为中原第一个皇帝，称始皇帝。"),
        source("明朝", "李自成攻克北京，崇祯帝自缢。"),
    ]

    supported = evaluate_answer_support("第一个皇帝是嬴政。[资料 1]", sources)
    unsupported = evaluate_answer_support("第一个皇帝是李自成。[资料 1]", sources)

    assert supported.passed is True
    assert unsupported.passed is False
    assert "unsupported_entity_term" in unsupported.issues


def test_refusal_answer_always_passes_support_gate() -> None:
    result = evaluate_answer_support("根据检索到的资料，无法确定。", [])

    assert result.passed is True


def test_answer_support_includes_cited_document_sibling_chunks() -> None:
    sources = [
        source("明朝", "魏忠贤打击东林党，崇祯帝频繁更换内阁大学士。"),
        source("明朝", "长期干旱、蝗灾造成粮食歉收和饥荒，随后爆发民变。"),
    ]
    sources[0].id = "ming-1"
    sources[1].id = "ming-2"
    sources[0].document_id = "ming"
    sources[1].document_id = "ming"

    result = evaluate_answer_support(
        "长期干旱和蝗灾造成粮食歉收，朝政也受到魏忠贤专权影响。[资料 1]",
        sources,
    )

    assert result.passed is True


def test_answer_support_does_not_treat_descriptive_words_as_entities() -> None:
    result = evaluate_answer_support(
        "朝政长期腐败，党争十分激烈。[资料 1]",
        [source("明朝", "朝廷党争不断，政治日益腐败。")],
    )

    assert "unsupported_entity_term" not in result.issues


def test_answer_support_checks_only_facts_added_beyond_the_question() -> None:
    result = evaluate_answer_support(
        "水浒传里赤手空拳打死老虎的是武松。[资料 1]",
        [source("武松", "武松是《水浒传》人物，以景阳冈打虎而闻名。")],
        question="水浒传里赤手空拳打死老虎的是谁？",
    )

    assert result.passed is True
    assert result.unsupported_terms == ()


def test_repair_answer_citations_maps_comparison_clauses_to_sources() -> None:
    sources = [
        source("尺八", "尺八是竹制木管乐器，音色苍凉辽阔。"),
        source("长笛", "长笛是高音旋律乐器，现代多使用金属材质。"),
    ]

    repaired = repair_answer_citations(
        "尺八是竹制木管乐器，音色苍凉辽阔；长笛是高音旋律乐器，现代多使用金属材质。[资料 1]",
        sources,
    )

    assert repaired == (
        "尺八是竹制木管乐器，音色苍凉辽阔[资料 1]；"
        "长笛是高音旋律乐器，现代多使用金属材质[资料 2]。"
    )
    assert evaluate_answer_support(repaired, sources).passed is True


def test_repair_answer_citations_points_to_exact_sibling_chunk() -> None:
    sources = [
        source("明朝", "魏忠贤打击东林党，朝廷党争不断。"),
        source("明朝", "长期干旱和蝗灾导致粮食歉收，爆发民变。"),
    ]
    sources[0].document_id = "ming"
    sources[1].document_id = "ming"

    repaired = repair_answer_citations(
        "明朝长期干旱和蝗灾导致粮食歉收，爆发民变。[资料 1]",
        sources,
    )

    assert repaired.endswith("[资料 2]。")
