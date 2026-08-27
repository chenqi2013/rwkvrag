import json

import httpx
import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.evidence_utils import (
    agent_evidence_answer,
    birthplace_evidence_answer,
    coordinated_time_evidence_answer,
    definition_evidence_answer,
    direct_evidence_answer,
    structured_list_answer,
    time_evidence_answer,
    location_evidence_answer,
    ordinal_evidence_answer,
    quantitative_evidence_answer,
)
from llamaindex_retrieval.evidence_quality import is_repetitive_garbage
from llamaindex_retrieval.generation import EvidenceAnswerGenerator
from llamaindex_retrieval.schemas import SourceItem


def source() -> SourceItem:
    return SourceItem(
        id="capital-1",
        document_id="capital",
        source="finewiki-zh",
        title="首都",
        score=1.0,
        snippet="中华人民共和国的首都是北京。",
    )


def test_cause_prompt_forbids_post_event_and_external_causes() -> None:
    generator = EvidenceAnswerGenerator(Settings())

    prompt = generator._prompt("明朝为什么灭亡？", [source()])

    assert "不得使用模型常识补充资料未出现的原因" in prompt
    assert "不得把事件发生后的结果当作原因" in prompt
    assert "必须同时概括主要长期因素和直接导火事件" in prompt
    assert "必须一次完整列出" in prompt


def test_fact_prompt_does_not_add_cause_only_instruction() -> None:
    generator = EvidenceAnswerGenerator(Settings())

    prompt = generator._prompt("中国的首都是哪里？", [source()])

    assert "不得把事件发生后的结果当作原因" not in prompt


def test_capital_answer_prefers_current_explicit_capital_sentence() -> None:
    evidence = SourceItem(
        id="china-capital",
        document_id="china-capital",
        source="finewiki-zh",
        title="中国首都",
        score=1.0,
        snippet=(
            "清朝入主中原后将北京定为国都。\n"
            "现时北京自1949年後定为中华人民共和国首都。"
        ),
    )

    assert direct_evidence_answer("中国的首都是哪个城市？", [evidence]) == (
        "中国的首都是北京。[资料 1]"
    )


def test_counted_list_extracts_inline_encyclopedia_enumeration() -> None:
    evidence = SourceItem(
        id="four-classics",
        document_id="four-classics",
        source="finewiki-zh",
        title="四大名著",
        score=1.0,
        snippet=(
            "四大名著，即四大小说名著，是指《三国演义》《西游记》"
            "《水浒传》《红楼梦》4部中国古典章回小说。"
        ),
    )

    assert structured_list_answer("中国四大名著是哪四个？", [evidence]) == (
        "四大名著包括：《三国演义》、《西游记》、《水浒传》、《红楼梦》。[资料 1]"
    )


def test_uncounted_list_extracts_inline_quoted_enumeration() -> None:
    evidence = source().model_copy(
        update={
            "title": "四大名著",
            "snippet": "四大名著，是指《三国演义》《西游记》《水浒传》《红楼梦》4部中国古典小说。",
        }
    )

    assert structured_list_answer("中国四大名著是哪几个？", [evidence]) == (
        "四大名著包括：《三国演义》、《西游记》、《水浒传》、《红楼梦》。[资料 1]"
    )


def test_author_answer_does_not_treat_mingzhu_as_authorship_relation() -> None:
    commentary = SourceItem(
        id="journey-west-commentary",
        document_id="journey-west",
        source="finewiki-zh",
        title="西游记",
        score=1.0,
        snippet=(
            "有研究者认为作者在书中对道士多有贬斥，"
            "可见作者不是站在空门的立场上来写《西游记》的。"
        ),
    )
    evidence = SourceItem(
        id="journey-west",
        document_id="journey-west",
        source="finewiki-zh",
        title="西游记",
        score=1.0,
        snippet=(
            "《西游记》是中国四大名著之一。"
            "成书于16世纪明朝中叶，一般认为作者是明朝的吴承恩。"
        ),
    )

    assert agent_evidence_answer("西游记是哪个作者写的？", [commentary, evidence]) == (
        "成书于16世纪明朝中叶，一般认为作者是明朝的吴承恩。 [资料 2]"
    )


def test_author_answer_prefers_attribution_over_bibliography() -> None:
    bibliography = SourceItem(
        id="bibliography",
        document_id="journey-west",
        source="finewiki-zh",
        title="西游记",
        score=1.0,
        snippet="《〈西游记〉作者之谜》，《社会科学报》，1996年12月12日。",
    )
    attribution = SourceItem(
        id="attribution",
        document_id="journey-west",
        source="finewiki-zh",
        title="西游记",
        score=1.0,
        snippet="成书于16世纪明朝中叶，一般认为作者是明朝的吴承恩。",
    )

    answer = agent_evidence_answer(
        "西游记是谁写的？",
        [bibliography, attribution],
        ("作者", "撰写", "创作", "著"),
    )

    assert "吴承恩" in answer
    assert "社会科学报" not in answer


def test_list_prompt_requires_all_evidence_and_role_distinction() -> None:
    generator = EvidenceAnswerGenerator(Settings())

    prompt = generator._prompt("2030年世界杯由哪些国家主办？", [source()])

    assert "必须检查全部资料" in prompt
    assert "不能把主办、承办、参加等不同关系混为一谈" in prompt
    assert "不得仅因无法证明已经穷尽而拒答" in prompt


def test_prompt_includes_structured_relation_contract() -> None:
    generator = EvidenceAnswerGenerator(Settings())

    prompt = generator._prompt(
        "马斯克创办了哪几家公司？",
        [source()],
        subject="马斯克",
        relations=("创办", "创立"),
    )

    assert '"subject": "马斯克"' in prompt
    assert '"relations": ["创办", "创立"]' in prompt
    assert "每个“对象—关系—具体值”必须由同一条资料中的同一句或相邻句直接支持" in prompt
    assert "不得把资料中的投资人、负责人、成员、收购方等其他角色改写成任务所问的关系" in prompt


def test_prompt_treats_evaluative_words_as_selection_not_required_quotes() -> None:
    generator = EvidenceAnswerGenerator(Settings())
    evidence = source().model_copy(
        update={
            "title": "秦始皇",
            "snippet": "秦始皇统一六国，并推行书同文。",
        }
    )

    prompt = generator._prompt("秦始皇有哪些伟大成就？", [evidence])

    assert "不要求资料逐字出现该评价词" in prompt
    assert "不要自行夸大评价" in prompt


def test_ordinal_prompt_requires_explicit_first_relation() -> None:
    generator = EvidenceAnswerGenerator(Settings())

    prompt = generator._prompt("中国历史上第一个皇帝是谁？", [source()])

    assert "只能依据资料中明确出现的“第一个、第一位、首位或最早”关系作答" in prompt


def test_semantic_prompt_uses_task_contract_without_question_word_rules() -> None:
    generator = EvidenceAnswerGenerator(Settings(semantic_pipeline_enabled=True))

    prompt = generator._prompt(
        "换一种说法也要找到这条事实",
        [source()],
        subject="中华人民共和国",
        relations=("首都", "国都"),
        answer_shape="single_fact",
        set_semantics="specific",
        fields=(("f1", "首都城市", ("首都", "国都")),),
    )

    assert '"answer_shape": "single_fact"' in prompt
    assert '"field_id": "f1"' in prompt
    assert '"question": "首都城市"' in prompt
    assert "任务契约中的 subject 是待处理对象" in prompt
    assert "引用编号可选" in prompt
    assert "绝不能只输出“[资料 N]”" in prompt
    assert "不得把事件发生后的结果当作原因" not in prompt
    assert "只能依据资料中明确出现的“第一个" not in prompt


def test_agent_answer_extracts_explicit_founder_role() -> None:
    evidence = source().model_copy(
        update={
            "title": "宇树科技",
            "snippet": "2016年，宇树科技创始人王兴兴开发了四足机器人，随后创办了宇树科技。",
        }
    )

    answer = agent_evidence_answer("宇树科技创始人是谁？", [evidence])

    assert answer is not None
    assert "创始人王兴兴" in answer
    assert answer.endswith("[资料 1]")


def test_ordinal_answer_extracts_explicit_first_relation_sentence() -> None:
    evidence = SourceItem(
        id="emperor",
        document_id="emperor",
        source="finewiki-zh",
        title="皇帝",
        score=1.0,
        snippet=(
            "皇帝 > 东亚 > 中国\n\n"
            "在中国历史中，嬴政创建了皇帝制度，自己成为中原第一个皇帝，称始皇帝。"
        ),
    )

    assert ordinal_evidence_answer("中国历史上第一个皇帝是谁？", [evidence]) == (
        "在中国历史中，嬴政创建了皇帝制度，自己成为中原第一个皇帝，称始皇帝。 [资料 1]"
    )


def test_time_answer_derives_fixed_term_end_from_grounded_dates() -> None:
    evidence = SourceItem(
        id="tedros",
        document_id="tedros",
        source="finewiki-zh",
        title="谭德塞",
        score=1.0,
        snippet="2022年5月24日，在没有其他候选人的情况下获得连任，任期5年。",
    )

    assert time_evidence_answer("谭德塞第二任期什么时候结束？", [evidence]) == (
        "按资料所载连任日期和5年任期推算，该任期于2027年5月24日结束。[资料 1]"
    )


def test_definition_grounding_requires_complete_entity_name() -> None:
    evidence = SourceItem(
        id="alpha-island",
        document_id="alpha-island",
        source="finewiki-zh",
        title="阿尔法岛",
        score=1.0,
        snippet="阿尔法岛是南极洲的岛屿。",
    )

    assessment = EvidenceAnswerGenerator.assess_evidence("阿尔法泽是谁？", [evidence])

    assert assessment.anchors == {"阿尔法泽"}
    assert assessment.matched_anchors == set()
    assert assessment.grounded is False


def test_structured_subject_rejects_partial_entity_overlap() -> None:
    evidence = SourceItem(
        id="alpha-island",
        document_id="alpha-island",
        source="finewiki-zh",
        title="阿尔法岛",
        score=1.0,
        snippet="阿尔法岛是南极洲的岛屿。",
    )

    assessment = EvidenceAnswerGenerator.assess_evidence(
        "这个对象是谁？",
        [evidence],
        subject="阿尔法泽",
    )

    assert assessment.anchors == {"阿尔法泽"}
    assert assessment.matched_anchors == set()
    assert assessment.grounded is False


def test_structured_subject_matches_tokenized_entity_and_metadata_alias() -> None:
    evidence = SourceItem(
        id="metro-line",
        document_id="metro-line",
        source="finewiki-zh",
        title="深圳地铁1号线",
        score=1.0,
        snippet="该线路由罗湖站至机场东站，共设30个车站。",
        metadata={"aliases": ["罗宝线"]},
    )

    canonical = EvidenceAnswerGenerator.assess_evidence(
        "这条线路有哪些站？",
        [evidence],
        subject="深圳地铁1号线",
    )
    aliased = EvidenceAnswerGenerator.assess_evidence(
        "这条线路有哪些站？",
        [evidence],
        subject="罗宝线",
    )

    assert canonical.matched_anchors == {"深圳地铁1号线"}
    assert aliased.matched_anchors == {"罗宝线"}
    assert canonical.grounded is True
    assert aliased.grounded is True


def test_definition_answer_extracts_embedded_definition_sentence() -> None:
    evidence = SourceItem(
        id="boat",
        document_id="boat",
        source="finewiki-zh",
        title="206型炮艇",
        score=1.0,
        snippet=(
            "206型炮艇由037型反潜护卫艇缩小，"
            "而037型反潜护卫艇由6604型猎潜艇放大而来。"
        ),
    )

    assert definition_evidence_answer("037型反潜护卫艇是什么？", [evidence]) == (
        "206型炮艇由037型反潜护卫艇缩小，而037型反潜护卫艇由6604型猎潜艇放大而来。 "
        "[资料 1]"
    )


def test_definition_answer_does_not_use_longer_entity_name() -> None:
    evidence = SourceItem(
        id="dongping-county",
        document_id="dongping-county",
        source="finewiki-zh",
        title="东平县",
        score=1.0,
        snippet="东平县是中国山东省泰安市下辖的一个县。",
    )

    assert definition_evidence_answer("东平是谁？", [evidence]) is None


def test_definition_answer_prefers_earlier_identity_over_shorter_detail() -> None:
    evidence = SourceItem(
        id="person-1",
        document_id="person",
        source="finewiki-zh",
        title="示例人物",
        score=1.0,
        snippet=(
            "示例人物是中国作家、教育家，曾长期从事文学创作。"
            "示例人物是协会会员。"
        ),
    )

    assert definition_evidence_answer("示例人物是谁？", [evidence]) == (
        "示例人物是中国作家、教育家，曾长期从事文学创作。 [资料 1]"
    )


def test_introduction_extends_a_short_definition_but_ignores_later_relation_word() -> None:
    evidence = SourceItem(
        id="place-1",
        document_id="place",
        source="finewiki-zh",
        title="示例县",
        score=1.0,
        snippet=(
            "示例县是某市下辖县。该县位于河流北岸，以农业和旅游业为主。"
            "示例县后来建设体育场，作为当地比赛场地。"
        ),
    )

    assert definition_evidence_answer("请简要介绍示例县。", [evidence]) == (
        "示例县是某市下辖县。该县位于河流北岸，以农业和旅游业为主。 [资料 1]"
    )


def test_definition_extraction_rejects_time_and_agent_questions() -> None:
    evidence = SourceItem(
        id="ccp",
        document_id="ccp",
        source="finewiki-zh",
        title="中国共产党",
        score=1.0,
        snippet="中国共产党是中华人民共和国的执政党，成立于1921年7月23日。",
    )
    assert definition_evidence_answer("中国共产党是什么时候成立的", [evidence]) is None
    assert definition_evidence_answer("中国共产党是谁创立的", [evidence]) is None


def test_station_list_rejects_train_fleet_list() -> None:
    evidence = SourceItem(
        id="fleet",
        document_id="metro",
        source="finewiki-zh",
        title="深圳地铁",
        score=1.0,
        snippet="深圳地铁 > 列车 > 列表\n线路列表：1号线、2号线、3号线、4号线。",
        metadata={"content_type": "table_summary"},
    )
    assert structured_list_answer("深圳地铁1号线有哪些站点", [evidence]) is None


def test_long_bullet_rows_are_extracted_from_exact_section() -> None:
    evidence = SourceItem(
        id="works",
        document_id="artist",
        source="finewiki-zh",
        title="示例漫画家",
        score=1.0,
        snippet=(
            "示例漫画家 > 作品列表\n"
            "- 校园故事（《周刊示例》2002年47号－2008年34号，全22册）\n"
            "- 夏日风暴（《月刊示例》2006年10号－2010年10号，全8册）\n"
            "- 一路平安（《别册示例》2011年6号－2012年6号，全2册）"
        ),
        metadata={"content_type": "list"},
    )

    answer = structured_list_answer("示例漫画家的作品列表有哪些？", [evidence])

    assert answer is not None
    assert "校园故事" in answer
    assert "夏日风暴" in answer
    assert "一路平安" in answer


def test_location_prefers_sentence_whose_subject_is_the_page_title() -> None:
    sources = [
        SourceItem(
            id="lead",
            document_id="town",
            source="finewiki-zh",
            title="示例镇",
            score=1.0,
            snippet="示例镇是位于示例县东部的城镇。当地工业发达。",
        ),
        SourceItem(
            id="industry",
            document_id="town",
            source="finewiki-zh",
            title="示例镇",
            score=0.9,
            snippet="示例糖厂位于镇中心，现已停止运营。",
        ),
    ]

    assert location_evidence_answer("想去示例镇，它的位置在哪里？", sources) == (
        "示例镇是位于示例县东部的城镇。 [资料 1]"
    )


def test_location_ignores_ascii_question_mark_inside_name_annotation() -> None:
    evidence = SourceItem(
        id="town",
        document_id="town",
        source="finewiki-zh",
        title="长泉町",
        score=1.0,
        snippet="长泉町（日语：Nagaizumi chō */?）是位于日本静冈县东部的町。",
    )

    answer = location_evidence_answer("长泉町位于哪里？", [evidence])

    assert answer is not None
    assert "日本静冈县东部" in answer


def test_population_data_is_extracted_without_using_density() -> None:
    evidence = SourceItem(
        id="town",
        document_id="town",
        source="finewiki-zh",
        title="切帕赫",
        score=1.0,
        snippet="切帕赫是瑞士城镇，面积1.84平方公里，2012年人口189，人口密度每平方公里103人。",
    )

    assert quantitative_evidence_answer("切帕赫的人口数据？", [evidence]) == (
        "切帕赫的人口为189人。[资料 1]"
    )


def test_station_list_repairs_truncated_single_character_name_from_context() -> None:
    summary = SourceItem(
        id="summary",
        document_id="metro",
        source="finewiki-zh",
        title="示例地铁1号线",
        score=1.0,
        snippet="站名列表：甲城、瑞、机场东。",
        metadata={"content_type": "table_summary"},
    )
    context = SourceItem(
        id="context",
        document_id="metro",
        source="finewiki-zh",
        title="示例地铁1号线",
        score=0.9,
        snippet="一趟列车在后瑞站发生故障停运。",
    )
    assert structured_list_answer("示例地铁1号线有哪些站点", [summary, context]) == (
        "示例地铁1号线的站名包括：甲城、后瑞、机场东。[资料 1]"
    )


def test_station_list_repairs_name_after_conjunction() -> None:
    summary = SourceItem(
        id="summary",
        document_id="metro",
        source="finewiki-zh",
        title="示例地铁1号线",
        score=1.0,
        snippet="站名列表：甲城、瑞、机场东。",
        metadata={"content_type": "table_summary"},
    )
    context = SourceItem(
        id="context",
        document_id="metro",
        source="finewiki-zh",
        title="示例地铁1号线",
        score=0.9,
        snippet="机场东站和后瑞站都是高架车站。",
    )

    assert structured_list_answer("示例地铁1号线有哪些站点", [summary, context]) == (
        "示例地铁1号线的站名包括：甲城、后瑞、机场东。[资料 1]"
    )


def test_flattened_station_table_extracts_only_complete_station_rows() -> None:
    overview = SourceItem(
        id="line",
        document_id="line",
        source="finewiki-zh",
        title="示例地铁1号线",
        score=1.0,
        snippet="示例地铁1号线前称示例线，共设3个车站。",
    )
    table = SourceItem(
        id="station-list",
        document_id="station-list",
        source="finewiki-zh",
        title="示例地铁车站列表",
        score=0.9,
        snippet=(
            "1號線前稱示例線，沿途共設3個車站。\n"
            "車站圖片所在區域轉乘路綫通車日期備註參考來源"
            "甲城站Jiacheng150px福田區2020年1月1日"
            "乙城站Yicheng150px-{后}-瑞站Hourui150px"
        ),
        metadata={"chunk_order": 1},
    )

    assert structured_list_answer(
        "示例地铁1号线有哪些站点", [overview, table]
    ) == "示例地铁1号线共有3个车站：甲城站、乙城站、后瑞站。[资料 2]"


def test_dynasty_table_aggregates_dynasty_field_without_confusing_other_columns() -> None:
    first = SourceItem(
        id="dynasty-1",
        document_id="dynasty",
        source="finewiki-zh",
        title="中国朝代",
        score=1.0,
        snippet=(
            "中国朝代 > 中国主要朝代列表\n"
            "朝代：夏朝；统治家族/民族：姒；建立：约公元前2070年\n"
            "朝代：商朝；统治家族/民族：子；建立：约公元前1600年\n"
        ),
        metadata={"content_type": "table"},
    )
    second = SourceItem(
        id="dynasty-2",
        document_id="dynasty",
        source="finewiki-zh",
        title="中国朝代",
        score=0.9,
        snippet=(
            "朝代：周朝；统治家族/民族：姬；建立：约公元前1046年\n"
            "朝代：秦朝；统治家族/民族：嬴；建立：公元前221年\n"
        ),
        metadata={"content_type": "table"},
    )

    assert structured_list_answer(
        "中国从古至今总共经历了哪些朝代？", [first, second]
    ) == "中国朝代的朝代包括：夏朝、商朝、周朝、秦朝。[资料 1]"


def test_time_and_agent_answers_select_relevant_evidence_sentences() -> None:
    time_source = SourceItem(
        id="ccp",
        document_id="ccp",
        source="finewiki-zh",
        title="中国共产党",
        score=1.0,
        snippet="中国共产党是一个政党。1921年7月23日正式组建为中国共产党。",
    )
    assert time_evidence_answer("中国共产党是什么时候成立的", [time_source]) == (
        "1921年7月23日正式组建为中国共产党。 [资料 1]"
    )

    duration = time_evidence_answer("中国共产党成立多久了？", [time_source])
    assert duration is not None
    assert "成立于1921年7月23日" in duration
    assert "已成立" in duration


def test_birth_year_prefers_lead_lifespan_over_later_activity_year() -> None:
    sources = [
        SourceItem(
            id="lead",
            document_id="person",
            source="finewiki-zh",
            title="示例人物",
            score=1.0,
            snippet="示例人物（1882年3月3日—1961年1月28日），日本画家。",
        ),
        SourceItem(
            id="life",
            document_id="person",
            source="finewiki-zh",
            title="示例人物",
            score=0.9,
            snippet="示例人物出生于浅草区，在1890年学习绘画。",
        ),
    ]

    assert time_evidence_answer("示例人物是哪一年出生的？", sources) == (
        "示例人物出生于1882年3月3日。[资料 1]"
    )


def test_death_year_does_not_use_relative_lifespan() -> None:
    sources = [
        SourceItem(
            id="family",
            document_id="person",
            source="finewiki-zh",
            title="王沃",
            score=1.0,
            snippet=(
                "王沃于1943年8月25日逝世。"
                "王沃长子王石定（1913年10月20日—1947年3月6日）曾任高雄市参议员。"
            ),
        ),
        SourceItem(
            id="lead",
            document_id="person",
            source="finewiki-zh",
            title="王沃",
            score=0.9,
            snippet="王沃（1887年7月13日—1943年8月），台湾日治时期企业家。",
        ),
    ]

    assert time_evidence_answer("王沃是哪一年逝世的？", sources) == (
        "王沃逝世于1943年8月。[资料 2]"
    )


def test_year_question_does_not_prefer_later_month_precision() -> None:
    evidence = SourceItem(
        id="bank",
        document_id="bank",
        source="finewiki-zh",
        title="示例银行",
        score=1.0,
        snippet=(
            "示例银行一般认为创建于1938年下半年。"
            "1940年8月成立示例银行总行，原银行改为分行。"
        ),
    )

    answer = time_evidence_answer("示例银行是哪一年成立的？", [evidence])

    assert answer is not None
    assert "1938年" in answer


def test_entity_origin_precedes_later_subsidiary_creation() -> None:
    evidence = SourceItem(
        id="company",
        document_id="company",
        source="finewiki-zh",
        title="示例公司",
        score=1.0,
        snippet=(
            "示例公司是一家制造企业，成立于1928年。"
            "2002年，示例集团成立，次年设立子公司示例公司有限公司。"
        ),
    )

    answer = time_evidence_answer("示例公司是哪一年成立的？", [evidence])

    assert answer is not None
    assert "1928年" in answer
    assert "2002年" not in answer


def test_entity_origin_accepts_year_before_established_verb() -> None:
    evidence = SourceItem(
        id="studio",
        document_id="studio",
        source="finewiki-zh",
        title="Frog City Software",
        score=1.0,
        snippet=(
            "Frog City Software，是一个游戏开发商，于在1995年成立。"
            "2006年4月那年，Frog City Software的部分员工创建了新的工作室。"
        ),
    )

    answer = time_evidence_answer("Frog City Software是什么时候成立的？", [evidence])

    assert answer is not None
    assert "1995年" in answer
    assert "2006年" not in answer


def test_coordinated_time_answer_extracts_each_subject() -> None:
    sources = [
        SourceItem(
            id="hong-kong",
            document_id="hong-kong",
            source="wiki",
            title="香港回归",
            score=1.0,
            snippet="1997年7月1日，香港回归中国。",
        ),
        SourceItem(
            id="macao",
            document_id="macao",
            source="wiki",
            title="澳门回归",
            score=1.0,
            snippet="1999年12月20日，澳门回归中国。",
        ),
    ]

    answer = coordinated_time_evidence_answer(
        sources,
        ("香港", "香港回归", "澳门", "澳门回归"),
    )

    assert answer == (
        "香港：1997年7月1日，香港回归中国。 [资料 1]；"
        "澳门：1999年12月20日，澳门回归中国。 [资料 2]"
    )


def test_coordinated_time_answer_is_not_validated_as_generic_list() -> None:
    sources = [
        SourceItem(
            id="hong-kong",
            document_id="hong-kong",
            source="finewiki-zh",
            title="香港回归",
            score=1.0,
            snippet="1997年7月1日，香港回归中国。",
        ),
        SourceItem(
            id="macao",
            document_id="macao",
            source="finewiki-zh",
            title="澳门回归",
            score=0.9,
            snippet="1999年12月20日，澳门回归中国。",
        ),
    ]
    answer = coordinated_time_evidence_answer(
        sources,
        ("香港", "香港回归", "澳门", "澳门回归"),
    )

    from llamaindex_retrieval.qa_analysis import validate_list_answer

    validation = validate_list_answer("香港和澳门分别是哪一年回归的？", answer or "", sources)
    assert validation.complete is None


def test_agent_origin_prefers_initial_actor_over_later_reopening() -> None:
    sources = [
        SourceItem(
            id="lead",
            document_id="silk-road",
            source="finewiki-zh",
            title="丝绸之路",
            score=1.0,
            snippet="东汉时班超再次出使西域，打通了荒废已久的丝绸之路。",
        ),
        SourceItem(
            id="route",
            document_id="silk-road",
            source="finewiki-zh",
            title="丝绸之路",
            score=0.9,
            snippet="东段从长安出发。（西汉时期由张骞开辟，东汉时期由班超打通。）",
        ),
    ]

    answer = agent_evidence_answer("中国历史上开启丝绸之路的是谁？", sources)
    assert answer is not None
    assert "张骞" in answer
    assert "资料 2" in answer


def test_agent_origin_combines_start_location_for_compound_question() -> None:
    sources = [
        SourceItem(
            id="route",
            document_id="silk-road",
            source="finewiki-zh",
            title="丝绸之路",
            score=1.0,
            snippet="东段从长安或洛阳出发。（西汉时期由张骞开辟，东汉时期由班超打通。）",
        )
    ]

    answer = agent_evidence_answer(
        "中国历史上是谁开启了丝绸之路，是从哪个地方开始的？",
        sources,
    )
    assert answer is not None
    assert answer.startswith("西汉时期由张骞开辟")
    assert "起点为长安或洛阳" in answer

    handover_source = SourceItem(
        id="handover",
        document_id="handover",
        source="finewiki-zh",
        title="澳门回归",
        score=1.0,
        snippet="澳門回歸指1999年12月20日中華人民共和國政府對澳門恢復行使主權。",
    )
    assert time_evidence_answer("澳门是什么时候回归中国的", [handover_source]) == (
        "澳門回歸指1999年12月20日中華人民共和國政府對澳門恢復行使主權。 [资料 1]"
    )
    hong_kong_source = SourceItem(
        id="hong-kong",
        document_id="hong-kong",
        source="finewiki-zh",
        title="香港回归",
        score=1.0,
        snippet="1997年7月1日，香港交还中华人民共和国。",
    )
    assert time_evidence_answer(
        "澳门是什么时候回归中国的",
        [hong_kong_source, handover_source],
    ) == "澳門回歸指1999年12月20日中華人民共和國政府對澳門恢復行使主權。 [资料 2]"

    battle_source = SourceItem(
        id="battle",
        document_id="battle",
        source="finewiki-zh",
        title="赤壁之战",
        score=1.0,
        snippet="赤壁之战是东汉末年曹操南攻荆州之战役。孙刘联军随后发动火攻。",
    )
    assert agent_evidence_answer("赤壁之战是谁发起的", [battle_source]) == (
        "赤壁之战是东汉末年曹操南攻荆州之战役。 [资料 1]"
    )

    tang_source = SourceItem(
        id="tang",
        document_id="tang",
        source="finewiki-zh",
        title="唐朝",
        score=1.0,
        snippet="李渊于618年接受隋恭帝禅让，登基称帝并建立唐朝，是唐朝的开国皇帝。",
    )
    assert agent_evidence_answer("唐朝的开国皇帝是谁", [tang_source]) == (
        "李渊于618年接受隋恭帝禅让，登基称帝并建立唐朝，是唐朝的开国皇帝。 [资料 1]"
    )

    youtube_source = SourceItem(
        id="youtube",
        document_id="youtube",
        source="finewiki-zh",
        title="YouTube",
        score=1.0,
        snippet="YouTube由查德·赫利、陈士骏和贾德·卡林三名前PayPal员工于2005年2月创立。",
    )
    assert agent_evidence_answer("YouTube是由哪几个人创立的？", [youtube_source]) == (
        "YouTube由查德·赫利、陈士骏和贾德·卡林三名前PayPal员工于2005年2月创立。 [资料 1]"
    )

    place_source = SourceItem(
        id="place",
        document_id="place",
        source="finewiki-zh",
        title="示例人物",
        score=1.0,
        snippet="示例人物出生于布拉格。示例城位于河流北岸。",
    )
    assert birthplace_evidence_answer("示例人物出生于哪里", [place_source]) == (
        "示例人物出生于布拉格。 [资料 1]"
    )
    assert location_evidence_answer("示例城位于哪里", [place_source]) == (
        "示例城位于河流北岸。 [资料 1]"
    )


def test_generator_removes_bracketed_chat_protocol_markers() -> None:
    assert (
        EvidenceAnswerGenerator._clean_answer("[助手 1] 北京。[资料 1]\n[用户] 中国首都是哪里？")
        == "北京。[资料 1]"
    )


def test_detects_periodic_parser_garbage_without_rejecting_normal_prose() -> None:
    assert is_repetitive_garbage("唐朝 > 军定\n\n" + "整須教定開領思" * 40) is True
    assert is_repetitive_garbage(
        "唐高祖李渊是唐朝开国皇帝。他于618年称帝并建立唐朝，此后逐步完成统一。" * 4
    ) is False


def test_generator_removes_bracketed_thinking_and_answer_markers() -> None:
    raw = "[思考] 用户问首都，资料说北京。\n[回答] 中国的首都是北京。[资料 1]"

    assert EvidenceAnswerGenerator._clean_answer(raw) == "中国的首都是北京。[资料 1]"


def test_generator_removes_stray_blockquote_markers() -> None:
    assert EvidenceAnswerGenerator._clean_answer(">\n北京。[资料 1]\n>\n>") == "北京。[资料 1]"
    assert EvidenceAnswerGenerator._clean_answer("]\n北京。[资料 1]") == "北京。[资料 1]"


def test_generator_extracts_answer_after_echoed_evidence_block() -> None:
    raw = (
        "[资料 1] 标题：中国首都\n"
        "中国首都 > 古都列表 > 北京\n\n"
        "北京为五朝帝都，中華民國北洋政府和中華人民共和國的首都。\n\n"
        "Assistant: 根据资料，中国的首都是北京。[资料 1]"
    )

    assert EvidenceAnswerGenerator._clean_answer(raw) == "根据资料，中国的首都是北京。[资料 1]"


def test_generator_rejects_protocol_payloads_and_evidence_echo() -> None:
    assert EvidenceAnswerGenerator._clean_answer('{"status":"ok","evidence":[]}') == ""
    assert EvidenceAnswerGenerator._clean_answer("[资料 1] 标题：中国首都\n北京") == ""
    assert EvidenceAnswerGenerator._clean_answer("根据资料回答问题，只回答答案。") == ""


@pytest.mark.asyncio
async def test_generator_uses_evidence_and_truncates_rwkv_continuation() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url == "http://rwkv.test/v1/chat/completions"
        assert payload["contents"][0].find("中华人民共和国的首都是北京。") >= 0
        assert payload["temperature"] == 0.2
        assert payload["top_k"] == 50
        assert payload["top_p"] == 0.6
        assert payload["alpha_presence"] == 1.0
        assert payload["alpha_frequency"] == 0.1
        assert payload["alpha_decay"] == 0.99
        assert payload["stream"] is True
        return httpx.Response(
            200,
            content=(
                'data: {"choices":[{"delta":{"content":"Assistant: <think>推理</think>北京。[资料 1]\\n用户："}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"续写内容"}}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    generator = EvidenceAnswerGenerator(
        Settings(generation_base_url="http://rwkv.test/v1", generation_password="test-password"),
        transport=httpx.MockTransport(handler),
    )

    assert await generator.generate("中国的首都是哪个城市？", [source()]) == "北京。[资料 1]"


@pytest.mark.asyncio
async def test_generator_cleans_echoed_evidence_from_model_output() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                'data: {"choices":[{"delta":{"content":"[资料 1] 标题：中国首都\\n北京为中华人民共和国的首都。\\n\\nAssistant: 中国的首都是北京。[资料 1]"}}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    generator = EvidenceAnswerGenerator(
        Settings(generation_base_url="http://rwkv.test/v1", generation_password="test-password"),
        transport=httpx.MockTransport(handler),
    )

    assert await generator.generate("中国的首都是哪个城市？", [source()]) == "中国的首都是北京。[资料 1]"


@pytest.mark.asyncio
async def test_semantic_generator_does_not_rewrite_model_answer() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                'data: {"choices":[{"delta":{"content":"北京。"}}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    generator = EvidenceAnswerGenerator(
        Settings(
            generation_base_url="http://rwkv.test/v1",
            generation_password="test-password",
            semantic_pipeline_enabled=True,
        ),
        transport=httpx.MockTransport(handler),
    )

    answer = await generator.generate(
        "中国的首都是哪个城市？",
        [source()],
        trusted_evidence=True,
    )

    assert answer == "北京。"


@pytest.mark.asyncio
async def test_generator_skips_model_call_without_sources() -> None:
    generator = EvidenceAnswerGenerator(Settings())

    assert await generator.generate("没有资料怎么办？", []) == "未检索到可用于回答该问题的资料。"


@pytest.mark.asyncio
async def test_generator_skips_model_call_when_evidence_does_not_cover_question() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        pytest.fail("the generation model must not be called for unrelated evidence")

    generator = EvidenceAnswerGenerator(
        Settings(generation_password="test-password"),
        transport=httpx.MockTransport(handler),
    )

    answer = await generator.generate(
        "中国有多少个民族，分别是哪些？",
        [
            SourceItem(
                id="china-1",
                document_id="china",
                source="finewiki-zh",
                title="中国",
                score=1.0,
                snippet="中华人民共和国是位于东亚的国家。",
            )
        ],
    )

    assert answer == "根据检索到的资料，无法确定。"


@pytest.mark.asyncio
async def test_generator_skips_model_call_when_subject_anchor_is_absent() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        pytest.fail("the generation model must not be called for subject-mismatched evidence")

    generator = EvidenceAnswerGenerator(
        Settings(generation_password="test-password"),
        transport=httpx.MockTransport(handler),
    )

    answer = await generator.generate(
        "秦始皇有哪些丰功伟绩？",
        [
            SourceItem(
                id="henan-1",
                document_id="henan",
                source="finewiki-zh",
                title="河南省",
                score=1.0,
                snippet="河南省拥有龙门石窟、殷墟等历史文化遗产。",
            )
        ],
    )

    assert answer == "根据检索到的资料，无法确定。"


@pytest.mark.asyncio
async def test_generator_adds_a_citation_when_model_omits_one() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content='data: {"choices":[{"delta":{"content":"北京。"}}]}\n\ndata: [DONE]\n\n',
        )

    generator = EvidenceAnswerGenerator(
        Settings(generation_password="test-password"),
        transport=httpx.MockTransport(handler),
    )

    assert await generator.generate("中国的首都是哪个城市？", [source()]) == "北京。 [资料 1]"


@pytest.mark.asyncio
async def test_generator_removes_invalid_citation_and_adds_valid_one() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content='data: {"choices":[{"delta":{"content":"北京。[资料 2]"}}]}\n\ndata: [DONE]\n\n',
        )

    generator = EvidenceAnswerGenerator(
        Settings(generation_password="test-password"),
        transport=httpx.MockTransport(handler),
    )

    assert await generator.generate("中国的首都是哪个城市？", [source()]) == "北京。 [资料 1]"


@pytest.mark.asyncio
async def test_generator_reads_current_model_from_models_endpoint() -> None:
    request_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        assert request.url == "http://models.test/v1/models"
        return httpx.Response(200, json={"data": [{"id": "rwkv-current"}]})

    generator = EvidenceAnswerGenerator(
        Settings(generation_models_url="http://models.test/v1/models"),
        transport=httpx.MockTransport(handler),
    )

    assert await generator.current_model() == "rwkv-current"
    assert await generator.current_model() == "rwkv-current"
    assert request_count == 1


@pytest.mark.asyncio
async def test_generator_allows_model_list_failure() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    generator = EvidenceAnswerGenerator(
        Settings(generation_models_url="http://models.test/v1/models"),
        transport=httpx.MockTransport(handler),
    )

    assert await generator.current_model() is None


def test_list_evidence_extracts_heading_rows_and_enumerated_sentence() -> None:
    from llamaindex_retrieval.evidence_utils import list_evidence_answer

    passes = SourceItem(
        id="passes",
        document_id="passes",
        source="wiki",
        title="关隘",
        score=1.0,
        snippet="历史上著名关卡\n函谷关\n潼关\n大散关\n山海关\n嘉峪关\n",
    )
    assert "函谷关" in list_evidence_answer("中国有哪些著名的长城关隘？", [passes])

    achievements = SourceItem(
        id="qin",
        document_id="qin",
        source="wiki",
        title="秦始皇",
        score=1.0,
        snippet="秦始皇一生并天下、称皇帝、废分封、置郡县、征百越、逐匈奴。",
    )
    assert "并天下" in list_evidence_answer("秦始皇有哪些丰功伟绩？", [achievements])

    historical_context = SourceItem(
        id="qin-context",
        document_id="qin",
        source="wiki",
        title="秦始皇",
        score=1.0,
        snippet=(
            "嬴政的功业已经超越三皇五帝，因此向嬴政献上泰皇的尊号。"
        ),
    )
    answer = list_evidence_answer(
        "秦始皇有哪些丰功伟绩？",
        [historical_context, achievements],
    )
    assert answer is not None
    assert "并天下" in answer
    assert "功业已经超越三皇五帝" not in answer
