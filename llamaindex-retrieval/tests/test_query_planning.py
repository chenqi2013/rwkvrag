from llamaindex_retrieval.query_planning import build_query_plan


def test_builds_general_cause_plan_with_subject_and_relation_queries() -> None:
    plan = build_query_plan("明朝是因为什么原因走上了灭亡？")

    assert plan.analysis.intent == "cause"
    assert plan.subject == "明朝"
    assert "灭亡" in plan.relations
    assert plan.context_policy == "section"
    assert plan.queries[0] == "明朝灭亡"
    assert any(query.startswith("明朝 ") for query in plan.queries)
    assert plan.original_question in plan.queries


def test_builds_ordinal_plan_with_equivalent_queries() -> None:
    plan = build_query_plan("中国历史上第一个皇帝是谁？")

    assert plan.analysis.intent == "ordinal"
    assert plan.subject == "中国"
    assert plan.queries == (
        "中国 第一个 皇帝",
        "中国 第一位 皇帝",
        "中国 首位 皇帝",
        "中国历史上第一个皇帝是谁？",
    )
    assert plan.merge_strategy == "rank_fusion"

    reverse = build_query_plan("谁是中国第一位皇帝？")
    assert reverse.analysis.intent == "ordinal"
    assert reverse.subject == "中国"
    assert reverse.queries[:3] == plan.queries[:3]

    bare = build_query_plan("中国历史上第一个皇帝")
    assert bare.analysis.intent == "ordinal"
    assert bare.subject == "中国"
    assert bare.queries[:3] == plan.queries[:3]


def test_cause_question_with_list_wording_still_uses_cause_pipeline() -> None:
    plan = build_query_plan("明朝走向灭亡的主要原因有哪些？")

    assert plan.analysis.intent == "cause"
    assert plan.subject == "明朝"
    assert plan.context_policy == "section"


def test_builds_definition_and_procedure_plans_without_specific_entities() -> None:
    definition = build_query_plan("硝酸是什么？")
    procedure = build_query_plan("宇航员如何进行舱外活动训练？")

    assert definition.subject == "硝酸"
    assert definition.relations == ("简介", "定义")
    assert definition.context_policy == "lead"
    assert procedure.subject == "宇航员 进行舱外活动训练"
    assert procedure.context_policy == "section"


def test_cleans_colloquial_cause_subject() -> None:
    plan = build_query_plan("明朝究竟为何一步步走向覆亡了？")

    assert plan.analysis.intent == "cause"
    assert plan.subject == "明朝"
    assert "灭亡" in plan.relations


def test_builds_action_agent_and_current_office_queries() -> None:
    silk_road = build_query_plan("中国历史上开启丝绸之路的是谁？")
    current_office = build_query_plan("现在美国副总统是谁？")

    assert silk_road.analysis.intent == "agent"
    assert silk_road.subject == "丝绸之路"
    assert silk_road.queries[:4] == (
        "丝绸之路 开启",
        "丝绸之路 开辟",
        "丝绸之路 出使",
        "丝绸之路 凿空",
    )
    assert current_office.analysis.intent == "agent"
    assert current_office.subject == "美国副总统"
    assert current_office.queries[:3] == (
        "美国副总统 现任",
        "美国副总统 目前",
        "美国副总统 当前",
    )


def test_agent_query_focuses_on_nearest_relational_object() -> None:
    plan = build_query_plan("现代主义建筑发展所依赖的安全电梯由谁发明？")

    assert plan.analysis.intent == "agent"
    assert plan.subject == "安全电梯"
    assert plan.queries[:2] == ("安全电梯 发明", "安全电梯")
    assert plan.normalized_question in plan.queries

    reverse_object = build_query_plan("奥的斯发明了什么？")
    assert reverse_object.analysis.intent == "agent"
    assert reverse_object.subject == "奥的斯"
    assert reverse_object.queries[:2] == ("奥的斯 发明", "奥的斯")


def test_builds_definition_subject_without_copula() -> None:
    plan = build_query_plan("请简要介绍山科友里")

    assert plan.subject == "山科友里"
    assert plan.queries[0] == "山科友里 简介 定义"


def test_decomposes_coordinated_time_question() -> None:
    plan = build_query_plan("香港和澳门分别是哪一年回归的？")

    assert plan.analysis.intent == "time"
    assert plan.queries[:4] == ("香港", "香港回归", "澳门", "澳门回归")
    assert plan.merge_strategy == "document_interleave"


def test_cleans_list_and_location_subjects() -> None:
    host = build_query_plan("2030年世界杯由哪些国家主办？")
    venue = build_query_plan("2026年世界杯决赛在哪个球场举行？")

    assert host.subject == "2030年世界杯"
    assert venue.subject == "2026年世界杯决赛"
    assert build_query_plan("中国有哪些著名的长城关隘？").subject == "长城关隘"


def test_cleans_common_conversational_question_shells() -> None:
    assert build_query_plan("请简要介绍一下山西博物院。").subject == "山西博物院"
    assert build_query_plan("我不太了解古茗，它指的是什么？").subject == "古茗"
    assert build_query_plan("你知道風景畫是干什么的吗？").subject == "风景画"
    assert build_query_plan("想去天通苑北站，它的位置在哪里？").subject == "天通苑北站"
    assert build_query_plan("请问北海银行什么时候成立？").subject == "北海银行"
    assert build_query_plan("说下乌克兰总统办公室大楼背后的设计者。").subject == "乌克兰总统办公室大楼"
    assert build_query_plan("你知道管风琴教堂由谁设计吗？").subject == "管风琴教堂"
    assert build_query_plan("虞国究竟是怎么一步步灭亡的？").subject == "虞国"
    assert build_query_plan("从资料看，萨珊王朝灭亡是哪些因素造成的？").subject == "萨珊王朝"
    assert build_query_plan("导致俄羅斯解體的原因有哪些？").subject == "俄罗斯"
    assert build_query_plan("能通俗说说李運騰吗？").subject == "李运腾"
    assert build_query_plan("请概括徐国灭亡的主要原因和过程。").subject == "徐国"
    assert build_query_plan("请问罗家桥站在什么地方？").subject == "罗家桥站"
    assert build_query_plan("说下阿方索六世出生的年份。").subject == "阿方索六世"
    assert build_query_plan("说下瓦森人口的具体数字。").subject == "瓦森"
    assert build_query_plan("你知道下安特亞蒙的人口数据吗？").subject == "下安特亚蒙"
