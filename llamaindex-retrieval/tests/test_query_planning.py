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


def test_cause_query_keeps_concrete_event_phrase() -> None:
    plan = build_query_plan("诸葛亮为什么要挥泪斩马谡")

    assert plan.subject == "诸葛亮"
    assert "挥泪斩马谡" in plan.relations
    assert "诸葛亮挥泪斩马谡" in plan.queries


def test_list_plan_extracts_general_requested_relation() -> None:
    plan = build_query_plan("秦始皇有哪些伟大成就？")

    assert plan.analysis.intent == "list"
    assert plan.subject == "秦始皇"
    assert plan.relations == ("伟大成就", "成就")


def test_query_plan_recognizes_colloquial_multi_item_answer_shape() -> None:
    plan = build_query_plan("马斯克创办了哪几家公司")

    assert plan.analysis.intent == "list"
    assert plan.subject == "马斯克"
    assert plan.relations[:4] == ("创办", "创立", "创建", "成立")
    assert plan.answer_shape == "list"
    assert plan.set_semantics == "partial"


def test_generic_possessive_question_becomes_subject_and_field() -> None:
    owner = build_query_plan("宇树科技的老板是谁")
    capital = build_query_plan("中国的首都在哪里")
    ending = build_query_plan("三国演义最后的结局是什么？")

    assert (owner.subject, owner.relations, owner.analysis.intent) == (
        "宇树科技", ("老板",), "agent",
    )
    assert (capital.subject, capital.relations) == ("中国", ("首都",))
    assert ending.subject == "三国演义"
    assert {"结局", "终结", "统一", "归一统"} <= set(ending.relations)

    story_agent = build_query_plan("水浒传里赤手空拳打死老虎的是谁？")
    assert story_agent.subject == "水浒传"
    assert story_agent.analysis.intent == "agent"
    assert {"赤手空拳打死老虎", "打死老虎"} <= set(story_agent.relations)
    assert "打死老虎" in story_agent.queries


def test_compact_role_question_preserves_company_subject() -> None:
    plan = build_query_plan("宇树科技老板是谁？")

    assert plan.subject == "宇树科技"
    assert plan.relations == ("老板",)
    assert plan.analysis.intent == "agent"


def test_nominal_question_without_interrogative_keeps_search_anchor() -> None:
    plan = build_query_plan("中国四大名著？")

    assert plan.subject == "中国四大名著"
    assert plan.answer_shape == "list"
    assert plan.set_semantics == "all"


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


def test_builds_non_current_office_and_founder_queries_without_crashing() -> None:
    vice_president = build_query_plan("美国的副总统是谁？")
    founder = build_query_plan("唐朝的开国皇帝是谁")
    president = build_query_plan("美国的总统是谁")

    assert vice_president.subject == "美国"
    assert vice_president.relations == ("副总统", "副總統")
    assert founder.subject == "唐朝"
    assert founder.relations[0] == "开国皇帝"
    assert president.subject == "美国"
    assert president.relations[0] == "总统"


def test_builds_role_agent_plan_with_role_equivalents() -> None:
    plan = build_query_plan("宇树科技创始人是谁？")

    assert plan.analysis.intent == "agent"
    assert plan.subject == "宇树科技"
    assert plan.relations[:3] == ("创始人", "创办人", "创办者")


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


def test_builds_counted_enumeration_as_complete_list() -> None:
    plan = build_query_plan("中国四大名著是哪四个？")

    assert plan.analysis.intent == "list"
    assert plan.subject == "中国四大名著"
    assert plan.answer_shape == "list"
    assert plan.set_semantics == "all"
    assert plan.relations == ("是指", "包括", "分别是", "分别为")
    assert plan.queries[:2] == ("中国四大名著 列表", "中国四大名著")

    generic_plan = build_query_plan("中国四大名著是哪几个？")
    assert generic_plan.analysis.intent == "list"
    assert generic_plan.answer_shape == "list"
    assert generic_plan.set_semantics == "all"
    assert generic_plan.relations == ("是指", "包括", "分别是", "分别为")
    assert generic_plan.queries[:2] == ("中国四大名著 列表", "中国四大名著")

    historical_plan = build_query_plan("中国历史上的四大名著是哪四个？")
    assert historical_plan.subject == "四大名著"
    assert historical_plan.queries[:2] == ("四大名著 列表", "四大名著")


def test_builds_role_attribution_from_question_grammar() -> None:
    author = build_query_plan("西游记是哪个作者写的？")
    director = build_query_plan("流浪地球是哪位导演拍的？")

    assert author.analysis.intent == "agent"
    assert author.analysis.entity_type == "person"
    assert author.subject == "西游记"
    assert author.relations == ("作者", "写")
    assert author.context_policy == "lead_append"
    assert director.subject == "流浪地球"
    assert director.relations == ("导演", "拍")


def test_outcome_question_requires_summary_answer() -> None:
    plan = build_query_plan("三国演义最后的结局是什么？")

    assert plan.answer_shape == "summary"


def test_terminal_and_passive_agent_questions_build_relation_contracts() -> None:
    memorial = build_query_plan("端午节的由来，是为了纪念谁？")
    victim = build_query_plan("岳飞是被谁害死的")

    assert memorial.analysis.intent == "agent"
    assert memorial.subject == "端午节"
    assert {"纪念", "由来"} <= set(memorial.relations)
    assert victim.analysis.intent == "agent"
    assert victim.subject == "岳飞"
    assert {"害死", "杀害"} <= set(victim.relations)


def test_transit_list_plan_cleans_actions_and_adds_companion_page_query() -> None:
    plan = build_query_plan("深圳地铁一号线都经过哪些车站？")

    assert plan.analysis.intent == "list"
    assert plan.subject == "深圳地铁1号线"
    assert "深圳地铁1号线" in plan.queries
    assert "深圳地铁车站列表" in plan.queries
    assert "深圳地铁车站列表 1号线" in plan.queries

    alias_plan = build_query_plan("罗宝线沿途停靠哪些站？")
    assert alias_plan.subject == "罗宝线"
    assert alias_plan.queries[0] == "罗宝线 站列表"


def test_list_question_with_earliest_word_is_not_misclassified_as_ordinal() -> None:
    plan = build_query_plan("深圳最早开通的地铁线路经过哪些站？")

    assert plan.analysis.intent == "list"
    assert plan.subject == "深圳最早开通的地铁线路"
    assert "深圳地铁车站列表" in plan.queries


def test_endpoint_description_adds_focused_route_query() -> None:
    plan = build_query_plan("连接罗湖和机场东的深圳地铁线路有哪些车站？")

    assert plan.queries[0] == "罗湖 机场东"
    assert plan.queries[1] == "由罗湖站至机场东站"
    assert "深圳地铁 罗湖 机场东 车站" in plan.queries


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
