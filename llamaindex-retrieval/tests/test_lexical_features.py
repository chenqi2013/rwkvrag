from llamaindex_retrieval.lexical_features import extract_document_aliases


def test_extracts_explicit_and_parenthetical_aliases() -> None:
    aliases = extract_document_aliases(
        "深圳地铁1号线（罗宝线）",
        "深圳地铁1号线，又称罗宝线，是深圳最早建成的地铁线路。",
        {"aliases": ["地铁一号线"]},
    )

    assert aliases == ["地铁一号线", "深圳地铁1号线", "罗宝线"]


def test_ignores_long_descriptive_alias_fragments() -> None:
    aliases = extract_document_aliases(
        "测试项目",
        "测试项目又称一种用于演示复杂系统运行过程的综合性测试项目。",
    )

    assert aliases == []


def test_extracts_former_name_from_finewiki_lead() -> None:
    aliases = extract_document_aliases(
        "深圳地铁1号线",
        "深圳地铁1号线，曾称罗宝线，是中国广东省深圳市的一条地铁路线。",
    )

    assert aliases == ["罗宝线"]


def test_parenthetical_station_name_does_not_capture_following_description() -> None:
    aliases = extract_document_aliases(
        "深圳地铁2号线",
        "线路以蛇口港站（原名蛇口客运港站）为起点，至世界之窗。",
    )

    assert aliases == ["蛇口客运港站"]
