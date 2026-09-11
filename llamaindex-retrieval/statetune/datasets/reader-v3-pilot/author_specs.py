"""Explicit source-reviewed author decisions; no model outputs or label inference.

Run from llamaindex-retrieval. This dated dataset artifact is not a second CLI.
The production statetune build/audit/export commands consume its annotations.
"""
from copy import deepcopy
import json
from pathlib import Path

from llamaindex_retrieval.statetune import digest, write_json, write_rows


def case(question, supports, *, history=None, tags=()):
    return {"question": question, "supports": supports,
            "history": history or [], "coverage_tags": list(tags)}


SPECS = {
    "3238689": [
        case("VL22型机车是以谁的名字命名的？", {"E1": ["以弗拉基米尔·列宁为这种机车命名"]}),
        case("停产时，诺沃切尔卡斯克电力机车厂一共制造了多少台VL22M型机车？", {
            "E2": ["至1958年停产，诺沃切尔卡斯克电力机车厂共生产了1541台VL22M型电力机车"]}),
        case("更正，只说明所有VL22I型机车在哪一年报废。", {
            "E3": ["所有VL22I型机车在1980年报废"]},
             history=[{"role": "user", "content": "先查VL22M生产数量。"}], tags=["history_correction"]),
        case("VL22M型机车牵引电动机的平均无故障运行里程是多少？", {}),
        case("现在改查VL22I单台改造费用，金额是多少卢布？", {},
             history=[{"role": "user", "content": "请查VL22I的报废年份。"}], tags=["history_correction"]),
        case("诺沃切尔卡斯克厂制造了多少台VL22M？VL22I单台改造费用又是多少卢布？", {
            "E2": ["至1958年停产，诺沃切尔卡斯克电力机车厂共生产了1541台VL22M型电力机车"]},
             tags=["partial_support"]),
    ],
    "8024544": [
        case("GagaOOLala名称中间的OO取自哪种语言的哪个词义？", {
            "E1": ['中間的"OO"則是取自法語的"或是"（ou）的發音。']}),
        case("GagaOOLala表中的《同愛一家》是哪一天上架的？", {
            "E2": ["《同愛一家》                  | 紀錄片   | 2020年5月20日"]}),
        case("改为同时查《同愛一家》和《日光樹影》的上架日期。", {
            "E2": ["《同愛一家》                  | 紀錄片   | 2020年5月20日"],
            "E3": ["《日光樹影》                  | 短片     | 2021年12月17日"]},
             history=[{"role": "user", "content": "先解释平台名称的含义。"}], tags=["history_correction", "multi_unit"]),
        case("《酷蓋爸爸》第2季总共拍摄了多少天？", {}),
        case("更正，不查上架日期，只查GagaOOLala在2021年的订阅收入总额。", {},
             history=[{"role": "user", "content": "请查《同愛一家》的上架日期。"}], tags=["history_correction"]),
    ],
    "6124888": [
        case("资料的地质背景指出，日本列岛处于哪四个板块交界处？", {
            "E1": ["日本列岛地处欧亚大陆板块、北美洲板块、太平洋板块和菲律宾板块四个板块的交界处"]}),
        case("2018年岛根地震后，大田市的50处避难所共接纳了至少多少人？", {
            "E2": ["大田市的50处避难所共接纳了至少171人。"]}),
        case("不再问避难所人数，改查岛根县为受损住宅修复补助划拨了多少预算。", {
            "E3": ["岛根县政府决定自从该县2018年一般会计补充预算中划拨2亿日元，以对受损住宅修复加以补助。"]},
             history=[{"role": "user", "content": "先查大田市避难所接纳人数。"}], tags=["history_correction"]),
        case("这次地震给石见银山造成的财产损失总额具体是多少日元？", {}),
        case("现在只查住宅修复补助每户最多能领多少日元，不查总预算。", {},
             history=[{"role": "user", "content": "请查住宅修复补助总预算。"}], tags=["history_correction", "aggregate_is_not_per_household"]),
    ],
    "5452404": [
        case("本届香港高级组银牌若法定时间赛和，接下来如何决出胜负？", {
            "E1": ["若雙方於法定時間九十分鐘內賽和，將會進行加時賽上、下半場各十五分鐘賽事，再未能分出勝負會以互射十二碼來決勝。"]}),
        case("SS02第一圈比赛，九巴元朗对和富大埔的比分是多少？", {
            "E2": ["2016年9月18日 SS02 | 九巴元朗 | 0–1  | 和富大埔"]}),
        case("更正，查SS10决赛傑志对東方龍獅的比分和比赛地点。", {
            "E3": ["2017年1月15日 SS10 | 傑志                    | 2–1  | 東方龍獅       | 香港大球場"]},
             history=[{"role": "user", "content": "先查SS02比赛比分。"}], tags=["history_correction"]),
        case("SS01理文流浪对香港飛馬一战，两队控球率各是多少？", {}),
        case("更正，不问决赛比分，只问傑志主教练当年的月薪。", {},
             history=[{"role": "user", "content": "查SS10决赛比分。"}], tags=["history_correction"]),
    ],
    "1279264": [
        case("资料列出的兰州理工大学校训是什么？", {"E1": ["校训：奋进求是"]}),
        case("甘肃工业大学1971年开始招收工农兵学员时，首年共招收多少人、学制几年？", {
            "E2": ["1971年，学校开始招收工农兵学员，在机械制造、铸造、焊接、水力机械、液压传动、化工机械、工业与民用建筑等7个专业共招收学生382名，学制为3年。"]}),
        case("更正，查原技术工程学院转设后的校名，以及由哪家公司全资持有。", {
            "E3": ["2021年2月2日，经中华人民共和国教育部批准，原属兰州理工大学管理的独立学院兰州理工大学技术工程学院脱离该校管理，并转设为民办普通高等学校兰州信息科技学院，由北京爱因生教育投资有限责任公司全资持有。"]},
             history=[{"role": "user", "content": "先查1971年工农兵学员人数。"}], tags=["history_correction"]),
        case("兰州理工大学温州研究生分院首届博士生录取了多少人？", {}),
        case("改查兰州理工大学2017年科研经费总额，不问学院转设。", {},
             history=[{"role": "user", "content": "技术工程学院转设后的校名是什么？"}], tags=["history_correction"]),
    ],
    "5706573": [
        case("1988年1月21日，宝山县和吴淞区撤销后合并建立了哪个区？", {
            "E1": ["1月21日——宝山县和吴淞区撤销，合并建立宝山区。"]}),
        case("1988年上海首次无偿献血周结束时，有多少人义务献血？", {
            "E2": ["5月11日——上海首次无偿献血周结束，1754人义务献血。"]}),
        case("改为同时查首次无偿献血周的献血人数，以及沪嘉高速全线通车日期。", {
            "E2": ["5月11日——上海首次无偿献血周结束，1754人义务献血。"],
            "E3": ["10月31日——沪嘉高速公路（上海交通路杨家桥-嘉定）建成并全线通车"]},
             history=[{"role": "user", "content": "先查宝山区成立。"}], tags=["history_correction", "multi_unit"]),
        case("沪嘉高速公路通车首日实际通过了多少辆车？", {}, tags=["capacity_is_not_actual_traffic"]),
        case("更正，只查1988年上海大众桑塔纳轿车的单辆售价。", {},
             history=[{"role": "user", "content": "请查桑塔纳生产流水线每天装配多少辆。"}], tags=["history_correction"]),
        case("沪嘉高速何时全线通车？通车首日实际通过了多少辆车？", {
            "E3": ["10月31日——沪嘉高速公路（上海交通路杨家桥-嘉定）建成并全线通车"]},
             tags=["partial_support", "capacity_is_not_actual_traffic"]),
    ],
    "1618999": [
        case("资料介绍的亚洲风剧场每周哪几天、哪个时段播映电视剧？", {
            "E1": ["亚洲风剧场，是星空卫视周一至周五晚19：00～21：00播映电视剧的时段。"]}),
        case("亚洲风剧场2011年《梦想高飞》的播出起止日期是什么？", {
            "E2": ["《梦想高飞》（2011年8月3日-2011年8月11日）"]}),
        case("改查2014年《秘密花園》的重播起止日期。", {
            "E3": ["《秘密花園》（重播）（2014年7月25日-8月6日）"]},
             history=[{"role": "user", "content": "先查2011年《梦想高飞》的播出日期。"}], tags=["history_correction", "rebroadcast_scope"]),
        case("亚洲风剧场2011年播出《梦想高飞》时的平均收视率是多少？", {}),
        case("更正，只查亚洲风剧场购买《想你》播映权花了多少钱。", {},
             history=[{"role": "user", "content": "先查《想你》的播出日期。"}], tags=["history_correction"]),
    ],
    "4397431": [
        case("白鹭引擎是基于哪种编程语言开发的？", {
            "E1": ["白鹭引擎是一个基于TypeScript语言开发的HTML5游戏引擎。"]}),
        case("资料记载，截至2017年Egret Runtime累计接入了多少台设备？", {
            "E2": ["截止到2017年，Runtime已累计接入设备5亿台。"]}),
        case("改为同时查Runtime截至2017年的接入设备数，以及Egret Native通过哪个SDK支持一键生成渠道包。", {
            "E2": ["截止到2017年，Runtime已累计接入设备5亿台。"],
            "E3": ["Egret Native通过对QuickSDK的支持，可使你一键生成所有渠道包"]},
             history=[{"role": "user", "content": "先查白鹭引擎使用的编程语言。"}], tags=["history_correction", "multi_unit"]),
        case("Egret Wing在2017年的标准版售价是多少？", {}),
        case("更正，只查Texture Merger每秒能处理多少张图片的实测吞吐量。", {},
             history=[{"role": "user", "content": "先介绍Texture Merger的功能。"}], tags=["history_correction"]),
        case("Egret Native通过哪个SDK支持一键生成渠道包？这个SDK当年的授权费是多少？", {
            "E3": ["Egret Native通过对QuickSDK的支持，可使你一键生成所有渠道包"]},
             tags=["partial_support"]),
    ],
}


def main():
    here = Path(__file__).resolve().parent
    drafts = [json.loads(line) for line in (here / "source-drafts.jsonl").read_text().splitlines()]
    annotations, sources = [], []
    for split in ("train", "dev", "heldout"):
        for draft in drafts:
            if draft["split"] != split:
                continue
            page = str(draft["source"]["page_id"])
            if page not in SPECS:
                continue
            sources.append({"id": draft["id"], "split": split,
                            "sha256": digest(draft["source"]["snippet"])})
            for i, spec in enumerate(SPECS[page], 1):
                row = deepcopy(draft)
                row.update(id=f"reader_v3_{page}_{i:02d}", question=spec["question"],
                           active_tasks=[spec["question"]], history=spec["history"],
                           reviewed=True, reviewer="Codex author self-review 2026-09-09; not independent",
                           coverage_tags=spec["coverage_tags"])
                text = row["source"]["snippet"]
                row["unit_judgments"] = []
                for unit in row["units"]:
                    quotes = spec["supports"].get(unit["id"], [])
                    evidence = []
                    for quote in quotes:
                        start = text.find(quote, unit["start"], unit["end"])
                        if start < 0:
                            raise ValueError(f"author quote not found: {row['id']} {unit['id']} {quote}")
                        evidence.append({"quote": quote, "quote_sha256": digest(quote),
                                         "source_span": {"start": start, "end": start + len(quote)}})
                    row["unit_judgments"].append({"unit_id": unit["id"], "unit_sha256": unit["sha256"],
                        "supported": bool(quotes), "evidence": evidence,
                        "reason": (("作者核对完整原文后确认，引文直接支持当前问题中的所求事实；不要求单元覆盖全部子问。"
                                    if quotes else "作者核对完整原文及历史更正后确认，本单元没有直接给出当前问题所求事实；主题或名称相近不计证据。")
                                   + "当前问题：" + spec["question"])})
                annotations.append(row)
    assert len(sources) == 8 and len(annotations) == 43
    write_rows(here / "annotations.jsonl", annotations)
    write_json(here / "SOURCE-BINDINGS.json", sources)
    write_json(here / "AUTHOR-SPECS.json", SPECS)


if __name__ == "__main__":
    main()
