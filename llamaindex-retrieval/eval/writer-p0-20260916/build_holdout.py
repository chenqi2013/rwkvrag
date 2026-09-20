"""Author-controlled synthetic holdout, frozen before either prompt is evaluated.

These are new controlled materials, not an external blind benchmark. Gold is
specified from source text, never inferred from a model's response.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def case(number, question, texts, facts, *, history=(), decision="answer"):
    materials = [{"id": f"h{number}-s{i}", "document_id": f"h{number}-d{i}",
                  "source": "synthetic-holdout-20260916", "title": title, "uri": None,
                  "score": 1.0, "snippet": text, "metadata": {"synthetic": True}}
                 for i, (title, text) in enumerate(texts, 1)]
    expected = []
    for i, (claim, material_number, quote) in enumerate(facts, 1):
        material = materials[material_number - 1]
        start = material["snippet"].index(quote)
        expected.append({"id": f"F{i}", "claim": claim, "supporting_quotes": [{
            "material_id": material["id"], "start": start, "end": start + len(quote),
            "quote": quote}]})
    return {"id": f"smoke_{number:03}", "phase": "frozen_holdout", "question": question,
            "materials": materials, "history": list(history), "expected_facts": expected,
            "expected_decision": decision,
            "oracle_notes": ["按冻结原文与完整历史逐题复核；每项事实和引用均须有据。",
                             "标题不能替代正文；资料不足不是数值为零；冲突未获裁定不能单选。"],
            "provenance": {"kind": "new_synthetic_author_controlled_holdout",
                           "materials_sha256": sha256(encode(materials)).hexdigest()}}


def build():
    a = "穗川 M4 的额定转速为 730 rpm。"
    b = "雨桥 N2 的冷却方式为自然散热，不安装风扇。"
    c = "石湾 K8 的巡检周期为每 19 天一次。"
    d = "青屿 P3 的故障记录使用 UTC 时间。"
    rows = [case(1, "分别核对穗川M4的额定转速、雨桥N2的冷却方式、石湾K8的巡检周期、青屿P3故障记录的时区；四项都要原文依据。", [
        ("四款设备参数目录", "本页仅列穗川M4、雨桥N2、石湾K8、青屿P3的名称，不包含参数。"),
        ("青屿P3记录说明", d), ("石湾K8维护说明", c), ("雨桥N2热设计", b),
        ("穗川M4转速说明", a)], [(a, 5, a), (b, 4, b), (c, 3, c), (d, 2, d)])]
    history = [
        {"role": "user", "content": "先比较澄岚2.0服务器版和桌面版的连接上限与价格。"},
        {"role": "assistant", "content": "需要确认版本与字段。"},
        {"role": "user", "content": "更正，是澄岚2.1桌面版，服务器版和价格都撤掉。保留并发连接上限，再加是否支持离线和开发代号，三项都要。"},
        {"role": "assistant", "content": "当前范围是2.1桌面版的三项。"},
        {"role": "user", "content": "讲义封面用了旧服务器截图，图片我会换，不代表又改回服务器版。图上画了20台终端只是示意，不能作为连接上限。"},
        {"role": "assistant", "content": "截图不作为参数证据。"},
        {"role": "user", "content": "发布日期不加，网络协议列表也不要；之前确认的三项保留。"},
    ]
    text = "澄岚 2.1 桌面版最多允许 7 个并发连接；支持离线运行；开发代号为 Juniper。"
    rows.append(case(2, "把刚定下的三项写好。", [
        ("澄岚2.0桌面版", "澄岚2.0桌面版最多4个并发连接，不支持离线，开发代号为 Maple。"),
        ("澄岚2.1服务器版", "澄岚2.1服务器版最多70个并发连接，开发代号为 Cedar。"),
        ("澄岚2.1桌面版", text)], [
        ("澄岚2.1桌面版最多7个并发连接。", 3, text),
        ("澄岚2.1桌面版支持离线运行。", 3, text),
        ("澄岚2.1桌面版开发代号为Juniper。", 3, text)], history=history))
    rows.append(case(3, "仅根据材料，给出虚构探头苇沙T9在发布首季的实际交付数量（台）；无依据请说明无法确定。",
                     [], [], decision="abstain"))
    text = "岚石 U5 的额定功率为 46 W，冷却方式为自然散热。"
    rows.append(case(4, "分别回答岚石U5的额定功率、冷却方式和去年全球销量；有资料的回答，未记载的明确说明。", [
        ("岚石U5全球销量", "这是销量报告目录，正文数据尚未收录。"),
        ("岚石U5规格", text)], [("额定功率46 W。", 2, text),
                               ("冷却方式自然散热。", 2, text)], decision="partial_abstain"))
    a = "版本号同为 3.0 的甲记录写道：柏津 S4 的储水容量是 9 L。"
    b = "版本号同为 3.0 的乙记录写道：柏津 S4 的储水容量是 12 L。"
    rows.append(case(5, "柏津S4 3.0的储水容量到底是多少？两份记录没有权威排序或更正声明，若无法确定，分别列出记载并说明冲突。", [
        ("甲记录", a), ("乙记录", b)], [("甲记录记载9 L。", 1, a),
                                          ("乙记录记载12 L。", 2, b)], decision="partial_abstain"))
    table_rows = ["| A | 2021 | 17 L/min | 不支持 |", "| A | 2024 | 11 L/min | 支持 |",
                  "| B | 2024 | 5 kg/min | 不支持 |"]
    table = "| 型号 | 年份 | 额定流量 | 离线 |\n| --- | --- | --- | --- |\n" + "\n".join(table_rows)
    rows.append(case(6, "逐项列出A款2021年、A款2024年、B款2024年的额定流量与离线支持，保留原单位，不换算。", [
        ("停用草案", "此草案已停用，不应用于正式参数：A款2024流量曾暂填99 L/min。"),
        ("正式参数表", table)], [
        ("A款2021流量17 L/min。", 2, table_rows[0]), ("A款2021不支持离线。", 2, table_rows[0]),
        ("A款2024流量11 L/min。", 2, table_rows[1]), ("A款2024支持离线。", 2, table_rows[1]),
        ("B款2024流量5 kg/min。", 2, table_rows[2]), ("B款2024不支持离线。", 2, table_rows[2])]))
    final = "最终核准：按住复位键 13 秒；校验码为 NIMBUS-731。"
    long_text = "旧版草案：按住7秒，校验码OLD-008；该段已废止。\n" + "\n".join(
        f"维护记录第{i:03}项：外壳、接线和封条已检查；本项不规定复位时长或校验码。" for i in range(280)
    ) + "\n" + final
    rows.append(case(7, "根据长记录最后的最终核准段，复位键应按住几秒，校验码是什么？逐项引用原文。", [
        ("维护长记录", long_text)], [("按住复位键13秒。", 1, final),
                                      ("校验码NIMBUS-731。", 1, final)]))
    text = "苇沙 T9 首月实际交付数量为 0 台。尚未进入量产，试验样机不计入交付。"
    rows.append(case(8, "根据正式记录，苇沙T9首月实际交付多少台，是否已经进入量产？不要把预约量当交付量。", [
        ("预约登记", "苇沙T9预约数量为100台，预约不等于交付。"),
        ("正式首月交付记录", text)], [("首月实际交付0台。", 2, text),
                                          ("尚未进入量产。", 2, text)]))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("holdout.jsonl"))
    args = parser.parse_args()
    with args.output.open("xb") as file:
        file.write(b"".join(encode(row) + b"\n" for row in build()))
