"""Synthetic comparison development cases with source-bound gold, never a blind benchmark."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fixtures", HERE.parent / "writer-p0-20260916/build_holdout.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def build():
    inputs = [
        ("比较甲型和乙型的容量、离线支持，两项都要分别引用。", [
            ("乙型", "乙型容量为 12 L，不支持离线。"),
            ("甲型", "甲型容量为 8 L，支持离线。")], "answer"),
        ("比较白榆1.0和2.0的并发上限和加密支持，不要混用版本。", [
            ("白榆2.0", "白榆2.0并发上限为 18，支持加密。"),
            ("白榆1.0", "白榆1.0并发上限为 6，不支持加密。")], "answer"),
        ("对比甲站与乙站的故障次数和停机时长；未记载的项目请说明。", [
            ("甲站", "甲站故障次数为 0 次，停机时长为 0 小时。"),
            ("乙站", "乙站故障次数为 2 次；停机时长未记载。")], "partial_abstain"),
        ("比较水泵A和粉料泵B的流量，保留原单位，并说明能否直接判断谁更大。", [
            ("粉料泵B", "粉料泵B额定流量为 5 kg/min。资料未给出物料密度。"),
            ("水泵A", "水泵A额定流量为 11 L/min。")], "partial_abstain"),
        ("比较团队实施日期与产品公开发布日期，不能把它们当成同一天。", [
            ("内部计划", "团队计划于 2026 年 10 月 15 日升级。"),
            ("发布记录", "产品公开发布日期为 2024 年 10 月 7 日。")], "answer"),
        ("比较甲设备和乙设备的核准功率，忽略已经废止的数值。", [
            ("旧草案", "甲设备功率曾暂填 99 W；此草案已经废止。"),
            ("最终核准", "最终核准：甲设备功率为 16 W，乙设备功率为 21 W。")], "answer"),
        ("对比南仓和北仓的面积。南仓两份记录没有权威排序，不要擅自选一个值。", [
            ("南仓记录一", "南仓面积为 90 平方米。"),
            ("北仓", "北仓面积为 120 平方米。"),
            ("南仓记录二", "南仓面积为 95 平方米。")], "partial_abstain"),
        ("对比星舟桌面版和服务器版的价格与离线能力；无材料时不要猜测。", [], "abstain"),
    ]
    rows = []
    for number, (question, texts, decision) in enumerate(inputs, 1):
        facts = [(text, i, text) for i, (_, text) in enumerate(texts, 1)
                 if not (number == 6 and i == 1)]
        row = fixtures.case(number, question, texts, facts, decision=decision)
        row["phase"] = "development_smoke_not_blind"
        row["provenance"]["kind"] = "synthetic_comparison_development"
        rows.append(row)
    return rows


if __name__ == "__main__":
    path = HERE / "development.jsonl"
    with path.open("x") as output:
        for row in build():
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
