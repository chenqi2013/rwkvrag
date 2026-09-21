"""Publish typed-funnel originals; full HTTP/token traces remain in the archive."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "llamaindex-retrieval/web/public/experiments"


def answer(label, response, notes):
    graph = response.get("retrieval", {}).get("funnel")
    generation = response.get("generation", {})
    return {"label": label, "raw_text": response.get("answer") or "",
        "finish_reason": generation.get("status"), "elapsed_s": generation.get("elapsed_ms", 0) / 1000,
        "notes": notes, "sources": [{"label": f"资料 {i}", "text": source["snippet"], "url": source.get("uri")}
            for i, source in enumerate(response.get("sources", []), 1)], "funnel": graph,
        "trace": {"model_calls": [{key: call[key] for key in
            ("stage", "purpose", "call_id", "status", "raw_text", "parsed_output", "parse_error", "prompt_sha256") if key in call}
            for call in generation.get("model_calls", [])],
            "detail": "本页保留原始输出和节点判断；完整prompt、HTTP及token记录在GitHub归档。"}}


def correction(name, ordinal):
    path = ROOT / "data/quality-runs" / name / "REVIEW-CORRECTIONS.json"
    if not path.exists():
        return ""
    note = json.loads(path.read_text()).get("cases", {}).get(str(ordinal), {}).get("notes", "")
    return " 追加复核：" + note if note else ""


def main():
    inputs = json.loads((ROOT / "llamaindex-retrieval/eval/typed-funnel-integration-20260921/INPUTS.json").read_text())
    review_path = ROOT / "data/quality-runs/typed-funnel-integration-20260921/REVIEW.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    cases = []
    for i, row in enumerate(inputs):
        new = ROOT / "data/quality-runs/typed-funnel-integration-20260921/run1" / f"{i:03d}.json"
        if not new.exists():
            raise RuntimeError("Refusing to publish an incomplete 36-question run")
        previous = ROOT / "data/quality-runs/funnel-structured-v4-20260921/run1" / f"{i:03d}.json"
        note = review.get("cases", {}).get(str(i), {}).get("notes", "语义审读尚未汇总，不以执行完成代表正确。")
        latest = ROOT / "data/quality-runs/typed-funnel-v10-integration-20260921/run1" / f"{i:03d}.json"
        if not latest.exists():
            raise RuntimeError("Refusing to publish an incomplete v10 run")
        latest_review_path = ROOT / "data/quality-runs/typed-funnel-v10-integration-20260921/REVIEW.json"
        latest_review = json.loads(latest_review_path.read_text()) if latest_review_path.exists() else {}
        latest_note = latest_review.get("cases", {}).get(str(i), {}).get("notes", "尚未完成语义审读，执行状态不代表质量。")
        note += correction("typed-funnel-integration-20260921", i)
        latest_note += correction("typed-funnel-v10-integration-20260921", i)
        cases.append({"id": row["id"], "question": row["question"], "history": row.get("history", []),
            "input_sources": [{"label": source["id"], "text": source["snippet"], "url": source.get("uri")} for source in row["sources"]], "answers": [
            answer("原漏斗v4 · 历史固定材料", json.loads(previous.read_text())["response"], "原始失败保留，未修改。"),
            answer("Typed漏斗v8 · 实际应用配置", json.loads(new.read_text())["response"], note),
            answer("整合修复v10 · 实际应用配置", json.loads(latest.read_text())["response"], latest_note)]})
    dataset = {"title": "Typed漏斗修复 · 36题完整回放",
        "summary": "7.2B零State、同一批固定材料，非新联网。13条原子节点通过不代表整链通过。展示原始回答、来源、硬条件及候选资格；正式服务未切换。",
        "cases": cases}
    path = PUBLIC / "typed-funnel-20260921.json"
    path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2))
    print({"cases": len(cases), "answers": 3 * len(cases), "bytes": path.stat().st_size})


if __name__ == "__main__":
    main()
