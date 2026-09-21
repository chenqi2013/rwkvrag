"""Publish immutable answer text and sources; never infer semantic accuracy."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "llamaindex-retrieval/web/public/experiments"
PREVIEW = ROOT / "data/frontend-model-comparison-20260921/dist/experiments"

def publish(name, data):
    raw = json.dumps(data, ensure_ascii=False, indent=2)
    for directory in [PUBLIC, PREVIEW]:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / (name + ".json")).write_text(raw)

def main():
    original = json.loads((PUBLIC / "model-size-paired-20260921.json").read_text())
    root = ROOT / "data/quality-runs/writer-convergence-20260921/run2"
    cases = []; count = 0; lengths = {}; controlled_reviews = {}
    for review_path in (ROOT / "data/quality-runs/writer-convergence-20260921/review1").glob("*-round[12].json"):
        controlled_reviews[review_path.stem] = {r["ordinal"]: r for r in json.loads(review_path.read_text())}
    for c in original["cases"]:
        answers = []
        for arm in ["baseline", "decision"]:
            for rnd in [1, 2]:
                f = root / f"{arm}-round{rnd}" / f"{c['ordinal']:04d}.json"
                if not f.exists(): continue
                row = json.loads(f.read_text()); text = row.get("raw_text", "")
                if text: assert hashlib.sha256(text.encode()).hexdigest() == row["raw_text_sha256"]
                key = f"{arm} 第{rnd}轮"; lengths[key] = lengths.get(key,0) + (row.get("finish_reason")=="length")
                answers.append({"label":key,"raw_text":text,"finish_reason":row.get("finish_reason",row["status"]),
                    "elapsed_s":row["elapsed_ms"]/1000,"notes":controlled_reviews.get(f"{arm}-round{rnd}",{}).get(c["ordinal"],{}).get("notes","触顶和复读须分开判断；本条语义待审读。所有输出原文保留。"),
                    "sources":c["evidence"],"trace":{"request":row["request"],"usage":row.get("usage"),"repetition_flags":row.get("repetition"),"raw_text_sha256":row.get("raw_text_sha256")}})
                count += 1
        cases.append({"id":c["id"],"question":c["question"],"answers":answers})
    publish("writer-convergence-20260921",{"title":"7.2B Writer指令受控对照", "summary":f"已记录{count}/752次。固定材料、零State，唯一变化为追加任务完成指令。触及2048输出上限：{lengths}。候选未晋级；普通题退步原样保留。", "cases":cases})
    root = ROOT / "data/quality-runs/live-comparison-20260921/run1"
    planned = json.loads((ROOT / "llamaindex-retrieval/eval/live-comparison-20260921/CASES.json").read_text())
    review_file = ROOT / "data/quality-runs/live-comparison-20260921/REVIEW.json"
    reviews = json.loads(review_file.read_text()) if review_file.exists() else {}
    cases = []; done = 0
    for ordinal, c in enumerate(planned):
        f = root / f"{ordinal:02d}.json"; answers = []
        if f.exists():
            done += 1; row = json.loads(f.read_text()); response = row.get("response",{}); gen=response.get("generation",{})
            text = response.get("answer", "")
            if gen.get("raw_model_answer") is not None: assert text == gen["raw_model_answer"]
            answers = [{"label":"7.2B零State · 真实联网候选诊断", "raw_text":text,
                "finish_reason":gen.get("provider_finish_reason") or gen.get("status") or row["status"], "elapsed_s":row["elapsed_s"],
                "notes":reviews.get(c["id"],{}).get("notes", "语义待审读；请求完成不等于正确。") if row["status"]=="recorded" else row.get("error", "请求失败"),
                "sources":[{"label":f"资料 {i}","text":s["snippet"],"url":s.get("uri")} for i,s in enumerate(response.get("sources",[]),1)],
                "trace":{"retrieval":response.get("retrieval"),"generation":gen,"http_status":row.get("http_status")}}]
        cases.append({"id":c["id"],"question":c["question"],"answers":answers})
    publish("live-comparison-20260921",{"title":"12道真实联网及知识库联合比较", "summary":f"已记录{done}/12题；10题web、2题hybrid。实时SearchReader/Tavily搜索，模型自行规划、选证据、写回答。候选诊断，不代表正式服务已修复；实现者审读，非独立盲测。", "cases":cases})
    root = ROOT / "data/quality-runs/github-project-comparison-20260921/run1"
    planned = json.loads((ROOT / "llamaindex-retrieval/eval/github-project-comparison-20260921/CASES.json").read_text())
    review_file = ROOT / "data/quality-runs/github-project-comparison-20260921/REVIEW.json"
    reviews = json.loads(review_file.read_text()) if review_file.exists() else {}
    cases = []; main_done = 0
    for ordinal, c in enumerate(planned):
        f = root / f"{ordinal:02d}.json"; answers = []
        if f.exists():
            main_done += 1; row=json.loads(f.read_text()); response=row.get("response",{});gen=response.get("generation",{})
            text=response.get("answer", "")
            if gen.get("raw_model_answer") is not None: assert text == gen["raw_model_answer"]
            answers=[{"label":"大型项目比较 · 7.2B零State · 候选诊断", "raw_text":text,
                "finish_reason":gen.get("provider_finish_reason") or gen.get("status") or row["status"],"elapsed_s":row["elapsed_s"],
                "notes":reviews.get(c["id"],{}).get("notes","语义待审读；执行成功不等于比较完整。") if row["status"]=="recorded" else row.get("error","请求失败"),
                "sources":[{"label":f"资料 {i}","text":v["snippet"],"url":v.get("uri")} for i,v in enumerate(response.get("sources",[]),1)],
                "trace":{"request":row["request"],"retrieval":response.get("retrieval"),"generation":gen,"http_status":row.get("http_status")}}]
        cases.append({"id":c["id"],"question":c["question"],"answers":answers})
    publish("github-project-comparison-20260921",{"title":"大型 GitHub 多项目比较 · 主验收", "summary":f"已记录{main_done}/8次问答：6个首轮场景比较4～5项目，2个真实多轮问题扩至6项目。重点检查覆盖、相对优势、适用场景、引用和条件遵循。实时联网候选诊断，非独立盲测，不代表商业验收。", "cases":cases})
    root = ROOT / "data/quality-runs/github-project-comparison-paced-20260921/run1"
    review_file = root.parent / "REVIEW.json"
    reviews = json.loads(review_file.read_text()) if review_file.exists() else {}
    cases = []; finished = 0; statuses = {}
    for ordinal, c in enumerate(planned):
        f = root / f"{ordinal:02d}.json"; answers = []
        if f.exists():
            finished += 1; row = json.loads(f.read_text()); response = row.get("response", {}); gen = response.get("generation", {})
            text = response.get("answer", "")
            if gen.get("raw_model_answer") is not None: assert text == gen["raw_model_answer"]
            status = gen.get("status") or row["status"]; statuses[status] = statuses.get(status, 0) + 1
            answers = [{"label":"Tavily实时检索 · 单凭据低频 · 7.2B零State", "raw_text":text,
                "finish_reason":status,"elapsed_s":row["elapsed_s"],
                "notes":reviews.get(c["id"],{}).get("notes","语义待审读；联网返回不等于比较结论可靠。"),
                "sources":[{"label":f"资料 {i}","text":v["snippet"],"url":v.get("uri")} for i,v in enumerate(response.get("sources",[]),1)],
                "trace":{"request":row["request"],"retrieval":response.get("retrieval"),"generation":gen,"http_status":row.get("http_status")}}]
        cases.append({"id":c["id"],"question":c["question"],"answers":answers})
    publish("github-project-comparison-paced-20260921",{"title":"大型 GitHub 多项目比较 · 低频实时检索", "summary":f"已记录{finished}/8次：{statuses}。单凭据串行检索，间隔至少6秒，首个提供方异常后熔断。6个首轮与2个真实历史追问；7.2B零State候选诊断，非正式部署或商业验收。", "cases":cases})
    print(f"Published {count} controlled records and {done} real retrieval records")

if __name__ == "__main__":main()
