"""New presentation snapshot; the source dataset and all answers remain unchanged."""
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llamaindex-retrieval/src"))
from llamaindex_retrieval.evidence_flow import evidence_flow

source = ROOT / "llamaindex-retrieval/web/public/experiments/typed-funnel-20260921.json"
raw = source.read_bytes()
data = json.loads(raw)
data["title"] = "最近实验 v10：回答与故障定位"
data["summary"] = "固定材料回放，未重新联网；本次更新的是诊断与展示，没有重新生成答案。默认展示最近完成的v10，仍有语义错误与复读，未部署正式服务。历史v4/v8可展开，所有原始回答可下载。"
data["presentation_protocol"] = "evidence-diagnostics-v1"
data["source_dataset_sha256"] = hashlib.sha256(raw).hexdigest()
labels = ["原漏斗v4 · 历史固定材料", "Typed漏斗v8 · 实际应用配置", "整合修复v10 · 实际应用配置"]
for case in data["cases"]:
    assert [a["label"] for a in case["answers"]] == labels
    for i, answer in enumerate(case["answers"]):
        answer["role"] = "candidate" if i == 2 else "historical"
        answer["evidence_flow"] = evidence_flow(answer["funnel"]) if answer.get("funnel") else None
path = source.with_name("typed-funnel-diagnostics-20260921.json")
with path.open("x") as stream:
    json.dump(data, stream, ensure_ascii=False, indent=2)
assert source.read_bytes() == raw
print({"cases":len(data["cases"]), "unchanged_answers":sum(len(c["answers"]) for c in data["cases"]), "source_sha256":data["source_dataset_sha256"]})
