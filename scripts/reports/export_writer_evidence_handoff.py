"""Publish a new paired Writer snapshot without changing any saved model output."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = "writer-evidence-handoff-20260921"


def main():
    run = ROOT / "data/quality-runs" / NAME
    frozen = ROOT / "llamaindex-retrieval/eval" / NAME / "INPUTS.json"
    inputs = json.loads(frozen.read_text())
    review = json.loads((run / "REVIEW.json").read_text())
    summary = json.loads((run / "run1/SUMMARY.json").read_text())
    assert summary["recorded"] == summary["planned"] == 144
    public = ROOT / "llamaindex-retrieval/web/public/experiments"
    upstream_path = public / "typed-funnel-diagnostics-20260921.json"
    upstream = json.loads(upstream_path.read_text())
    cases = []
    for row, old in zip(inputs, upstream["cases"], strict=True):
        assert row["id"] == old["id"]
        answers = []
        for rnd in [1, 2]:
            for arm in ["baseline", "candidate"]:
                record = json.loads((run / "run1" / f"{row['ordinal']:03d}-{arm}-r{rnd}.json").read_text())
                assert record["id"] == row["id"] and record["round"] == rnd and record["arm"] == arm
                trace = record["trace"]
                answers.append({
                    "role": arm,
                    "label": f"第{rnd}轮 · {'原Writer基线' if arm == 'baseline' else '处理状态交接候选'}",
                    "raw_text": record["raw_text"], "finish_reason": record["status"],
                    "elapsed_s": trace["elapsed_ms"] / 1000,
                    "notes": review["cases"][str(row["ordinal"])][arm]["notes"],
                    "evidence_flow": row["flow"],
                    "sources": [{"label": f"资料 {i}", "text": s["snippet"], "url": s.get("uri")}
                                for i, s in enumerate(row["sources"], 1)],
                    "trace": {k: trace[k] for k in ["stage", "status", "prompt", "prompt_sha256", "raw_text_sha256", "parameters", "usage", "finish_reason"]},
                })
        cases.append({"id": row["id"], "question": row["question"], "history": row["history"],
                      "input_sources": old["input_sources"], "answers": answers})
    data = {"title": "Writer处理状态交接：36题双轮原始回答对照", "paired": True,
            "summary": review["summary_zh"], "cases": cases,
            "frozen_inputs_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
            "upstream_snapshot_sha256": hashlib.sha256(upstream_path.read_bytes()).hexdigest()}
    with (public / f"{NAME}.json").open("x") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print({"cases": len(cases), "unchanged_raw_answers": sum(len(c["answers"]) for c in cases)})


if __name__ == "__main__":
    main()
