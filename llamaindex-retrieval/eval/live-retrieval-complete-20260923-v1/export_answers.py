"""Export only user-visible answers and selected source snippets from a frozen run."""

import argparse
from hashlib import sha256
from html import escape
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CASES = json.loads((HERE / "INPUTS.json").read_text(encoding="utf-8"))["cases"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    lines = ["# 65 道联网题：问题、原始答案与最终来源", "",
             "本文件只抽取本轮 `/v1/ask` 的原始 `answer` 与最终 `sources`，不替它修正文案或引用。模型输出可能错误；来源编号存在不代表语义支持。完整 HTTP 与模型 trace 保留在本机运行目录，逐题原始文件 SHA-256 见同目录 `SUMMARY.json`。", ""]
    for case in CASES:
        path = args.run / f"{case['ordinal']:02d}.ask.json"
        raw = path.read_bytes()
        record = json.loads(raw)
        response = record.get("response") or {}
        generation = response.get("generation") or {}
        sources = response.get("sources") or []
        answer = response.get("answer") or ""
        lines.extend([
            f"## {case['ordinal']:02d} · {escape(case['uid'])}", "",
            f"- 题型：`{case.get('kind') or ('single' if case['source_suite'] == 'live-retrieval-20260923-v2' else 'compare')}`；模式：`{case['retrieval_mode']}`；内部状态：`{generation.get('status', 'none')}`；最终来源：{len(sources)}",
            f"- 原始记录 SHA-256：`{sha256(raw).hexdigest()}`", "",
            "**问题**", "", f"<pre>{escape(case['question'])}</pre>", "",
            "**原始答案**", "", f"<pre>{escape(answer)}</pre>", "",
            "**最终来源**", "",
        ])
        if not sources:
            lines.extend(["无。", ""])
        for number, source in enumerate(sources, 1):
            lines.extend([
                f"{number}. {escape(source.get('title') or '')} · {escape(source.get('uri') or '无网址')} · `{escape((source.get('metadata') or {}).get('content_status') or 'unknown')}`",
                "", f"<pre>{escape(source.get('snippet') or '')}</pre>", "",
            ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
