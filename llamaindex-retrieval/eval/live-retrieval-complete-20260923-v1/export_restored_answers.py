"""Export all restored-index replay answers without rewriting failed output."""

import argparse
from hashlib import sha256
from html import escape
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CASES = json.loads((HERE.parent / "restored-retrieval-v2-20260920" / "cases.json").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    files = [args.run / "calls" / f"{index:04d}.json" for index in range(len(CASES))]
    missing = [path.name for path in files if not path.exists()]
    if missing:
        raise RuntimeError(f"incomplete_run: {len(missing)} records missing")
    lines = ["# 恢复索引 434 题：原始答案和最终来源", "",
             "以下仅抽取本轮原始响应中的问题、答案与最终来源。错误、复读和空回答不修改；`completed` 只是接口记录成功，不能解释为语义正确。逐题文件哈希及运行绑定见同目录 `SUMMARY.json`。", ""]
    for index, (case, path) in enumerate(zip(CASES, files)):
        raw = path.read_bytes()
        record = json.loads(raw)
        try:
            response = json.loads(record.get("raw_response") or "{}")
        except json.JSONDecodeError:
            response = {}
        answer = response.get("answer") or ""
        sources = response.get("sources") or []
        lines.extend([
            f"## {index:04d} · {escape(case['id'])}", "",
            f"- Suite：`{case['suite']}`；HTTP：`{record.get('http_status')}`；生成状态：`{(record.get('diagnostics') or {}).get('generation_status', 'none')}`；最终来源：{len(sources)}",
            f"- 原始记录 SHA-256：`{sha256(raw).hexdigest()}`", "",
            "**问题**", "", f"<pre>{escape(case['payload'].get('question') or '')}</pre>", "",
            "**原始答案**", "", f"<pre>{escape(answer)}</pre>", "",
            "**最终来源**", "",
        ])
        if not sources:
            lines.extend(["无。", ""])
        for number, source in enumerate(sources, 1):
            lines.extend([
                f"{number}. {escape(source.get('title') or '')} · `{escape(source.get('document_id') or '')}`",
                "", f"<pre>{escape((source.get('snippet') or '')[:500])}</pre>", "",
            ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
