"""Balanced per-cell verification pairs, with case-disjoint validation and holdout."""
import hashlib
import json
from pathlib import Path

from llamaindex_retrieval.rwkv_pipeline import conversation
from llamaindex_retrieval.rwkvos_batch import render_batch_prompt
from llamaindex_retrieval.schemas import ConversationMessage, SourceItem
from llamaindex_retrieval.state_tokens import Vocabulary, encode_training
from llamaindex_retrieval.task_matrix import cell_review_prompt

HERE = Path(__file__).resolve().parent


def corrupted(case, cell_index, kind):
    lines = case["target"].splitlines()
    row = case["matrix"][cell_index]
    if kind == "missing":
        del lines[cell_index]
        return "\n".join(lines)
    if kind == "citation":
        # Unknown source is only one of the negatives; existing wrong sources
        # are preferred so a verifier cannot pass by checking label range alone.
        wrong = next((i for i, s in enumerate(case["sources"], 1) if s["id"] not in row["source_ids"]), len(case["sources"]) + 1)
        import re
        line = re.sub(r"\[资料 \d+\]", f"[资料 {wrong}]", lines[cell_index])
        if line == lines[cell_index]:
            line = f"{row['object']}的{row['dimension']}：38 L[资料 {wrong}]。"
        lines[cell_index] = line
        return "\n".join(lines)
    records = [r for r in case["records"] if r["eligible"] and r["object"] == row["object"] and row["dimension"] in r["values"]]
    if row["dimension"] == "离线支持":
        value = "不支持离线运行" if records and records[0]["values"][row["dimension"]] == "支持离线运行" else "支持离线运行"
    else:
        original = records[0]["values"][row["dimension"]] if records else "37 L"
        number, unit = original.split()
        value = f"{int(number) + 1} {unit}"
    label = next((i for i, s in enumerate(case["sources"], 1) if s["id"] in row["source_ids"]), 1)
    lines[cell_index] = f"{row['object']}的{row['dimension']}：{value}[资料 {label}]。"
    return "\n".join(lines)


def main():
    output = HERE / "data-review-v2"
    output.mkdir(exist_ok=False)
    vocab = Vocabulary(HERE.parents[1] / "statetune/assets/rwkv_vocab_v20230424.txt")
    pins = {}
    for split in ("train", "validation", "holdout"):
        cases = [json.loads(line) for line in (HERE / f"data-v4/{split}.cases.jsonl").read_text().splitlines()]
        if split == "train":
            cases = [c for i, c in enumerate(cases) if i % 32 < 16]
        rows = []
        for case_index, case in enumerate(cases):
            task = conversation(case["question"], [ConversationMessage(**m) for m in case["history"]])
            sources = [SourceItem(**s) for s in case["sources"]]
            for i, cell in enumerate(case["matrix"]):
                variants = ["correct", ["value", "citation", "missing"][(case_index+i) % 3]] if split == "train" else ["correct", "value", "citation", "missing"]
                for kind in variants:
                    answer = case["target"] if kind == "correct" else corrupted(case, i, kind)
                    prompt = cell_review_prompt(task, cell, answer, sources)
                    rendered, _ = render_batch_prompt([{"role": "user", "content": prompt}], "<think></think", "complete")
                    rendered += "\n"
                    target = json.dumps({"answer": "YES" if kind == "correct" else "NO"})
                    rows.append({"id": f"{case['id']}-c{i}-{kind}", "case_id": case["id"],
                        "stage": "resolver", "kind": kind, "prompt": rendered, "target": target,
                        "prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                        "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
                        **encode_training(rendered, target, vocab, max_tokens=4096)})
        path = output / f"{split}.resolver.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        pins[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows)}
    (output / "MANIFEST.json").write_text(json.dumps({"files": pins, "synthetic": True,
        "split_by_case": True, "independent_review": False, "holdout_used_for_training": False,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))
    print(json.dumps(pins, indent=2))


if __name__ == "__main__":
    main()
