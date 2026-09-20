"""Separate evidence assessment and missing-evidence follow-up StateTune datasets."""
import hashlib
import json
from pathlib import Path

from llamaindex_retrieval.rwkv_pipeline import conversation
from llamaindex_retrieval.rwkvos_batch import render_batch_prompt
from llamaindex_retrieval.schemas import ConversationMessage, SourceItem
from llamaindex_retrieval.state_tokens import Vocabulary, encode_training
from llamaindex_retrieval.task_matrix import Cell, assessment_prompt, followup_prompt

HERE = Path(__file__).resolve().parent


def main():
    vocab = Vocabulary(HERE.parents[1] / "statetune/assets/rwkv_vocab_v20230424.txt")
    for role in ("assessment", "followup"):
        output = HERE / f"data-{role}-v2"
        output.mkdir(exist_ok=False)
        pins = {}
        for split in ("train", "validation", "holdout"):
            cases = [json.loads(line) for line in (HERE / f"data-v4/{split}.cases.jsonl").read_text().splitlines()]
            rows = []
            for case in cases:
                examples = []
                sources = [SourceItem(**s) for s in case["sources"]]
                if role == "assessment":
                    for cell in case["matrix"]:
                        selected = [s for s in sources if s.id in cell["source_ids"]]
                        examples.append((assessment_prompt(Cell(**{k: cell[k] for k in ("id", "object", "dimension", "question")}), selected),
                            {"status": cell["status"], "source_ids": cell["source_ids"]}))
                else:
                    task = conversation(case["question"], [ConversationMessage(**m) for m in case["history"]])
                    unresolved = [dict(c) for c in case["matrix"] if c["status"] != "supported"]
                    for c in unresolved:
                        if c["assessment_status"] == "completed":
                            c["assessment_source_ids"] = c["source_ids"]
                    previous = [{"cell_id": c["id"], "query": c["question"]} for c in case["matrix"]]
                    queries = [{"cell_id": c["id"], "query": f"{c['object']} {c['dimension']} 正式说明 原文"} for c in unresolved]
                    examples.append((followup_prompt(task, unresolved, previous), {"stop": not bool(queries), "queries": queries}))
                for i, (prompt, target) in enumerate(examples):
                    rendered, _ = render_batch_prompt([{"role": "user", "content": prompt}], "<think></think", "complete")
                    target = json.dumps(target, ensure_ascii=False, sort_keys=True)
                    rows.append({"id": f"{case['id']}-{role}-{i}", "case_id": case["id"], "stage": "planner",
                        "prompt": rendered, "target": target,
                        "prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                        "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
                        **encode_training(rendered, target, vocab, max_tokens=4096)})
            path = output / f"{split}.planner.jsonl"
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            pins[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows)}
        (output / "MANIFEST.json").write_text(json.dumps({"files": pins, "role": role,
            "synthetic": True, "independent_review": False, "holdout_used_for_training": False,
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))
        print(role, json.dumps(pins))


if __name__ == "__main__":
    main()
