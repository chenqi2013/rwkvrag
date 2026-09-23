"""Compile reviewed retrieval-plan labels to exact batch prompts and masked tokens.

Rows are drafts until the admission audit, source isolation and independent
semantic review pass. This script never starts training or edits frozen data.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from llamaindex_retrieval.retrieval_plan import RetrievalPlanV1, training_prompt
from llamaindex_retrieval.schemas import ConversationMessage
from llamaindex_retrieval.state_tokens import Vocabulary, encode_training


ROOT = Path(__file__).resolve().parents[2]
VOCAB = ROOT / "llamaindex-retrieval/statetune/assets/rwkv_vocab_v20230424.txt"


def compile_row(case, vocab, max_tokens=8192):
    required = {"id", "split", "kind", "question", "history", "plan", "source_families",
                "source_hashes", "review"}
    if not isinstance(case, dict) or not required <= case.keys():
        raise ValueError("annotated plan case missing required fields")
    history = [ConversationMessage.model_validate(item, strict=True) for item in case["history"]]
    plan = RetrievalPlanV1.model_validate(case["plan"], strict=True)
    prompt = training_prompt(case["question"], history)
    target = json.dumps(plan.model_dump(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    encoded = encode_training(prompt, target, vocab, max_tokens)
    return {"id": case["id"], "split": case["split"], "role": "plan", "kind": case["kind"],
            "question": case["question"], "source_families": case["source_families"],
            "source_hashes": case["source_hashes"], "review": case["review"],
            "prompt": prompt, "target": target,
            "prompt_sha256": sha256(prompt.encode()).hexdigest(),
            "target_sha256": sha256(target.encode()).hexdigest(),
            "plan_protocol": "retrieval-plan-v1", "prompt_protocol": "rwkvos_batch_complete_no_final_lf_v1",
            **encoded}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, default=VOCAB)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()
    vocab = Vocabulary(args.vocab)
    rows = [compile_row(json.loads(line), vocab, args.max_tokens)
            for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("no cases or duplicate IDs")
    with args.out.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "sha256": sha256(args.out.read_bytes()).hexdigest(),
                      "training_started": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
