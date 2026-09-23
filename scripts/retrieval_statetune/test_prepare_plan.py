"""The exported plan target must match the experimental runtime contract."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_plan import VOCAB, compile_row
from llamaindex_retrieval.state_tokens import Vocabulary


def test_compiled_plan_masks_prompt_and_preserves_full_three_by_three_target():
    vocab = Vocabulary(VOCAB)
    case = {"id": "example-1", "split": "train", "kind": "comparison",
            "question": "比较 A、B、C 的部署、硬件和优化。", "history": [],
            "source_families": ["repo:new/example"], "source_hashes": [],
            "review": {"accepted": True, "author": "teacher", "reviewer": "reviewer", "reason": "checked"},
            "plan": {"coverage": "grid", "objects": ["A", "B", "C"],
                     "dimensions": ["部署", "硬件", "优化"], "conditions": [], "listed_pairs": [],
                     "initial_queries": [{"object": obj, "query": f"{obj} 部署 硬件 优化 文档"}
                                         for obj in "ABC"]}}
    row = compile_row(case, vocab)
    assert row["role"] == "plan"
    assert row["input_ids"][-1] == 0
    assert row["labels"][:row["prompt_tokens"]] == [-100] * row["prompt_tokens"]
    assert row["labels"][row["prompt_tokens"]:] == row["input_ids"][row["prompt_tokens"]:]
    assert row["target"].count('"query"') == 3
    assert len(row["input_ids"]) <= 8192
