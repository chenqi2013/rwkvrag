import importlib.util
from pathlib import Path

BASE = Path(__file__).parents[1] / "reader-label-20260920/protocol.py"
spec = importlib.util.spec_from_file_location("frozen_label_protocol", BASE)
labels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labels)
metrics = labels.metrics


def prompt(case):
    return labels.prompt(case, "neutral-labels")


def parse(raw):
    if not raw.startswith(">"):
        raise ValueError("missing_canonical_fake_think_boundary")
    return labels.parse_label(raw[1:], "neutral-labels")


def parameters(condition):
    return {"temperature": 1.0, "top_k": 1 if condition["arm"] == "top1" else 32,
            "top_p": 1.0 if condition["arm"] == "top1" else .28,
            "presence_penalty": 0., "frequency_penalty": 0., "penalty_decay": .996,
            "seed": condition["seed"], "max_tokens": 32,
            "stop_token_ids": [0], "stop": ["✿", "\nUser:", "\n### User"],
            "skip_special_tokens": False, "return_token_ids": True}
