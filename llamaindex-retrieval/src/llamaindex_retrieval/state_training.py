"""Train-only token adapter. CPU validation; no model or evaluation-file imports."""

import hashlib
import json
from pathlib import Path
import random
import re


def read_training_tokens(path, expected_sha256, expected_count, *, max_tokens=4096):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("training export SHA mismatch")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows or len(rows) != expected_count:
        raise ValueError("training export count mismatch")
    seen = set()
    for row in rows:
        identity = row["id"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("empty or duplicate training identity")
        seen.add(identity)
        ids, labels, n = row["input_ids"], row["labels"], row["prompt_tokens"]
        if (not isinstance(ids, list) or not isinstance(labels, list)
                or not 2 < len(ids) <= max_tokens or len(ids) != len(labels)
                or type(n) is not int or not 0 < n < len(ids) - 1
                or any(type(x) is not int for x in ids + labels)
                or ids[-1] != 0 or any(not 0 < x < 65536 for x in ids[:-1])
                or labels != [-100] * n + ids[n:]
                or type(row["target_tokens_with_eos"]) is not int
                or row["target_tokens_with_eos"] != len(ids) - n):
            raise ValueError("invalid full-sequence tokens, prompt mask, boundary or EOS")
        if not isinstance(row["prompt_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["prompt_sha256"]):
            raise ValueError("invalid prompt hash")
    return rows


def accumulation_groups(rows, *, seed, size=2):
    """One visit per sample, with no class resampling or discarded final group."""
    if type(size) is not int or size < 1:
        raise ValueError("accumulation size must be positive")
    ordered = sorted(rows, key=lambda row: row["id"])
    random.Random(seed).shuffle(ordered)
    return [ordered[i:i + size] for i in range(0, len(ordered), size)]


def group_mean(losses):
    """Mean of per-example target/EOS losses, including a partial final group."""
    if not losses:
        raise ValueError("empty accumulation group")
    return sum(losses) / len(losses)


def longest_training_row(rows):
    return min(rows, key=lambda row: (-len(row["input_ids"]), row["id"]))
