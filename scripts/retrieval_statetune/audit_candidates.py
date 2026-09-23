"""Fail-closed provenance and diversity audit for a new retrieval StateTune batch.

This does not certify semantic labels or export training tokens. It accepts only
explicitly reviewed rows and never reads historic answers as training targets.
"""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import unicodedata

ROLES = {"plan", "evidence", "followup"}
SPLITS = {"train", "dev", "heldout"}
KINDS = {"ordinary", "comparison", "selection", "missing", "conflict", "history"}
REQUIRED = {"id", "split", "role", "kind", "question", "prompt", "target",
            "prompt_sha256", "target_sha256", "source_families", "source_hashes", "review"}


def normalize(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()


def digest(value):
    return sha256(value.encode("utf-8")).hexdigest()


def read_rows(path):
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON: {error}") from error
    return rows


def audit(rows, exclusions, minimums):
    """Return a report; any listed issue blocks export or an optimizer step."""
    issues = []
    ids = set()
    by_family = defaultdict(set)
    by_source = defaultdict(set)
    by_question = defaultdict(set)
    question_frequency = Counter()
    role_counts = Counter()
    kind_counts = Counter()
    train_families = Counter()
    exact_questions = set()
    excluded_questions = set(exclusions.get("question_hashes", []))
    excluded_families = set(exclusions.get("source_families", []))
    excluded_sources = set(exclusions.get("source_hashes", []))
    if not all(isinstance(x, str) and x for values in (
            excluded_questions, excluded_families, excluded_sources) for x in values):
        issues.append("invalid exclusion registry")
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not REQUIRED <= row.keys():
            issues.append(f"row {index}: required fields missing")
            continue
        identity = row["id"]
        if not isinstance(identity, str) or not identity or identity in ids:
            issues.append(f"row {index}: empty or duplicate id")
            continue
        ids.add(identity)
        split, role, kind = row["split"], row["role"], row["kind"]
        if split not in SPLITS or role not in ROLES or kind not in KINDS:
            issues.append(f"{identity}: invalid split, role or kind")
            continue
        fields = ("question", "prompt", "target")
        if any(not isinstance(row[name], str) or not row[name].strip() for name in fields):
            issues.append(f"{identity}: empty question, prompt or target")
            continue
        if digest(row["prompt"]) != row["prompt_sha256"] or digest(row["target"]) != row["target_sha256"]:
            issues.append(f"{identity}: prompt/target hash mismatch")
        try:
            json.loads(row["target"])
        except json.JSONDecodeError:
            issues.append(f"{identity}: target is not JSON")
        families, sources = row["source_families"], row["source_hashes"]
        if (not isinstance(families, list) or not families or
                any(not isinstance(x, str) or not x for x in families) or len(set(families)) != len(families)):
            issues.append(f"{identity}: invalid source families")
            continue
        if (not isinstance(sources, list) or any(not isinstance(x, str) or
                not re.fullmatch(r"[0-9a-f]{64}", x) for x in sources) or len(set(sources)) != len(sources)):
            issues.append(f"{identity}: invalid source hashes")
            continue
        if role == "evidence" and not sources:
            issues.append(f"{identity}: evidence judgment has no source snapshot")
        if role == "plan" and len(sources) != len(families):
            issues.append(f"{identity}: plan source families/hashes are not aligned")
        review = row["review"]
        if (not isinstance(review, dict) or review.get("accepted") is not True or
                review.get("independent_of_author") is not True or
                not isinstance(review.get("author"), str) or not review["author"].strip() or
                not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip() or
                review.get("reviewer") == review.get("author") or not review.get("reason")):
            issues.append(f"{identity}: independent review not recorded")
        normalized = normalize(row["question"])
        if not normalized:
            issues.append(f"{identity}: empty normalized question")
            continue
        qhash = digest(normalized)
        question_frequency[qhash] += 1
        if qhash in excluded_questions or any(f in excluded_families for f in families) or any(
                s in excluded_sources for s in sources):
            issues.append(f"{identity}: overlaps excluded evaluation material")
        by_question[qhash].add(split)
        exact_questions.add(qhash)
        for family in families:
            by_family[family].add(split)
            if split == "train":
                train_families[family] += 1
        for source in sources:
            by_source[source].add(split)
        if split == "train":
            role_counts[role] += 1
            kind_counts[kind] += 1
    for name, groups in (("question", by_question), ("source family", by_family), ("source hash", by_source)):
        mixed = [key for key, splits in groups.items() if len(splits) > 1]
        if mixed:
            issues.append(f"{name} crosses train/dev/heldout: {len(mixed)} identities")
    over_repeated = sum(1 for count in question_frequency.values() if count > minimums.get("max_question_repetitions", 3))
    if over_repeated:
        issues.append(f"normalized questions repeated too often: {over_repeated}")
    train = sum(role_counts.values())
    if train < minimums.get("train", 0):
        issues.append(f"train rows {train} below minimum {minimums['train']}")
    for role, minimum in minimums.get("roles", {}).items():
        if role_counts[role] < minimum:
            issues.append(f"{role} rows {role_counts[role]} below minimum {minimum}")
    for kind, minimum in minimums.get("kinds", {}).items():
        if kind_counts[kind] < minimum:
            issues.append(f"{kind} rows {kind_counts[kind]} below minimum {minimum}")
    family_minimum = minimums.get("train_families", 0)
    if len(train_families) < family_minimum:
        issues.append(f"train families {len(train_families)} below minimum {family_minimum}")
    if train:
        largest = max(train_families.values(), default=0) / train
        if largest > minimums.get("max_family_fraction", 1):
            issues.append("largest source family exceeds train share limit")
    else:
        largest = 0
    unique_fraction = len(exact_questions) / len(rows) if rows else 0
    if unique_fraction < minimums.get("min_unique_question_fraction", 0):
        issues.append("normalized question uniqueness below minimum")
    return {"admitted": not issues, "rows": len(rows), "train_rows": train,
            "train_roles": dict(role_counts), "train_kinds": dict(kind_counts),
            "train_families": len(train_families), "largest_train_family_fraction": largest,
            "unique_question_fraction": unique_fraction, "issues": issues,
            "limits": "Integrity and isolation checks only; semantic labels and runtime prompt alignment need separate review."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--minimums", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(read_rows(args.rows), json.loads(args.exclusions.read_text()),
                   json.loads(args.minimums.read_text()))
    body = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        with args.out.open("x", encoding="utf-8") as destination:
            destination.write(body)
    print(body, end="")
    if not report["admitted"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
