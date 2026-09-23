"""Choose a mechanically balanced review queue, never an admitted train set."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import unicodedata


def load_rows(paths):
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate candidate IDs")
    return rows


def select(rows, policy):
    reason = Counter()
    accepted = []
    for row in rows:
        question = row["question"]
        kind = row["kind"]
        if kind == "missing" and not any(word in question for word in
                                          policy["missing_question_must_contain_one"]):
            reason["missing_request_not_explicit"] += 1
        elif kind == "conflict" and not any(word in question for word in
                                            policy["conflict_question_must_contain_one"]):
            reason["same_scope_conflict_not_explicit"] += 1
        else:
            accepted.append(row)
    rng = random.Random(policy["seed"])
    rng.shuffle(accepted)
    family_use, kind_use = Counter(), Counter()
    picked, picked_ids = [], set()
    cap = policy["max_family_occurrences"]

    def choose(candidates):
        available = [row for row in candidates if row["id"] not in picked_ids and
                     all(family_use[family] < cap for family in row["source_families"])]
        if not available:
            return False
        row = min(available, key=lambda item: (
            max(family_use[family] for family in item["source_families"]),
            sum(family_use[family] for family in item["source_families"]), rng.random()))
        picked.append(row)
        picked_ids.add(row["id"])
        kind_use[row["kind"]] += 1
        for family in row["source_families"]:
            family_use[family] += 1
        return True

    for kind, target in policy["preferred_kind_counts"].items():
        candidates = [row for row in accepted if row["kind"] == kind]
        for _ in range(target):
            if not choose(candidates):
                break
    while len(picked) < policy["target_rows"] and choose(accepted):
        pass
    if len(picked) != policy["target_rows"]:
        raise ValueError(f"balanced shortlist only has {len(picked)} rows")
    questions = [re.sub(r"\s+", "", unicodedata.normalize("NFKC", row["question"])).casefold()
                 for row in picked]
    return picked, {"lexically_flagged": dict(reason),
                    "eligible_before_balancing": len(accepted),
                    "selected_kinds": dict(kind_use),
                    "selected_source_families": len(family_use),
                    "largest_family_occurrences": max(family_use.values()),
                    "largest_family_fraction": max(family_use.values()) / len(picked),
                    "normalized_question_unique_fraction": len(set(questions)) / len(picked)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, action="append", required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    if policy.get("schema") != "retrieval_plan_mechanical_shortlist_v1":
        raise ValueError("shortlist policy mismatch")
    sources = json.loads(args.sources.read_text())
    if sources.get("schema") != "retrieval_curated_sources_v4":
        raise ValueError("source manifest mismatch")
    source_by_family = {row["family"]: row for row in sources["sources"]}
    rows = load_rows(args.candidates)
    for row in rows:
        if row["split"] != "train" or row["review"].get("accepted") is not False:
            raise ValueError("non-train or reviewed row in candidate input")
        for family, source_hash in zip(row["source_families"], row["source_hashes"], strict=True):
            source = source_by_family[family]
            if source["split"] != "train" or source["sha256"] != source_hash:
                raise ValueError(f"source binding mismatch: {row['id']}")
    picked, summary = select(rows, policy)
    output = {"schema": "retrieval_plan_review_shortlist_v1",
              "selected_ids": [row["id"] for row in picked],
              "selected_rows": len(picked), **summary,
              "candidate_hashes": {str(path): sha256(path.read_bytes()).hexdigest()
                                   for path in args.candidates},
              "source_sha256": sha256(args.sources.read_bytes()).hexdigest(),
              "policy_sha256": sha256(args.policy.read_bytes()).hexdigest(),
              "independent_review": False, "admitted_training_rows": 0,
              "training_started": False}
    with args.out.open("x", encoding="utf-8") as destination:
        json.dump(output, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(json.dumps({k: output[k] for k in ("selected_rows", "lexically_flagged",
                        "selected_kinds", "selected_source_families",
                        "largest_family_fraction", "independent_review")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
