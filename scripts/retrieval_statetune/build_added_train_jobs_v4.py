"""Register extra train jobs that always include a newly acquired source."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random


FOCI = ("ordinary", "comparison", "selection", "ordinary", "missing", "conflict", "history")
SIZES = {"ordinary": 1, "comparison": 3, "selection": 4,
         "history": 2, "missing": 2, "conflict": 3}


def build(all_sources, additions, jobs_per_cohort=12):
    groups = defaultdict(list)
    added = defaultdict(list)
    for row in all_sources:
        if row["split"] == "train":
            groups[row["cohort"]].append(row)
    for row in additions:
        if row["split"] != "train":
            raise ValueError("non-train source in addition manifest")
        added[row["cohort"]].append(row)
    rng = random.Random(20260924)
    usage = Counter()
    jobs = []
    for cohort in sorted(added):
        sources = groups[cohort]
        new_sources = added[cohort]
        if len(sources) < 4 or not new_sources:
            raise ValueError("cohort lacks comparable sources or additions")
        for index in range(jobs_per_cohort):
            focus = FOCI[index % len(FOCI)]
            size = SIZES[focus]
            tie = {row["family"]: rng.random() for row in sources}
            anchor = min(new_sources, key=lambda row: (usage[row["family"]], tie[row["family"]]))
            rest = [row for row in sources if row["family"] != anchor["family"]]
            rest.sort(key=lambda row: (usage[row["family"]], tie[row["family"]]))
            chosen = [anchor, *rest[:size - 1]]
            rng.shuffle(chosen)
            for row in chosen:
                usage[row["family"]] += 1
            jobs.append({"id": f"train-v4-{cohort}-{index:03}", "split": "train",
                         "topic": cohort, "focus": focus,
                         "role": chosen[0]["role"],
                         "comparison_rule": chosen[0]["comparison_rule"],
                         "coverage_mode": ("listed" if focus in {"comparison", "selection"}
                                           and index % 2 else "grid"),
                         "required_objects": size if focus in {"comparison", "selection"} else None,
                         "repos": [row["repo"] for row in chosen],
                         "source_families": [row["family"] for row in chosen],
                         "source_hashes": [row["sha256"] for row in chosen]})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--added", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.sources.read_bytes()
    sources = json.loads(raw)
    additions = json.loads(args.added.read_text())
    if (sources.get("schema") != "retrieval_curated_sources_v4" or
            additions.get("schema") != "retrieval_added_train_sources_v4"):
        raise ValueError("source protocol mismatch")
    jobs = build(sources["sources"], additions["sources"])
    with args.out.open("x", encoding="utf-8") as destination:
        for job in jobs:
            destination.write(json.dumps(job, ensure_ascii=False) + "\n")
    print(json.dumps({"jobs": len(jobs), "possible_rows": len(jobs) * 8,
                      "source_sha256": sha256(raw).hexdigest(),
                      "jobs_sha256": sha256(args.out.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
