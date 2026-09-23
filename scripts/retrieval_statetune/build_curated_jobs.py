"""Build source-separated, balanced label-first draft jobs from V2 cohorts."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random


TRAIN_FOCI = ("ordinary", "comparison", "selection", "history", "missing", "conflict")
BLIND_FOCI = ("ordinary", "comparison", "selection", "missing", "conflict",
              "ordinary", "comparison", "selection", "comparison", "selection",
              "ordinary", "missing", "conflict", "comparison", "selection", "ordinary")
SIZES = {"ordinary": 1, "comparison": 3, "selection": 4,
         "history": 2, "missing": 2, "conflict": 3}


def build(sources, split, count, seed):
    groups = defaultdict(list)
    for source in sources:
        if source["split"] == split:
            groups[source["cohort"]].append(source)
    minimum = 3 if split == "heldout" else 4
    if not groups or any(len(rows) < minimum for rows in groups.values()):
        raise ValueError(f"each selected cohort needs at least {minimum} sources")
    rng = random.Random(seed)
    usage = Counter()
    jobs = []
    if split == "heldout":
        schedule = [(sorted(groups)[index % len(groups)], focus)
                    for index, focus in enumerate(BLIND_FOCI)]
    else:
        focus_list = TRAIN_FOCI if split == "train" else ("ordinary", "comparison", "selection")
        schedule = [(cohort, focus_list[index % len(focus_list)])
                    for cohort in sorted(groups) for index in range(count)]
    serial = Counter()
    for cohort, focus in schedule:
        rows = groups[cohort]
        size = min(SIZES[focus], len(rows))
        tie = {row["family"]: rng.random() for row in rows}
        chosen = sorted(rows, key=lambda row: (usage[row["family"]], tie[row["family"]]))[:size]
        rng.shuffle(chosen)
        for row in chosen:
            usage[row["family"]] += 1
        index = serial[cohort]
        serial[cohort] += 1
        jobs.append({"id": f"{split}-{cohort}-{index:03}", "split": split,
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
    parser.add_argument("--split", choices=("train", "dev", "heldout"), required=True)
    parser.add_argument("--jobs-per-cohort", type=int, default=18)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.jobs_per_cohort <= 50:
        raise ValueError("job count outside bounds")
    source_bytes = args.sources.read_bytes()
    manifest = json.loads(source_bytes)
    if manifest.get("schema") != "retrieval_curated_sources_v2":
        raise ValueError("source schema mismatch")
    jobs = build(manifest["sources"], args.split, args.jobs_per_cohort, 20260924)
    with args.out.open("x", encoding="utf-8") as destination:
        for job in jobs:
            destination.write(json.dumps(job, ensure_ascii=False) + "\n")
    print(json.dumps({"jobs": len(jobs), "planned_rows": len(jobs) * 8,
                      "split": args.split, "source_sha256": sha256(source_bytes).hexdigest(),
                      "jobs_sha256": sha256(args.out.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
