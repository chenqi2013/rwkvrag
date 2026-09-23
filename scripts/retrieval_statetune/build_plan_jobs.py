"""Deterministically group new repositories for source-separated plan drafts."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random


FOCI = ("ordinary", "comparison", "selection", "history", "missing", "conflict")
SIZES = {"ordinary": 1, "comparison": 3, "selection": 4,
         "history": 2, "missing": 2, "conflict": 3}


def jobs(source_rows, per_train_topic, per_dev_topic, seed=20260923, balanced_v2=False):
    by_topic = defaultdict(list)
    for source in source_rows:
        if source["split"] != "heldout":
            by_topic[(source["split"], source["topic"])].append(source)
    result = []
    rng = random.Random(seed)
    for (split, topic), sources in sorted(by_topic.items(),
                                         key=lambda item: (("train", "dev").index(item[0][0]), item[0][1])):
        count = per_train_topic if split == "train" else per_dev_topic
        if len(sources) < 4:
            raise ValueError(f"topic {topic} has fewer than four sources")
        for index in range(count):
            focus = FOCI[index % len(FOCI)]
            size = min(SIZES[focus], len(sources))
            chosen = rng.sample(sources, size)
            job = {"id": f"{split}-{topic}-{index:03}", "split": split,
                           "topic": topic, "focus": focus,
                           "repos": [source["repo"] for source in chosen],
                           "source_families": [source["family"] for source in chosen],
                           "source_hashes": [source["sha256"] for source in chosen]}
            if balanced_v2:
                job["coverage_mode"] = ("listed" if focus in {"comparison", "selection"}
                                        and (index // len(FOCI)) % 2 else "grid")
                job["required_objects"] = size if focus in {"comparison", "selection"} else None
            result.append(job)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-jobs-per-topic", type=int, default=12)
    parser.add_argument("--dev-jobs-per-topic", type=int, default=3)
    parser.add_argument("--balanced-v2", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.train_jobs_per_topic <= 100 or not 1 <= args.dev_jobs_per_topic <= 30:
        raise ValueError("job count outside bounded range")
    sources = json.loads(args.sources.read_text())
    if sources.get("schema") != "retrieval_github_sources_v1":
        raise ValueError("source protocol mismatch")
    result = jobs(sources["sources"], args.train_jobs_per_topic, args.dev_jobs_per_topic,
                  balanced_v2=args.balanced_v2)
    with args.out.open("x", encoding="utf-8") as output:
        for item in result:
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({"jobs": len(result), "sha256": sha256(args.out.read_bytes()).hexdigest(),
                      "labels_generated": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
