"""Create a new source manifest with 30 extra train families; V3 stays frozen."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from assemble_curated_sources import ROOT


V3 = ROOT / "llamaindex-retrieval/statetune/retrieval-v3-20260924"
V4 = ROOT / "llamaindex-retrieval/statetune/retrieval-v4-20260924"


def merge():
    old_path, added_path = V3 / "SOURCES-CURATED.json", V4 / "SOURCES-ADDED.json"
    old, added = json.loads(old_path.read_text()), json.loads(added_path.read_text())
    if old["schema"] != "retrieval_curated_sources_v3" or added["schema"] != "retrieval_added_train_sources_v4":
        raise ValueError("source protocol mismatch")
    rows = old["sources"] + added["sources"]
    names, hashes = set(), set()
    for row in rows:
        if row["family"] in names or row["sha256"] in hashes:
            raise ValueError("source family or bytes repeated")
        names.add(row["family"])
        hashes.add(row["sha256"])
        if sha256((ROOT / row["readme_path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"source snapshot changed: {row['repo']}")
    counts = dict(Counter(row["split"] for row in rows))
    if counts["train"] < 150:
        raise ValueError("train source expansion did not reach 150 families")
    return {"schema": "retrieval_curated_sources_v4", "sources": rows,
            "counts": counts, "base_sha256": sha256(old_path.read_bytes()).hexdigest(),
            "added_sha256": sha256(added_path.read_bytes()).hexdigest(),
            "role_review": "author description-level curation; label semantics unreviewed",
            "training_labels_admitted": 0, "training_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = merge()
    with args.out.open("x", encoding="utf-8") as destination:
        json.dump(data, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(json.dumps({"counts": data["counts"],
                      "sha256": sha256(args.out.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
