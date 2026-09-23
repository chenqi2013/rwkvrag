"""Extend the curated manifest with new heldout sources; leave V2 untouched."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from assemble_curated_sources import ROOT


V2 = ROOT / "llamaindex-retrieval/statetune/retrieval-v2-20260923"
V3 = ROOT / "llamaindex-retrieval/statetune/retrieval-v3-20260924"
RULES = {
    "uptime_monitors": "Compare service/website uptime checks and alerts; distinguish self-hosted and GitHub Actions deployment.",
    "api_clients": "Compare API request exploration and testing; distinguish graphical workspaces from command-line clients.",
    "terminal_emulators": "Compare terminal-emulator workflows; treat operating-system availability as a condition."
}


def merge():
    old_path, added_path = V2 / "SOURCES-CURATED.json", V3 / "SOURCES-HELDOUT.json"
    old, added = json.loads(old_path.read_text()), json.loads(added_path.read_text())
    if old["schema"] != "retrieval_curated_sources_v2" or added["schema"] != "retrieval_blind_sources_v3":
        raise ValueError("source protocol mismatch")
    rows = old["sources"] + [{**row, "comparison_rule": RULES[row["cohort"]]}
                             for row in added["sources"]]
    names, hashes = set(), set()
    for row in rows:
        if row["family"] in names or row["sha256"] in hashes:
            raise ValueError("family or content hash repeats across splits")
        names.add(row["family"])
        hashes.add(row["sha256"])
        if sha256((ROOT / row["readme_path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"source snapshot changed: {row['repo']}")
    return {"schema": "retrieval_curated_sources_v3", "sources": rows,
            "counts": dict(Counter(row["split"] for row in rows)),
            "rejected_acquisitions": added["rejected"],
            "base_sha256": sha256(old_path.read_bytes()).hexdigest(),
            "added_sha256": sha256(added_path.read_bytes()).hexdigest(),
            "training_labels_admitted": 0, "independent_question_review": False,
            "training_started": False}


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
