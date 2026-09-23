"""Merge pinned batches and apply a transparent source-role rejection ledger."""

import argparse
from collections import Counter
from hashlib import sha1, sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "llamaindex-retrieval/statetune/retrieval-v1-20260923"
V2 = ROOT / "llamaindex-retrieval/statetune/retrieval-v2-20260923"


def assemble():
    batch_paths = [V2 / "SOURCES.json", V2 / "SOURCES-SUPPLEMENT.json"]
    batches = [json.loads(path.read_text()) for path in batch_paths]
    policy_file = V2 / "ROLE-POLICY.json"
    policy = json.loads(policy_file.read_text())
    exclusions = json.loads((V1 / "EXCLUSIONS.json").read_text())
    forbidden_families = {x.casefold() for x in exclusions["source_families"]}
    forbidden_hashes = set(exclusions["source_hashes"])
    rows, rejected, names, hashes = [], [], set(), set()
    for batch in batches:
        for source in batch["sources"]:
            name = source["repo"].casefold()
            if name in names or source["family"] != f"repo:{name}":
                raise ValueError(f"duplicate or mismatched source family: {name}")
            names.add(name)
            raw = (ROOT / source["readme_path"]).read_bytes()
            blob = sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            if sha256(raw).hexdigest() != source["sha256"] or blob != source["blob_sha"]:
                raise ValueError(f"source snapshot changed: {name}")
            if source["family"] in forbidden_families or source["sha256"] in forbidden_hashes:
                raise ValueError(f"evaluation source leaked: {name}")
            if source["sha256"] in hashes:
                raise ValueError(f"duplicated README across source families: {name}")
            hashes.add(source["sha256"])
            if source["cohort"] not in policy["cohorts"]:
                raise ValueError(f"unregistered cohort: {source['cohort']}")
            if name in {key.casefold() for key in policy["reject"]}:
                reason = next(value for key, value in policy["reject"].items()
                              if key.casefold() == name)
                rejected.append({"repo": source["repo"], "split": source["split"],
                                 "cohort": source["cohort"], "reason": reason})
                continue
            rows.append({**source, "role": policy["cohorts"][source["cohort"]]["role"],
                         "comparison_rule": policy["cohorts"][source["cohort"]]["comparison_rule"]})
    by_split = Counter(row["split"] for row in rows)
    if by_split["train"] < 120:
        raise ValueError(f"fewer than 120 train source families: {by_split['train']}")
    return {"schema": "retrieval_curated_sources_v2", "sources": rows,
            "role_rejections": rejected, "counts": dict(by_split),
            "batch_hashes": {path.name: sha256(path.read_bytes()).hexdigest() for path in batch_paths},
            "policy_sha256": sha256(policy_file.read_bytes()).hexdigest(),
            "exclusions_sha256": sha256((V1 / "EXCLUSIONS.json").read_bytes()).hexdigest(),
            "role_review": "author description-level role curation; no independent semantic question review",
            "training_labels_admitted": 0, "training_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = assemble()
    with args.out.open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(json.dumps({"counts": manifest["counts"],
                      "role_rejections": len(manifest["role_rejections"]),
                      "sha256": sha256(args.out.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
