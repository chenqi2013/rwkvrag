"""Acquire extra official README snapshots without changing the frozen V2 batch."""

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import time

from collect_curated_sources_v2 import ALLOWED_LICENSES, ROOT, V1, blob_sha


V2 = ROOT / "llamaindex-retrieval/statetune/retrieval-v2-20260923"
LICENSES = ALLOWED_LICENSES | {"Unlicense"}


def gh_retry(path):
    for attempt in range(3):
        result = subprocess.run(["gh", "api", "-X", "GET", path], capture_output=True,
                                text=True, timeout=45)
        if result.returncode == 0:
            return json.loads(result.stdout)
        if attempt < 2:
            time.sleep(1 + attempt * 2)
    raise RuntimeError(f"GitHub API failed after three attempts: {path}")


def collect(cohorts, out):
    prior = json.loads((V2 / "SOURCES.json").read_text())
    forbidden = json.loads((V1 / "EXCLUSIONS.json").read_text())
    used = {row["repo"].casefold() for row in prior["sources"]}
    excluded = {row.casefold() for row in forbidden["source_families"]}
    rows, rejected = [], []
    (out / "readmes").mkdir(parents=True)
    for split, groups in cohorts.items():
        if split not in {"train", "dev", "heldout"}:
            raise ValueError("invalid split")
        for cohort, repos in groups.items():
            for requested in repos:
                name = requested.casefold()
                if name in used or f"repo:{name}" in excluded:
                    rejected.append({"repo": requested, "split": split,
                                     "cohort": cohort, "reason": "duplicate_or_excluded_family"})
                    continue
                used.add(name)
                try:
                    info = gh_retry(f"repos/{requested}")
                    readme = gh_retry(f"repos/{requested}/readme")
                    raw = base64.b64decode(readme["content"], validate=False)
                    text = raw.decode("utf-8")
                    license_id = (info.get("license") or {}).get("spdx_id")
                    if info["full_name"].casefold() != name or info.get("archived") or info.get("fork"):
                        raise ValueError("alias, archived repository or fork needs review")
                    if license_id not in LICENSES:
                        raise ValueError(f"license metadata {license_id} outside acquisition policy")
                    if not 300 <= len(text) <= 60_000 or blob_sha(raw) != readme["sha"]:
                        raise ValueError("README length or Git blob identity invalid")
                    digest = sha256(raw).hexdigest()
                    if digest in forbidden["source_hashes"]:
                        raise ValueError("excluded evaluation source hash")
                    target = out / "readmes" / (info["full_name"].replace("/", "--") + ".md")
                    with target.open("xb") as destination:
                        destination.write(raw)
                    rows.append({"repo": info["full_name"], "family": f"repo:{name}",
                                 "split": split, "cohort": cohort,
                                 "description": info["description"],
                                 "license_metadata": license_id,
                                 "default_branch": info["default_branch"],
                                 "readme_path": str(target.relative_to(ROOT)),
                                 "blob_sha": readme["sha"], "sha256": digest,
                                 "characters": len(text),
                                 "blob_uri": f"https://api.github.com/repos/{info['full_name']}/git/blobs/{readme['sha']}",
                                 "reused_v1_snapshot": False})
                except (KeyError, UnicodeError, ValueError, RuntimeError,
                        subprocess.TimeoutExpired, OSError) as error:
                    rejected.append({"repo": requested, "split": split,
                                     "cohort": cohort, "reason": str(error)})
            print(json.dumps({"split": split, "cohort": cohort,
                              "accepted": sum(r["split"] == split and r["cohort"] == cohort for r in rows)},
                             ensure_ascii=False), flush=True)
    return {"schema": "retrieval_github_sources_supplement_v1", "sources": rows,
            "rejected": rejected,
            "cohorts_sha256": sha256((V2 / "COHORTS-SUPPLEMENT.json").read_bytes()).hexdigest(),
            "prior_sources_sha256": sha256((V2 / "SOURCES.json").read_bytes()).hexdigest(),
            "exclusions_sha256": sha256((V1 / "EXCLUSIONS.json").read_bytes()).hexdigest(),
            "semantic_labels_reviewed": False, "training_exported": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("raw output must remain under repository")
    out.mkdir(parents=True, exist_ok=False)
    cohorts = json.loads((V2 / "COHORTS-SUPPLEMENT.json").read_text())
    if cohorts.pop("schema") != "retrieval_source_supplement_v1":
        raise ValueError("cohort schema mismatch")
    manifest = collect(cohorts, out)
    with args.manifest.open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(json.dumps({"accepted": len(manifest["sources"]),
                      "rejected": len(manifest["rejected"]),
                      "sha256": sha256(args.manifest.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
