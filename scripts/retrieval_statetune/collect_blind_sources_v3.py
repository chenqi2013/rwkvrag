"""Acquire separate, directly comparable heldout README cohorts for V3."""

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path

from collect_curated_sources_v2 import ALLOWED_LICENSES, ROOT, blob_sha
from collect_source_supplement import gh_retry


V1 = ROOT / "llamaindex-retrieval/statetune/retrieval-v1-20260923"
V2 = ROOT / "llamaindex-retrieval/statetune/retrieval-v2-20260923"
V3 = ROOT / "llamaindex-retrieval/statetune/retrieval-v3-20260924"


def collect(out):
    seed_path = V3 / "BLIND-COHORTS.json"
    seed = json.loads(seed_path.read_text())
    if seed.pop("schema") != "retrieval_blind_cohorts_v3" or set(seed) != {"heldout"}:
        raise ValueError("blind cohort schema mismatch")
    prior_path = V2 / "SOURCES-CURATED.json"
    prior = json.loads(prior_path.read_text())
    known = {row["repo"].casefold() for row in prior["sources"]}
    exclusions = json.loads((V1 / "EXCLUSIONS.json").read_text())
    forbidden = {x.casefold() for x in exclusions["source_families"]}
    rows, rejected = [], []
    (out / "readmes").mkdir(parents=True)
    for cohort, repos in seed["heldout"].items():
        if len(repos) != 4:
            raise ValueError("blind comparison cohorts must have four registered projects")
        for requested in repos:
            name = requested.casefold()
            if name in known or f"repo:{name}" in forbidden:
                raise ValueError(f"source family reused or excluded: {requested}")
            known.add(name)
            try:
                info = gh_retry(f"repos/{requested}")
                readme = gh_retry(f"repos/{requested}/readme")
                raw = base64.b64decode(readme["content"], validate=False)
                decoded = raw.decode("utf-8")
                license_id = (info.get("license") or {}).get("spdx_id")
                if info["full_name"].casefold() != name or info.get("archived") or info.get("fork"):
                    raise ValueError("alias, archived repository or fork")
                if license_id not in ALLOWED_LICENSES:
                    raise ValueError(f"license metadata {license_id} outside acquisition policy")
                if not 300 <= len(decoded) <= 60_000 or blob_sha(raw) != readme["sha"]:
                    raise ValueError("README length or Git blob mismatch")
                digest = sha256(raw).hexdigest()
                if digest in exclusions["source_hashes"]:
                    raise ValueError("excluded evaluation source hash")
                target = out / "readmes" / (info["full_name"].replace("/", "--") + ".md")
                with target.open("xb") as destination:
                    destination.write(raw)
                rows.append({"repo": info["full_name"], "family": f"repo:{name}",
                             "split": "heldout", "cohort": cohort,
                             "description": info["description"],
                             "license_metadata": license_id,
                             "default_branch": info["default_branch"],
                             "readme_path": str(target.relative_to(ROOT)),
                             "blob_sha": readme["sha"], "sha256": digest,
                             "characters": len(decoded),
                             "blob_uri": f"https://api.github.com/repos/{info['full_name']}/git/blobs/{readme['sha']}",
                             "role": cohort,
                             "comparison_rule": "Compare the stated user workflow and platform requirements; verify each project independently."})
            except (KeyError, UnicodeError, ValueError, RuntimeError, OSError) as error:
                rejected.append({"repo": requested, "cohort": cohort, "reason": str(error)})
        print(json.dumps({"cohort": cohort,
                          "accepted": sum(r["cohort"] == cohort for r in rows)}, ensure_ascii=False), flush=True)
    return {"schema": "retrieval_blind_sources_v3", "sources": rows,
            "rejected": rejected, "seed_sha256": sha256(seed_path.read_bytes()).hexdigest(),
            "prior_sources_sha256": sha256(prior_path.read_bytes()).hexdigest(),
            "exclusions_sha256": sha256((V1 / "EXCLUSIONS.json").read_bytes()).hexdigest(),
            "question_labels_reviewed": False, "training_exported": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("source output must remain under repository")
    out.mkdir(parents=True, exist_ok=False)
    manifest = collect(out)
    with args.manifest.open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(json.dumps({"accepted": len(manifest["sources"]),
                      "rejected": len(manifest["rejected"]),
                      "sha256": sha256(args.manifest.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
