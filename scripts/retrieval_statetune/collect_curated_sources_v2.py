"""Pin official README snapshots for explicitly reviewed functional cohorts.

This collects source material only. A GitHub code-license field does not grant
rights to republish README text or certify training use. Raw files stay local.
"""

import argparse
import base64
from hashlib import sha1, sha256
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "llamaindex-retrieval/statetune/retrieval-v1-20260923"
ALLOWED_LICENSES = {"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "ISC", "MPL-2.0"}


def gh(path):
    result = subprocess.run(["gh", "api", "-X", "GET", path], capture_output=True,
                            text=True, timeout=45)
    if result.returncode:
        raise RuntimeError(f"GitHub API request failed: {path}")
    return json.loads(result.stdout)


def blob_sha(raw):
    return sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def collect(cohorts, exclusions, previous, out):
    old = {row["repo"].casefold(): row for row in previous["sources"]}
    forbidden = {family.casefold() for family in exclusions["source_families"]}
    forbidden_hashes = set(exclusions["source_hashes"])
    rows, rejected, seen = [], [], set()
    (out / "readmes").mkdir(parents=True)
    for split, groups in cohorts.items():
        if split not in {"train", "dev", "heldout"}:
            raise ValueError("invalid split")
        for cohort, repos in groups.items():
            if len(repos) < 4 or len(set(map(str.casefold, repos))) != len(repos):
                raise ValueError(f"invalid cohort {cohort}")
            for requested in repos:
                identity = requested.casefold()
                if identity in seen:
                    raise ValueError(f"repository listed twice: {requested}")
                seen.add(identity)
                if f"repo:{identity}" in forbidden:
                    rejected.append({"repo": requested, "split": split, "cohort": cohort,
                                     "reason": "excluded_eval_family"})
                    continue
                try:
                    cached = old.get(identity)
                    if cached:
                        source = ROOT / cached["readme_path"]
                        raw = source.read_bytes()
                        if sha256(raw).hexdigest() != cached["sha256"] or blob_sha(raw) != cached["blob_sha"]:
                            raise ValueError("V1 source hash mismatch")
                        metadata = {"full_name": cached["repo"],
                                    "license": {"spdx_id": cached["license_metadata"]},
                                    "archived": False, "fork": False,
                                    "description": cached["description"],
                                    "default_branch": cached["default_branch"]}
                        readme_sha = cached["blob_sha"]
                    else:
                        metadata = gh(f"repos/{requested}")
                        readme = gh(f"repos/{requested}/readme")
                        raw = base64.b64decode(readme["content"], validate=False)
                        readme_sha = readme["sha"]
                    canonical = metadata["full_name"]
                    license_id = (metadata.get("license") or {}).get("spdx_id")
                    if canonical.casefold() != identity:
                        raise ValueError("repository redirects to an alias; review needed")
                    if metadata.get("archived") or metadata.get("fork"):
                        raise ValueError("archived or fork")
                    if license_id not in ALLOWED_LICENSES:
                        raise ValueError(f"license metadata {license_id} outside acquisition policy")
                    decoded = raw.decode("utf-8")
                    if not 300 <= len(decoded) <= 60_000:
                        raise ValueError("README length outside bounds")
                    if blob_sha(raw) != readme_sha:
                        raise ValueError("Git blob identity mismatch")
                    digest = sha256(raw).hexdigest()
                    if digest in forbidden_hashes:
                        raise ValueError("excluded evaluation source hash")
                    target = out / "readmes" / (canonical.replace("/", "--") + ".md")
                    with target.open("xb") as destination:
                        destination.write(raw)
                    rows.append({"repo": canonical, "family": f"repo:{identity}",
                                 "split": split, "cohort": cohort,
                                 "description": metadata["description"],
                                 "license_metadata": license_id,
                                 "default_branch": metadata["default_branch"],
                                 "readme_path": str(target.relative_to(ROOT)),
                                 "blob_sha": readme_sha, "sha256": digest,
                                 "characters": len(decoded),
                                 "blob_uri": f"https://api.github.com/repos/{canonical}/git/blobs/{readme_sha}",
                                 "reused_v1_snapshot": bool(cached)})
                except (KeyError, UnicodeError, ValueError, RuntimeError,
                        subprocess.TimeoutExpired, OSError) as error:
                    rejected.append({"repo": requested, "split": split, "cohort": cohort,
                                     "reason": str(error)})
            print(json.dumps({"split": split, "cohort": cohort,
                              "accepted": sum(r["cohort"] == cohort and r["split"] == split for r in rows),
                              "rejected": sum(r["cohort"] == cohort and r["split"] == split for r in rejected)},
                             ensure_ascii=False), flush=True)
    return {"schema": "retrieval_github_sources_v2", "sources": rows, "rejected": rejected,
            "cohorts_sha256": sha256((ROOT / "llamaindex-retrieval/statetune/retrieval-v2-20260923/COHORTS.json").read_bytes()).hexdigest(),
            "exclusions_sha256": sha256((V1 / "EXCLUSIONS.json").read_bytes()).hexdigest(),
            "source_role_review": "author-curated cohorts; individual comparison questions not independently reviewed",
            "semantic_labels_reviewed": False, "training_exported": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("raw source output must stay under repository")
    out.mkdir(exist_ok=False, parents=True)
    cohort_file = ROOT / "llamaindex-retrieval/statetune/retrieval-v2-20260923/COHORTS.json"
    cohorts = json.loads(cohort_file.read_text())
    if cohorts.pop("schema") != "retrieval_source_cohorts_v2":
        raise ValueError("cohort schema mismatch")
    cohorts.pop("review_status")
    manifest = collect(cohorts, json.loads((V1 / "EXCLUSIONS.json").read_text()),
                       json.loads((V1 / "SOURCES.json").read_text()), out)
    with args.manifest.open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(json.dumps({"sources": len(manifest["sources"]), "rejected": len(manifest["rejected"]),
                      "sha256": sha256(args.manifest.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
