"""Collect fresh official README snapshots, separated by topic before labels.

Uses the local gh login without reading or printing its token. Raw files stay
under an ignored corpus directory; the returned manifest pins Git blob and
SHA-256 identities. This is source acquisition, not training or evaluation.
"""

import argparse
import base64
from hashlib import sha1, sha256
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
TOPICS = ROOT / "llamaindex-retrieval/statetune/retrieval-v1-20260923/SOURCE-TOPICS.json"
EXCLUSIONS = ROOT / "llamaindex-retrieval/statetune/retrieval-v1-20260923/EXCLUSIONS.json"
LIVE = ROOT / "llamaindex-retrieval/eval/live-retrieval-complete-20260923-v1/INPUTS.json"
ALLOWED_LICENSES = {"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "ISC", "MPL-2.0"}


def gh(path, *fields):
    command = ["gh", "api", "-X", "GET", path]
    for key, value in fields:
        command += ["-f", f"{key}={value}"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=35)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"GitHub API request timed out for {path}") from error
    if result.returncode:
        raise RuntimeError(f"GitHub API request failed for {path}; status hidden to avoid leaking local auth")
    return json.loads(result.stdout)


def git_blob_sha(raw):
    return sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def acquire(args):
    topics = json.loads(TOPICS.read_text())
    exclusions = json.loads(EXCLUSIONS.read_text())
    old_repos = {value.removeprefix("repo:").casefold() for value in exclusions["source_families"]
                 if value.startswith("repo:")}
    seen_objects = {str(obj).casefold() for case in json.loads(LIVE.read_text())["cases"]
                    for obj in (case.get("objects") or [])}
    output_dir = args.out.resolve()
    if not output_dir.is_relative_to(ROOT):
        raise ValueError("source output must remain under the repository")
    output_dir.mkdir(parents=True, exist_ok=False)
    readmes = output_dir / "readmes"
    readmes.mkdir()
    rows, failures, used = [], [], set()
    for split, names in topics.items():
        for topic in names:
            try:
                found = gh("search/repositories", ("q", f"topic:{topic} stars:>100 archived:false fork:false"),
                           ("sort", "stars"), ("order", "desc"), ("per_page", "50"))["items"]
            except (RuntimeError, KeyError) as error:
                failures.append({"topic": topic, "stage": "search", "error": str(error)})
                continue
            selected = 0
            for repo in found:
                identity = repo["full_name"]
                license_id = (repo.get("license") or {}).get("spdx_id")
                if (identity.casefold() in used or identity.casefold() in old_repos or
                        repo["name"].casefold() in seen_objects or license_id not in ALLOWED_LICENSES or
                        repo.get("archived") or repo.get("fork") or not repo.get("description")):
                    continue
                try:
                    readme = gh(f"repos/{identity}/readme")
                    raw = base64.b64decode(readme["content"], validate=False)
                    text = raw.decode("utf-8")
                    if not 300 <= len(text) <= 60_000 or git_blob_sha(raw) != readme["sha"]:
                        raise ValueError("README text length or blob identity invalid")
                    target = readmes / (identity.replace("/", "--") + ".md")
                    relative_path = str(target.relative_to(ROOT))
                    with target.open("xb") as output:
                        output.write(raw)
                    rows.append({"repo": identity, "family": "repo:" + identity.casefold(),
                                 "topic": topic, "split": split, "description": repo["description"],
                                 "license_metadata": license_id, "default_branch": repo["default_branch"],
                                 "readme_path": relative_path, "blob_sha": readme["sha"],
                                 "sha256": sha256(raw).hexdigest(), "characters": len(text),
                                 "blob_uri": f"https://api.github.com/repos/{identity}/git/blobs/{readme['sha']}"})
                    used.add(identity.casefold())
                    selected += 1
                    if selected >= args.per_topic:
                        break
                except (RuntimeError, ValueError, KeyError, UnicodeError) as error:
                    failures.append({"topic": topic, "repo": identity, "stage": "readme", "error": str(error)})
            if selected < args.per_topic:
                failures.append({"topic": topic, "stage": "quota", "selected": selected,
                                 "requested": args.per_topic})
            print(json.dumps({"topic": topic, "split": split, "selected": selected,
                              "total": len(rows)}, ensure_ascii=False), flush=True)
    return {"schema": "retrieval_github_sources_v1", "sources": rows,
            "failures": failures, "topics_sha256": sha256(TOPICS.read_bytes()).hexdigest(),
            "exclusions_sha256": sha256(EXCLUSIONS.read_bytes()).hexdigest(),
            "live_inputs_sha256": sha256(LIVE.read_bytes()).hexdigest(),
            "quality_verified": False, "training_exported": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--per-topic", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.per_topic <= 10:
        raise ValueError("per-topic must be 1..10")
    result = acquire(args)
    with args.manifest.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({"sources": len(result["sources"]), "failures": len(result["failures"]),
                      "sha256": sha256(args.manifest.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
