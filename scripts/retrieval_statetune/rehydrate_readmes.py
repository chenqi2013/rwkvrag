"""Verify or restore pinned README blobs referenced by a source manifest.

Existing mismatched files are never overwritten. Only paths below data/ can be
restored, and the Git blob SHA and SHA-256 must both match the manifest.
"""

import argparse
import base64
from hashlib import sha1, sha256
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def verify(raw, source):
    blob = sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if blob != source["blob_sha"] or sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError(f"pinned README identity mismatch: {source['repo']}")
    raw.decode("utf-8")


def restore(sources, fetch_missing=False):
    existing = fetched = 0
    for source in sources:
        path = (ROOT / source["readme_path"]).resolve()
        if not path.is_relative_to(DATA.resolve()):
            raise ValueError("README path escapes local data directory")
        if path.exists():
            verify(path.read_bytes(), source)
            existing += 1
            continue
        if not fetch_missing:
            raise FileNotFoundError(path)
        repo = source["repo"]
        blob = source["blob_sha"]
        result = subprocess.run(["gh", "api", "-X", "GET", f"repos/{repo}/git/blobs/{blob}"],
                                capture_output=True, text=True, timeout=45)
        if result.returncode:
            raise RuntimeError(f"GitHub blob fetch failed: {repo} {blob}")
        raw = base64.b64decode(json.loads(result.stdout)["content"], validate=False)
        verify(raw, source)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as destination:
            destination.write(raw)
        fetched += 1
    return existing, fetched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--fetch-missing", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.sources.read_text())
    if manifest.get("schema") not in {"retrieval_curated_sources_v2", "retrieval_curated_sources_v3",
                                      "retrieval_curated_sources_v4"}:
        raise ValueError("source schema mismatch")
    existing, fetched = restore(manifest["sources"], args.fetch_missing)
    print(json.dumps({"sources": len(manifest["sources"]), "verified_existing": existing,
                      "fetched_missing": fetched, "all_hashes_match": True}))


if __name__ == "__main__":
    main()
