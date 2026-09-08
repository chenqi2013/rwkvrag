#!/usr/bin/env python3
"""Freeze, package and verify public experiment bytes; no model or retrieval calls.

Standard library only. A selection pins original files before compression. All
records remain byte-for-byte unchanged. This is an integrity check, not a rerun
of the experiment's source, evidence or semantic audits.
"""
import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile


LIMIT = 50_000_000  # Every distributed archive must be strictly smaller.
DIRECTORIES = {
    "materials": ["materials-smoke-v1", "materials-smoke-v2", "materials-smoke-v3",
                  "source-materials-v2", "source-materials-v3",
                  "materials-v2-download", "materials-v2-remote",
                  "materials-v3-download", "materials-v3-remote",
                  "review/materials-smoke-v2", "review/materials-smoke-v3"],
    "wiki": ["wiki-smoke-v1", "wiki-smoke-v2", "source-wiki-v1", "source-wiki-v2",
             "source-wiki-v3", "wiki-preflight-v1", "wiki-preflight-v2", "wiki-preflight-v3",
             "wiki-planner-prefill-v3-tests", "wiki-smoke-v1-trace-review",
             "review/wiki-smoke-v1", "review/wiki-smoke-v2"],
    "resolver-reminder": ["resolver-reminder-experiment"],
    "literal-quote": ["literal-quote-feasibility"],
    "index": ["index-import-v1", "source-index-v1"],
}
CORPUS_FILES = ["finewiki-zh-5000/manifest.jsonl", "finewiki-zh-5000/freeze.json",
                "selection-policy.json", "upstream/README.md"]
CREDENTIAL_KEYS = {"authorization", "proxy_authorization", "api_key", "apikey",
                   "access_token", "refresh_token", "password", "client_secret",
                   "cookie", "set_cookie"}
SECRET_PATTERNS = {
    "private-key": rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "provider-token": rb"\b(?:sk-(?:proj-)?|hf_|gh[pousr]_|github_pat_)[A-Za-z0-9_-]{20,}",
    "bearer-credential": rb"\bBearer[ \t]+[A-Za-z0-9_.~+/-]{8,}",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or "\\" in name or "\x00" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError("unsafe archive/member path")
    return path


def checked_source(root, relative):
    parts = safe_name(relative).parts
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symbolic source rejected: {relative}")
    if not path.is_file():
        raise ValueError(f"source is not a regular file: {relative}")
    return path


def secret_scan(data, name, depth=0):
    """Stop without printing any credential value, including encoded wire bodies."""
    for category, pattern in SECRET_PATTERNS.items():
        if re.search(pattern, data):
            raise ValueError(f"possible secret: {name} ({category}); no values printed")
    if depth > 3:
        return
    try:
        obj = json.loads(data)
    except (ValueError, UnicodeError):
        return

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = key.lower().replace("-", "_")
                if normalized in CREDENTIAL_KEYS and child:
                    raise ValueError(f"possible secret: {name} (credential field); no values printed")
                if normalized.endswith(("base64", "_b64")) and isinstance(child, str):
                    try:
                        decoded = base64.b64decode(child, validate=True)
                    except ValueError:
                        continue
                    secret_scan(decoded, name, depth + 1)
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(obj)


def scan_file(path, name):
    # JSONL may contain the 126 MB full index scroll; do not parse it as one object.
    if path.suffix == ".jsonl":
        with path.open("rb") as stream:
            for line in stream:
                secret_scan(line, name)
    else:
        secret_scan(path.read_bytes(), name)


def exclusion(path):
    if any(p in {".venv", "__pycache__", ".git", "node_modules"} for p in path.parts):
        return "dependency/cache/repository internals"
    if path.suffix == ".pyc":
        return "Python bytecode"
    if path.name == ".env" or path.name.startswith(".env."):
        return "environment file (excluded without reading its contents)"
    if path.name.startswith(("RESUME", "PUBLISHED")):
        return "mutable progress marker"
    if path.name.endswith((".tar.gz", ".tgz", ".zip")):
        return "redundant transport archive; selected originals retained separately"
    return None


def top_group(name):
    if name.startswith(("materials-smoke-v1-", "materials-smoke-v2-", "materials-smoke-v3-",
                        "materials-v2-", "materials-v3-", "MATERIALS-LABEL-",
                        "native-smoke-offline")):
        return "materials"
    if name.startswith(("wiki-smoke-v1", "wiki-smoke-v2", "WIKI-SMOKE-V1-", "WIKI-SMOKE-V2-",
                        "wiki-preflight-v1-", "wiki-preflight-v2-", "wiki-preflight-v3-",
                        "wiki-planner-prefill-v3", "wiki-runner-v2", "wiki-v1-", "wiki-v2-",
                        "wiki-v3-", "audit-wiki-smoke-v1", "audit-wiki-smoke-v2",
                        "audit-wiki-planner-prefill-pair", "run-wiki-smoke-v1-",
                        "run-wiki-smoke-v2-")):
        return "wiki"
    if name.startswith(("index-", "ingest-")):
        return "index"
    if name in {"baseline-pytest.txt", "pristine-base-pytest.txt", "current-pytest.txt",
                "pytest-comparison.json", "current-pytest-contract-v2.txt",
                "pytest-comparison-contract-v2.json", "changed-ruff.txt"}:
        return "checks"
    return None


def prepare(records, corpus, destination):
    rows, excluded, selected = [], [], set()

    def add(origin, root, path, group):
        rel = path.relative_to(root).as_posix()
        if (origin, rel) in selected:
            return
        selected.add((origin, rel))
        reason = exclusion(path.relative_to(root))
        if reason:
            excluded.append({"origin": origin, "path": rel, "reason": reason})
            return
        path = checked_source(root, rel)
        scan_file(path, f"{origin}/{rel}")
        rows.append({"origin": origin, "source_path": rel, "path": f"{origin}/{rel}",
                     "group": group, "bytes": path.stat().st_size, "sha256": file_sha(path)})

    for group, names in DIRECTORIES.items():
        for name in names:
            parent = records / name
            if not parent.is_dir() or parent.is_symlink():
                raise ValueError(f"missing/unsafe completed record directory: {name}")
            for folder, dirs, files in os.walk(parent, followlinks=False):
                for entry in list(dirs):
                    child = Path(folder) / entry
                    reason = exclusion(child.relative_to(records))
                    if reason or child.is_symlink():
                        dirs.remove(entry)
                        excluded.append({"origin": "records", "path": str(child.relative_to(records)),
                                         "reason": reason or "symlink directory"})
                for filename in files:
                    add("records", records, Path(folder) / filename, group)
    for path in sorted(records.iterdir()):
        group = top_group(path.name) if path.is_file() else None
        if group:
            add("records", records, path, group)
        elif path.name not in {p.split("/")[0] for names in DIRECTORIES.values() for p in names}:
            excluded.append({"origin": "records", "path": path.name,
                             "reason": "not in the completed-record allowlist"})
    add("records", records, records / "review/build_wiki_smoke_reviews.py", "wiki")
    for name in CORPUS_FILES:
        add("corpus", corpus, corpus / name, "index")
    rows.sort(key=lambda row: row["path"])
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("duplicate selected member")
    save_new(destination, {"schema": "bm250820-public-archive-selection-v1", "members": rows,
                           "exclusions": sorted(excluded, key=lambda x: (x["origin"], x["path"])),
                           "scope": "completed development records only; not an accuracy certification",
                           "secret_scan": "pattern, JSON credential key and base64 wire checks passed"})
    print(json.dumps({"selection": str(destination), "members": len(rows),
                      "bytes": sum(row["bytes"] for row in rows), "sha256": file_sha(destination)}))


def write_tar(path, rows, roots):
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as zipped:
            with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for row in rows:
                    source = checked_source(roots[row["origin"]], row["source_path"])
                    if source.stat().st_size != row["bytes"] or file_sha(source) != row["sha256"]:
                        raise ValueError(f"source changed since selection: {row['path']}")
                    scan_file(source, row["path"])
                    info = tarfile.TarInfo(row["path"])
                    info.size, info.mode, info.mtime = row["bytes"], 0o644, 0
                    with source.open("rb") as stream:
                        tar.addfile(info, stream)


def build(records, corpus, selection, output):
    inventory = read_json(selection)
    if output.exists():
        raise ValueError("build output must be a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    groups = {}
    for row in inventory["members"]:
        safe_name(row["path"])
        groups.setdefault(row["group"], []).append(row)
    manifest = {"schema": "bm250820-public-archives-v1", "selection_sha256": file_sha(selection),
                "builder_sha256": file_sha(Path(__file__)), "max_archive_bytes_exclusive": LIMIT,
                "archives": [], "members": [], "exclusions": inventory["exclusions"],
                "integrity_only": True, "raw_contents_modified": False}
    roots = {"records": records, "corpus": corpus}

    def archive_group(group, rows):
        path = output / f"{group}.tar.gz"
        write_tar(path, rows, roots)
        if path.stat().st_size >= LIMIT:
            path.unlink()
            if len(rows) < 2:
                raise ValueError("single compressed member exceeds limit; cannot discard or truncate")
            midpoint = len(rows) // 2
            archive_group(group + "-a", rows[:midpoint])
            archive_group(group + "-b", rows[midpoint:])
            return
        manifest["archives"].append({"path": path.name, "bytes": path.stat().st_size,
                                      "sha256": file_sha(path), "members": len(rows)})
        manifest["members"].extend([{**row, "archive": path.name} for row in rows])
    for group, rows in sorted(groups.items()):
        archive_group(group, rows)
    save_new(output / "RECORDS-MANIFEST.json", manifest)
    verify(output / "RECORDS-MANIFEST.json")
    print(json.dumps({"manifest_sha256": file_sha(output / "RECORDS-MANIFEST.json"),
                      "members": len(manifest["members"]), "archives": manifest["archives"]}))


def verify(manifest_path, extract=None):
    manifest = read_json(manifest_path)
    expected, seen = {}, set()
    for row in manifest["members"]:
        safe_name(row["path"])
        safe_name(row["archive"])
        if row["path"] in expected:
            raise ValueError("duplicate manifest member")
        expected[row["path"]] = row
    names = [item["path"] for item in manifest["archives"]]
    if len(names) != len(set(names)):
        raise ValueError("duplicate archive declaration")
    if set(row["archive"] for row in expected.values()) != set(names):
        raise ValueError("archive declarations do not match member groups")
    # Validate every archive and member before creating an extraction directory.
    for item in manifest["archives"]:
        if len(safe_name(item["path"]).parts) != 1:
            raise ValueError("archive must be adjacent to manifest")
        archive = checked_source(manifest_path.parent, item["path"])
        if archive.stat().st_size >= LIMIT or archive.stat().st_size != item["bytes"]:
            raise ValueError("archive size mismatch")
        if file_sha(archive) != item["sha256"]:
            raise ValueError("archive SHA mismatch")
        count = 0
        with tarfile.open(archive, "r:gz", ignore_zeros=True) as tar:
            for member in tar:
                safe_name(member.name)
                if not member.isfile() or member.issparse() or member.name in seen:
                    raise ValueError("nonregular, sparse or duplicate tar member")
                row = expected.get(member.name)
                if row is None or row["archive"] != item["path"] or member.size != row["bytes"]:
                    raise ValueError("unexpected or mismatched tar member")
                with tar.extractfile(member) as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                        raise ValueError("member SHA mismatch")
                seen.add(member.name)
                count += 1
        if count != item["members"]:
            raise ValueError("archive member count mismatch")
    if seen != set(expected):
        raise ValueError("missing declared members")
    if extract is not None:
        if extract.exists() or extract.is_symlink():
            raise ValueError("extract destination must be new and absent")
        extract.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".verified-records-", dir=extract.parent) as temp:
            staging = Path(temp)
            extracted = set()
            for item in manifest["archives"]:
                archive = checked_source(manifest_path.parent, item["path"])
                if file_sha(archive) != item["sha256"]:
                    raise ValueError("archive changed before extraction")
                with tarfile.open(archive, "r:gz", ignore_zeros=True) as tar:
                    for member in tar:
                        # Re-check names/types on the extraction pass too; no tar.extractall.
                        safe_name(member.name)
                        if not member.isfile() or member.name not in expected:
                            raise ValueError("archive changed before extraction")
                        destination = staging / member.name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with tar.extractfile(member) as source, destination.open("xb") as target:
                            shutil.copyfileobj(source, target)
                        row = expected[member.name]
                        if destination.stat().st_size != row["bytes"] or file_sha(destination) != row["sha256"]:
                            raise ValueError("archive changed before extraction")
                        extracted.add(member.name)
            if extracted != set(expected):
                raise ValueError("archive lost members before extraction")
            os.rename(staging, extract)
    print(json.dumps({"verified_members": len(seen), "archives": len(names), "integrity_only": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "build"):
        sub = commands.add_parser(name)
        sub.add_argument("--records", type=Path, required=True)
        sub.add_argument("--corpus", type=Path, required=True, help="directory containing finewiki-zh-5000")
        sub.add_argument("--selection", type=Path, required=True)
        if name == "build":
            sub.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser("verify")
    sub.add_argument("--manifest", type=Path, required=True)
    sub.add_argument("--extract", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.records.resolve(), args.corpus.resolve(), args.selection)
    elif args.command == "build":
        build(args.records.resolve(), args.corpus.resolve(), args.selection, args.output)
    else:
        verify(args.manifest, args.extract)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, tarfile.TarError) as error:
        raise SystemExit(str(error)) from None
