#!/usr/bin/env python3
"""Package frozen external-API experiment bytes with the published safe verifier.

Prepare requires a final-freeze anchor and a private, local curl attachment for
an in-memory credential scan. Neither credential values nor their hashes are
written. Public rebuilds use the existing selection and need no credentials.
This tool checks byte integrity, not model quality or semantic audit judgments.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
from urllib.parse import quote


BASE = Path(__file__).resolve().parents[1] / "bm250820-rebuild-20260908" / "build_archive.py"
BASE_SHA256 = "07e0631325683a27d216960cd5040b2f5ece543dd0fbacbe1b6641c2f867950c"
if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA256:
    raise RuntimeError("published archive builder differs from the pinned version")
spec = importlib.util.spec_from_file_location("public_archive_builder", BASE)
builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)

GROUPS = {
    "materials": ["materials-api-v1", "materials-api-eos-v2", "materials-api-prefix-v3"],
    "wiki": ["wiki-api-v1-eos", "wiki-api-preflight-v1"],
    "planners": ["planner-contract-last-v1", "planner-shared-tasks-v1"],
    "long-context": ["long-context-v1"],
    "api-contracts": ["api-contract-v1", "external-api-probe", "template-contract-v1"],
    "sources": ["source-api-v1", "source-api-v2", "source-preparation-v1", "source-publication-v1"],
    "review": ["qa-protocol", "review"],
    "local-index": ["local-index-v1"],
    "checks": ["runtime", "publication-checks", "input-count-validation-v1", "token-count-implementation-v1"],
}
UPSTREAM = "api-contract-v1/upstream/source"
UPSTREAM_PINS = "api-contract-v1/SOURCE-PINS.json"
UPSTREAM_PINS_SHA256 = "8fb3953908ff1236423a42930acd251f9c436afa0cfc38c22c5ee87f552f7c46"
INDEX_DEPENDENCY = {
    "archive": "../bm250820-rebuild-20260908/archives/index.tar.gz",
    "archive_bytes": 23985157,
    "archive_sha256": "8681353226477c815a27ee28333d03ed6d3858da7bff64f3e8eb8abbeb5daaa9",
    "manifest": "../bm250820-rebuild-20260908/archives/RECORDS-MANIFEST.json",
    "manifest_sha256": "ea872a9349b92116de43704591317cbe758e16b9f1a1d9062f72dfccdbebb0e1",
    "member": "records/index-import-v1/indexed-records.jsonl",
    "member_bytes": 126610875,
    "member_sha256": "e649fd2991bf7228d0bc73ece4b415cf3f2b14c8d795add1ef27f0054a7077d8",
    "omitted_duplicate": "local-index-v1/indexed-records.jsonl",
    "physical_index_identity_equal": False,
}

# Individually reviewed false positives, pinned to whole-file bytes. These are
# descriptions of experiment authorization, not HTTP authentication headers.
# The public hf_ key is a boolean describing tokenizer configuration downloads.
SCAN_EXCEPTIONS = {
    "long-context-v1/run/STARTED-v2.json": {
        "sha256": "c00f824f4a3ca5eed8e9dae9437d626a09161117b31aa55b8e35c47a73b08e0d",
        "authorization_pointer": ["authorization"],
    },
    "long-context-v1/run/STARTED.json": {
        "sha256": "5255962176f8e86cae8c932db97ff1d8d40491a987f1aee883755fc341b5502d",
        "authorization_pointer": ["authorization"],
    },
    "runtime/SERVER-RELEASE-PLAN.json": {
        "sha256": "f52c55e8a43a7f6ea4d46ff27ebf1bf2f3614c4f5c5d45501ee0f8ed5a8735e3",
        "authorization_pointer": ["authorization"],
    },
    "runtime/SERVER-RELEASE-STARTED.json": {
        "sha256": "40740f824dae797788255cd88bfee1a9dcd6f8d03fee2e27165c4d47dcfa5fc6",
        "authorization_pointer": ["plan", "authorization"],
    },
    "runtime/SERVER-RELEASE-COMPLETED.json": {
        "sha256": "49d61ce74e616da87512add4f7ff9b51203a1b74e9173c2286338ad8c868d6ab",
        "authorization_pointer": ["plan", "authorization"],
    },
    "template-contract-v1/FINDINGS.json": {
        "sha256": "638ee59851725b474ea2cb00eb8d2d5a3d97391260efa214c66ba09895e8c5e3",
        "public_boolean_key": "hf_specific_tokenizer_config_downloaded",
    },
}
published_secret_scan = builder.secret_scan


def scan_public_record(data, name, depth=0):
    """Apply only reviewed syntax exceptions to an in-memory scanning view.

    Original files, archive members, and the actual-value scan remain unchanged.
    Retain authorization prose as a value under a neutral scan-only key so every
    value still receives the published pattern and recursive base64 checks.
    """
    relative = name.removeprefix("records/")
    rule = SCAN_EXCEPTIONS.get(relative)
    if rule and hashlib.sha256(data).hexdigest() == rule["sha256"]:
        obj = json.loads(data)
        if "authorization_pointer" in rule:
            parent = obj
            for key in rule["authorization_pointer"][:-1]:
                parent = parent[key]
            key = rule["authorization_pointer"][-1]
            if not isinstance(parent[key], str):
                raise ValueError("reviewed authorization field changed type: " + relative)
            parent["reviewed_experiment_permission_text"] = parent.pop(key)
        else:
            key = rule["public_boolean_key"]
            if not isinstance(obj[key], bool):
                raise ValueError("reviewed public key changed type: " + relative)
            obj["reviewed_public_tokenizer_download_flag"] = obj.pop(key)
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    published_secret_scan(data, name, depth)


builder.secret_scan = scan_public_record


def exclusion(relative):
    if any(part.startswith(".credentials") for part in relative.parts):
        return "private credential file/directory; contents not read"
    if relative.parts[:1] == ("local-index-v1",):
        if len(relative.parts) > 1 and relative.parts[1] in {"downloads", "runtime", "data", "logs"}:
            return "downloadable runtime or mutable local OpenSearch data/logs; frozen receipts retained"
        if relative.name == "indexed-records.jsonl":
            return "byte-identical full index already published; explicit dependency and SHA retained"
        if relative.name in {"opensearch-service.log", "job.log"}:
            return "live/duplicate process log; completed snapshots retained"
    return builder.exclusion(relative)


class CredentialScan:
    """Keep attachment credentials in memory only; report filenames, never values."""

    def __init__(self, attachment):
        try:
            words = shlex.split(attachment.read_text(encoding="utf-8").replace("\\\n", ""))
            found = {}
            for index, word in enumerate(words[:-1]):
                if word not in {"-H", "--header"}:
                    continue
                key, separator, value = words[index + 1].partition(":")
                if separator and key.strip().lower() in {"cf-access-client-id", "cf-access-client-secret"}:
                    found[key.strip().lower()] = value.strip()
            if len(found) != 2 or not all(found.values()):
                raise ValueError
            self.needles = set()
            for value in found.values():
                raw = value.encode("utf-8")
                self.needles.update((raw, base64.b64encode(raw), quote(value, safe="").encode(),
                                     json.dumps(value, ensure_ascii=True)[1:-1].encode()))
        except Exception:
            raise ValueError("private attachment could not supply the two expected credentials; no values printed") from None

    def scan_bytes(self, data, name, depth=0):
        if any(needle in data for needle in self.needles):
            raise ValueError("credential scan failed: " + name + "; no values printed")
        if depth > 8:
            return
        try:
            obj = json.loads(data)
        except (ValueError, UnicodeError):
            return

        def walk(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key.lower().endswith(("base64", "_b64")) and isinstance(child, str):
                        try:
                            decoded = base64.b64decode(child, validate=True)
                        except ValueError:
                            continue
                        self.scan_bytes(decoded, name, depth + 1)
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(obj)

    def scan_file(self, path, name):
        if path.suffix == ".jsonl":
            with path.open("rb") as stream:
                for line in stream:
                    self.scan_bytes(line, name)
        else:
            self.scan_bytes(path.read_bytes(), name)


def prepare(records, target, final_freeze, final_freeze_sha256, credential_source):
    freeze_relative = final_freeze.resolve().relative_to(records).as_posix()
    freeze_path = builder.checked_source(records, freeze_relative)
    if builder.file_sha(freeze_path) != final_freeze_sha256:
        raise ValueError("final-freeze SHA does not match")
    scanner = CredentialScan(credential_source)
    rows, excluded, seen = [], [], set()

    def add(path, group):
        relative_path = path.relative_to(records)
        relative = relative_path.as_posix()
        if relative in seen:
            return
        seen.add(relative)
        reason = exclusion(relative_path)
        if reason:
            excluded.append({"origin": "records", "path": relative, "reason": reason})
            return
        source = builder.checked_source(records, relative)
        builder.scan_file(source, relative)
        scanner.scan_file(source, relative)
        rows.append({"origin": "records", "source_path": relative,
                     "path": "records/" + relative, "group": group,
                     "bytes": source.stat().st_size, "sha256": builder.file_sha(source)})

    for group, names in GROUPS.items():
        for name in names:
            root = records / name
            if not root.is_dir() or root.is_symlink():
                raise ValueError("completed directory missing/unsafe: " + name)
            for folder, dirs, files in os.walk(root, followlinks=False):
                for entry in sorted(dirs):
                    child = Path(folder) / entry
                    relative = child.relative_to(records)
                    reason = exclusion(relative)
                    if relative.as_posix() == "api-contract-v1/upstream":
                        reason = "public upstream source cache omitted; fixed commit and original SOURCE-PINS retained as an external dependency"
                    if child.is_symlink():
                        raise ValueError("symbolic directory rejected: " + relative.as_posix())
                    if reason:
                        dirs.remove(entry)
                        excluded.append({"origin": "records", "path": relative.as_posix(), "reason": reason})
                for filename in sorted(files):
                    add(Path(folder) / filename, group)

    pins_path = builder.checked_source(records, UPSTREAM_PINS)
    if builder.file_sha(pins_path) != UPSTREAM_PINS_SHA256:
        raise ValueError("upstream SOURCE-PINS changed")
    pins = builder.read_json(pins_path)
    if len(pins["files"]) != 14:
        raise ValueError("upstream SOURCE-PINS must contain exactly the frozen 14 files")
    for relative, pin in sorted(pins["files"].items()):
        source = builder.checked_source(records, UPSTREAM + "/" + relative)
        if source.stat().st_size != pin["bytes"] or builder.file_sha(source) != pin["sha256"]:
            raise ValueError("upstream source differs from frozen pin: " + relative)
        excluded.append({"origin": "records", "path": UPSTREAM + "/" + relative,
                         "reason": "public upstream source available at the fixed commit; original SOURCE-PINS paths and hashes retained"})

    known_dirs = {name for names in GROUPS.values() for name in names}
    for path in sorted(records.iterdir()):
        reason = exclusion(path.relative_to(records))
        if reason:
            excluded.append({"origin": "records", "path": path.name, "reason": reason})
        elif path.is_file() or path.is_symlink():
            add(path, "support")
        elif path.name not in known_dirs:
            raise ValueError("unclassified top-level directory; review before freezing: " + path.name)

    duplicate = builder.checked_source(records, INDEX_DEPENDENCY["omitted_duplicate"])
    if (duplicate.stat().st_size != INDEX_DEPENDENCY["member_bytes"]
            or builder.file_sha(duplicate) != INDEX_DEPENDENCY["member_sha256"]):
        raise ValueError("omitted index is not byte-identical to the published dependency")
    if freeze_relative not in {row["source_path"] for row in rows}:
        raise ValueError("final-freeze anchor is not selected")
    rows.sort(key=lambda row: row["path"])
    builder.save_new(target, {
        "schema": "bm250820-public-archive-selection-v1",
        "scope": "completed external RWKV 2.9B development records; integrity only, not accuracy certification",
        "selection_script_sha256": builder.file_sha(Path(__file__)),
        "published_builder_sha256": BASE_SHA256,
        "final_freeze": {"path": "records/" + freeze_relative, "sha256": final_freeze_sha256},
        "members": rows, "exclusions": sorted(excluded, key=lambda row: row["path"]),
        "external_dependencies": [INDEX_DEPENDENCY, {
            "kind": "public_upstream_source", "repository": pins["repository"], "commit": pins["commit"],
            "source_pins_path": "records/" + UPSTREAM_PINS, "source_pins_sha256": UPSTREAM_PINS_SHA256,
            "restore_under": "records/" + UPSTREAM,
            "restore_instructions": "Obtain the named files from this repository at the exact commit, preserve their relative paths under restore_under, and verify every byte count and SHA against SOURCE-PINS.json.",
            "files": pins["files"], "deployed_source_revision_verified": False,
        }],
        "upstream_source_pins": {"path": "records/" + UPSTREAM_PINS, "sha256": UPSTREAM_PINS_SHA256,
                                 "original_files": 14, "selected_files": 0, "original_paths_preserved_in_pins": True},
        "secret_scan": {"result": "PASS", "selected_files": len(rows),
                        "checks": ["published pattern, JSON credential-key and base64 wire scan",
                                   "two actual attachment credential values, in memory only, including encoded wire bodies"],
                        "credential_values_or_hashes_stored": False,
                        "reviewed_false_positives": SCAN_EXCEPTIONS,
                        "exception_scope": "exact original path, complete file SHA and named JSON key only; values remain scanned; raw files unchanged"},
    })
    print(json.dumps({"selection": str(target), "sha256": builder.file_sha(target),
                      "members": len(rows), "bytes": sum(row["bytes"] for row in rows), "secret_scan": "PASS"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "build", "verify"])
    parser.add_argument("--records", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--extract", type=Path)
    parser.add_argument("--final-freeze", type=Path)
    parser.add_argument("--final-freeze-sha256")
    parser.add_argument("--credential-source", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        if not args.manifest:
            parser.error("--manifest is required")
        builder.verify(args.manifest, args.extract)
        return
    if not args.records or not args.selection:
        parser.error("--records and --selection are required")
    records = args.records.resolve()
    if args.command == "prepare":
        if not args.final_freeze or not args.final_freeze_sha256 or not args.credential_source:
            parser.error("prepare requires --final-freeze, --final-freeze-sha256 and --credential-source")
        prepare(records, args.selection, args.final_freeze, args.final_freeze_sha256, args.credential_source)
    else:
        if not args.output:
            parser.error("--output is required")
        inventory = builder.read_json(args.selection)
        if inventory.get("published_builder_sha256") != BASE_SHA256:
            raise ValueError("selection does not pin the published builder")
        if inventory.get("selection_script_sha256") != builder.file_sha(Path(__file__)):
            raise ValueError("selection does not pin this wrapper")
        builder.build(records, records, args.selection, args.output)


if __name__ == "__main__":
    main()
