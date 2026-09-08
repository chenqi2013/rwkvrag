#!/usr/bin/env python3
"""Freeze and package the completed follow-up experiments using the existing verifier."""
import argparse
import importlib.util
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1] / "bm250820-rebuild-20260908" / "build_archive.py"
spec = importlib.util.spec_from_file_location("public_archive_builder", BASE)
builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)

GROUPS = {
    "readers": ["resolver-closed-prefill-experiment", "resolver-task-units-experiment"],
    "planners": ["planner-task-contract-review", "planner-shared-tasks-experiment",
                 "planner-shared-tasks-v2", "planner-existing-queries-scope-review"],
    "writers": ["materials-writer-closed-prefill", "materials-writer-citation-suffix"],
    "wiki": ["wiki-smoke-v3", "wiki-smoke-v4", "wiki-smoke-v5", "wiki-smoke-v6",
             "source-wiki-v4", "source-wiki-v5", "source-wiki-v6",
             "source-wiki-v4-preparation-incomplete", "review/wiki-smoke-v3-preparation", "review/wiki-smoke-v2-v6-summary",
             "wiki-preflight-v4", "wiki-preflight-v5", "wiki-preflight-v6", "wiki-preflight-v7",
             "review/wiki-smoke-v3", "review/wiki-smoke-v4", "review/wiki-smoke-v5", "review/wiki-smoke-v6"],
    "diagnostics": ["candidate-scheduling-diagnostic"],
}


# Explicit completed-run support files. These are not selected by the run-name
# prefix loop below; omitting them would lose the control and reproduction path.
FILES = {
    "wiki": [
        "audit-wiki-smoke-v3.py", "audit-wiki-smoke-v4.py",
        "audit-wiki-smoke-v5.py", "audit-wiki-smoke-v6.py",
        "audit-wiki-reader-v3-control.py", "audit-wiki-task-source-v4-control.py",
        "audit-wiki-candidate-order-v5-control.py", "audit-wiki-source-budget-v6-control.py",
        "run-wiki-smoke-v3-server.py", "run-wiki-smoke-v4-server.py",
        "run-wiki-smoke-v5-server.py", "run-wiki-smoke-v6-server.py",
        "wiki-reader-v4-SOURCE-CONTROL.json",
        "wiki-task-source-v5-SOURCE-CONTROL.json", "wiki-task-source-v5.diff",
        "wiki-candidate-order-v6-SOURCE-CONTROL.json", "wiki-candidate-order-v6.diff",
        "wiki-source-v4-preparation-correction.json",
        "review/build_wiki_smoke_v3_review.py", "review/build_wiki_smoke_v4_review.py",
        "review/build_wiki_smoke_v5_review.py", "review/build_wiki_smoke_v6_review.py",
        "review/inspect_wiki_v3.py", "review/summarize_wiki_v2_v6.py",
        "wiki-preflight-v4-process.json", "wiki-preflight-v5-process.json",
        "wiki-preflight-v6-process.json", "wiki-preflight-v7-process.json",
        "wiki-smoke-v3.log", "wiki-smoke-v4.log", "wiki-smoke-v5.log", "wiki-smoke-v6.log",
        "wiki-v4-venv-RESULT.json", "wiki-v5-venv-RESULT.json", "wiki-v6-venv-RESULT.json",
    ],
    "checks": [
        "current-pytest-publication-v1.txt", "pytest-comparison-publication-v1.json",
        "current-pytest-publication-v2.txt", "pytest-comparison-publication-v2.json",
        "current-pytest-publication-v3.txt", "pytest-comparison-publication-v3.json",
    ],
}


def prepare(records, target):
    rows, excluded, seen = [], [], set()

    def add(path, group):
        relative = path.relative_to(records).as_posix()
        if relative in seen:
            return
        seen.add(relative)
        reason = builder.exclusion(path.relative_to(records))
        if reason:
            excluded.append({"origin": "records", "path": relative, "reason": reason})
            return
        source = builder.checked_source(records, relative)
        builder.scan_file(source, relative)
        rows.append({"origin": "records", "source_path": relative,
                     "path": "records/" + relative, "group": group,
                     "bytes": source.stat().st_size, "sha256": builder.file_sha(source)})

    for group, names in GROUPS.items():
        for name in names:
            root = records / name
            if not root.is_dir() or root.is_symlink():
                raise ValueError("completed directory missing/unsafe: " + name)
            for path in sorted(root.rglob("*")):
                if path.is_file() or path.is_symlink():
                    add(path, group)
    for version in range(3, 7):
        for prefix in [f"wiki-smoke-v{version}-", f"WIKI-SMOKE-V{version}-"]:
            for path in sorted(records.glob(prefix + "*")):
                if path.is_file():
                    add(path, "wiki")
    for group, names in FILES.items():
        for name in names:
            add(records / name, group)
    rows.sort(key=lambda row: row["path"])
    builder.save_new(target, {"schema": "bm250820-public-archive-selection-v1",
        "scope": "completed follow-up development experiments; no accuracy certification",
        "selection_script_sha256": builder.file_sha(Path(__file__)),
        "members": rows, "exclusions": excluded,
        "secret_scan": "pattern, structured credential keys and base64 wire checks passed"})
    print({"members": len(rows), "bytes": sum(row["bytes"] for row in rows)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "build", "verify"])
    parser.add_argument("--records", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--extract", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        if not args.manifest:
            parser.error("--manifest is required")
        builder.verify(args.manifest, args.extract)
    else:
        if not args.records or not args.selection:
            parser.error("--records and --selection are required")
        records = args.records.resolve()
        if args.command == "prepare":
            prepare(records, args.selection)
        else:
            if not args.output:
                parser.error("--output is required")
            builder.build(records, records, args.selection, args.output)


if __name__ == "__main__":
    main()
