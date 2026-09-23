"""Index exact README text blocks for later evidence-label annotation.

This is a source bank, not an evidence judgment dataset. Every recorded block
round-trips to the pinned README bytes and carries its heading context.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")
LIST = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+")
FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})")


def blocks(raw):
    text = raw.decode("utf-8")
    lines = text.splitlines(keepends=True)
    positions, offset = [], 0
    for line in lines:
        encoded = line.encode("utf-8")
        positions.append((offset, offset + len(encoded), line))
        offset += len(encoded)
    headings = []
    current, kind = [], None
    output = []
    fence_marker = None

    def flush():
        nonlocal current, kind
        if not current:
            return
        start, end = current[0][0], current[-1][1]
        chunk = raw[start:end]
        if not chunk.strip():
            current, kind = [], None
            return
        output.append({"start_byte": start, "end_byte": end,
                       "kind": kind, "heading_path": headings.copy(),
                       "text": chunk.decode("utf-8"),
                       "sha256": sha256(chunk).hexdigest()})
        current, kind = [], None

    for start, end, line in positions:
        stripped = line.strip()
        fence = FENCE.match(line)
        if fence_marker:
            current.append((start, end))
            if fence and fence.group(1)[0] == fence_marker[0] and len(fence.group(1)) >= len(fence_marker):
                flush()
                fence_marker = None
            continue
        if fence:
            flush()
            kind = "code"
            current.append((start, end))
            fence_marker = fence.group(1)
            continue
        heading = HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            headings = headings[:level - 1] + [heading.group(2).strip()]
            continue
        if not stripped:
            flush()
            continue
        next_kind = ("table" if line.count("|") >= 2 else
                     "list" if LIST.match(line) else "paragraph")
        if current and kind != next_kind:
            flush()
        kind = next_kind
        current.append((start, end))
    flush()
    for row in output:
        exact = raw[row["start_byte"]:row["end_byte"]]
        if exact.decode("utf-8") != row["text"] or sha256(exact).hexdigest() != row["sha256"]:
            raise ValueError("block byte span does not round-trip")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    sources_raw = args.sources.read_bytes()
    sources = json.loads(sources_raw)
    if sources.get("schema") not in {"retrieval_curated_sources_v3", "retrieval_curated_sources_v4"}:
        raise ValueError("source schema mismatch")
    stats, count, maximum = Counter(), 0, 0
    with args.out.open("x", encoding="utf-8") as destination:
        for source in sources["sources"]:
            raw = (ROOT / source["readme_path"]).read_bytes()
            if sha256(raw).hexdigest() != source["sha256"]:
                raise ValueError(f"source snapshot changed: {source['repo']}")
            for index, item in enumerate(blocks(raw)):
                row = {"id": f"{source['family']}:{index}",
                       "repo": source["repo"], "family": source["family"],
                       "split": source["split"], "cohort": source["cohort"],
                       "source_sha256": source["sha256"], **item}
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats[(source["split"], item["kind"])] += 1
                count += 1
                maximum = max(maximum, item["end_byte"] - item["start_byte"])
    manifest = {"schema": "retrieval_evidence_block_bank_v1",
                "source_manifest_sha256": sha256(sources_raw).hexdigest(),
                "block_file_sha256": sha256(args.out.read_bytes()).hexdigest(),
                "sources": len(sources["sources"]), "blocks": count,
                "counts": {f"{split}:{kind}": value for (split, kind), value in sorted(stats.items())},
                "max_block_bytes": maximum,
                "judgments_labeled": 0, "training_exported": False,
                "limits": "Exact README spans and heading context only; no answerability, relevance or citation semantics."}
    with args.manifest.open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(json.dumps({"sources": manifest["sources"], "blocks": count,
                      "counts": manifest["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
