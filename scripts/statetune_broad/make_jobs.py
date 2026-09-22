"""Freeze source groups before generation. Text remains byte-exact; no teacher labels here."""
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import zlib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "llamaindex-retrieval/statetune/defect-batch-20260922"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def shingles(text):
    clean = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).lower())
    clean = re.sub(r"\d+", "#", clean)
    return {zlib.crc32(clean[i:i+5].encode()) for i in range(max(1, len(clean)-4))}


def cluster(rows):
    """LSH candidates followed by exact shingle Jaccard. Not semantic deduplication."""
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    rng = np.random.default_rng(20260922)
    a = rng.integers(1, 2**31, size=32, dtype=np.uint64)
    b = rng.integers(0, 2**31, size=32, dtype=np.uint64)
    buckets, sets, near_pairs = defaultdict(list), [], []
    for i, row in enumerate(rows):
        tokens = shingles(row["text"])
        vals = np.array(sorted(tokens), dtype=np.uint64)
        sig = ((vals[:, None] * a + b) % np.uint64(4294967311)).min(axis=0)
        candidates = set()
        keys = [(j, tuple(int(x) for x in sig[j*4:j*4+4])) for j in range(8)]
        for k in keys:
            candidates.update(buckets[k])
        for j in candidates:
            similarity = len(tokens & sets[j]) / len(tokens | sets[j])
            if similarity >= .75:
                parent[find(i)] = find(j)
                near_pairs.append([rows[j]["page_id"], row["page_id"], similarity])
        for k in keys:
            buckets[k].append(i)
        sets.append(tokens)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[find(i)].append(r)
    return list(groups.values()), near_pairs


def walk_titles(value):
    if isinstance(value, dict):
        for key, v in value.items():
            if key in {"expected_title", "title", "source_title"} and isinstance(v, str):
                yield v
            yield from walk_titles(v)
    elif isinstance(value, list):
        for v in value:
            yield from walk_titles(v)


def main():
    corpus = ROOT / "data/corpora/finewiki-zh-5000"
    prior = ROOT / "llamaindex-retrieval/eval/restored-retrieval-v2-20260920/cases.json"
    excluded = set(walk_titles(json.loads(prior.read_text())))
    rows, rejected, hashes = [], Counter(), {}
    for line in (corpus / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["title"] in excluded:
            rejected["known_regression_title"] += 1
            continue
        if not 300 <= int(row["text_characters"]) <= 12000:
            rejected["outside_whole_document_length"] += 1
            continue
        path = corpus / row["text_path"]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["text_sha256"]
        text = raw.decode()
        if row["text_sha256"] in hashes:
            rejected["exact_duplicate"] += 1
            continue
        hashes[row["text_sha256"]] = row["page_id"]
        rows.append({"page_id": str(row["page_id"]), "title": row["title"], "text": text,
            "url": row["revision_url"], "license": row["license"], "sha256": row["text_sha256"],
            "source_path": str(path.relative_to(ROOT))})
    groups, near = cluster(rows)
    pools = defaultdict(list)
    for group in groups:
        ids = sorted(r["page_id"] for r in group)
        family = sha("wiki-family:" + ",".join(ids))
        bucket = int(family[:8], 16) % 10
        split = "dev" if bucket == 8 else "holdout" if bucket == 9 else "train"
        representative = min(group, key=lambda r: sha("representative:" + r["page_id"]))
        pools[split].append(dict(representative, source_family=family, related_page_ids=ids))
    jobs = []
    for split, count in [("train", 512), ("dev", 64), ("holdout", 64)]:
        pool = sorted(pools[split], key=lambda r: sha("selection:" + r["page_id"]))
        real_count = count * 5 // 8
        needed = real_count * 2 + count - real_count
        if len(pool) < needed:
            raise ValueError(f"Not enough distinct {split} source groups: {len(pool)} < {needed}")
        cursor = 0
        for i in range(count):
            real = i < real_count
            docs = pool[cursor:cursor+(2 if real else 1)]
            cursor += len(docs)
            identity = sha(split + ":" + ",".join(d["source_family"] for d in docs))[:20]
            jobs.append({"id": identity, "split": split, "material_mode": "real" if real else "synthetic",
                "source_families": [d["source_family"] for d in docs],
                "sources": [dict(d, id=f"S{n}") for n, d in enumerate(docs, 1)],
                "defect_evidence": "statetune-defects-20260922", "status": "unlabelled"})
    # Interleave modes/splits for pilot inspection; allocation is already frozen.
    jobs.sort(key=lambda j: sha("execution:" + j["id"]))
    with (OUT / "JOBS.jsonl").open("x") as stream:
        for job in jobs:
            stream.write(json.dumps(job, ensure_ascii=False) + "\n")
    report = {"documents_after_length_exact_and_known_title_checks": len(rows), "source_clusters": len(groups),
        "near_duplicate_pairs": near, "rejected": dict(rejected), "jobs": len(jobs),
        "by_split": dict(Counter(j["split"] for j in jobs)),
        "by_mode": dict(Counter(j["material_mode"] for j in jobs)),
        "used_source_families": len({f for j in jobs for f in j["source_families"]}),
        "known_regression_cases_sha256": hashlib.sha256(prior.read_bytes()).hexdigest(),
        "limits": "Known 434-case title exclusion and exact/LSH lexical checks only; no claim of exhaustive historical or semantic decontamination. No generated tasks have been admitted."}
    with (OUT / "SOURCE-AUDIT.json").open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print({k:v for k,v in report.items() if k not in {"near_duplicate_pairs", "limits"}})


if __name__ == "__main__":
    main()
