"""Read-only top-20 BM25 recall on all 434 restored-corpus questions."""

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import time

from opensearchpy import OpenSearch

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.lexical_index import LexicalIndex


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.parent / "restored-retrieval-v2-20260920"
SETTINGS = ROOT / "data/services/restored-regression-20260920-v2/settings.json"
OUT = ROOT / "data/quality-runs/live-retrieval-complete-20260923-v1/kb-run1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save_new(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main():
    cases_path = OLD / "cases.json"
    coverage_path = OLD / "COVERAGE.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    expected = {row["case_id"]: row for row in coverage["cases"]}
    assert len(cases) == len(expected) == 434
    local = json.loads(SETTINGS.read_text(encoding="utf-8"))
    physical = local["opensearch_index"]
    client = OpenSearch(hosts=[local["opensearch_url"]], timeout=30, max_retries=0,
                        retry_on_timeout=False)
    alias = physical + "-active"
    aliases = client.indices.get_alias(name=alias)
    if len(aliases) != 1 or physical not in aliases:
        raise RuntimeError("restored_alias_changed")
    before = client.count(index=physical)["count"]
    if before != 58594:
        raise RuntimeError("restored_corpus_count_changed")
    settings = Settings(opensearch_index=physical, opensearch_url=local["opensearch_url"], _env_file=None)
    index = LexicalIndex(settings, client=client, initialize=False)
    index.index_name = physical
    OUT.mkdir(parents=True, exist_ok=True)
    binding = {"protocol": "restored-retrieval-readonly-20260923-v1",
               "started_at": datetime.now(timezone.utc).isoformat(),
               "cases_sha256": sha(cases_path.read_bytes()),
               "coverage_sha256": sha(coverage_path.read_bytes()),
               "settings_sha256": sha(SETTINGS.read_bytes()),
               "runner_sha256": sha(Path(__file__).read_bytes()),
               "index": physical, "index_count": before,
               "candidate_k": 20, "operation": "search_chunks_only"}
    if (OUT / "BINDING.json").exists():
        old = json.loads((OUT / "BINDING.json").read_text(encoding="utf-8"))
        for key in ("protocol", "cases_sha256", "coverage_sha256", "settings_sha256", "runner_sha256", "index", "index_count", "candidate_k"):
            if binding[key] != old[key]:
                raise RuntimeError(f"resume_binding_changed: {key}")
    else:
        save_new(OUT / "BINDING.json", binding)
    counts = Counter()
    for ordinal, case in enumerate(cases):
        path = OUT / f"{ordinal:04d}.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
        else:
            started = time.monotonic()
            target = expected[case["id"]]
            row = {"ordinal": ordinal, "case_id": case["id"], "suite": case["suite"],
                   "question": case["payload"]["question"],
                   "knowledge_base_id": case["payload"]["knowledge_base_id"],
                   "expected_document_ids": target["actual_document_ids"],
                   "expected_titles": target["expected_titles"]}
            try:
                hits = index.search_chunks(row["question"], candidate_k=20,
                    knowledge_base_id=row["knowledge_base_id"])
                ids = [hit.document_id for hit in hits]
                row["hit_rank"] = next((i for i, item in enumerate(ids, 1)
                    if item in row["expected_document_ids"]), None)
                row["hits"] = [{"rank": i, "document_id": hit.document_id,
                    "title": hit.metadata.get("title"), "node_id": hit.node_id,
                    "score": hit.score, "text_sha256": sha(hit.text.encode()),
                    "preview": hit.text[:160]} for i, hit in enumerate(hits, 1)]
                row["status"] = "recorded"
            except Exception as error:
                row["status"] = "failed"
                row["error_type"] = type(error).__name__
            row["elapsed_ms"] = round((time.monotonic() - started) * 1000)
            save_new(path, row)
        counts[row["status"]] += 1
        counts["at_5"] += row.get("hit_rank") is not None and row["hit_rank"] <= 5
        counts["at_20"] += row.get("hit_rank") is not None and row["hit_rank"] <= 20
        if (ordinal + 1) % 50 == 0 or ordinal + 1 == len(cases):
            print("completed", ordinal + 1, "at5", counts["at_5"],
                  "at20", counts["at_20"], "failed", counts["failed"], flush=True)
    after = client.count(index=physical)["count"]
    summary = {"planned": len(cases), "counts": dict(counts),
               "index_count_after": after,
               "finished_at": datetime.now(timezone.utc).isoformat()}
    if not (OUT / "SUMMARY.json").exists():
        save_new(OUT / "SUMMARY.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
