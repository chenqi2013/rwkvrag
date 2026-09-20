"""Real index + Reader test with frozen oracle queries, independent of Planner/Writer."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from llama_index.core.schema import TextNode
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.lexical_index import LexicalIndex
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline, conversation, source_from_hit
from llamaindex_retrieval.schemas import ConversationMessage, SearchRequest


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--reader-state", default="reader-matrix-v1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases_hash = hashlib.sha256(args.cases.read_bytes()).hexdigest()
    config = json.loads(args.settings.read_text())
    config.update(native_base_url=args.base_url, rwkvos_binary_reader_state_id=args.reader_state,
        opensearch_index=f"rwkvrag-task-matrix-eval-20260920-{cases_hash[:8]}")
    settings = Settings(**config)
    rows = [json.loads(line) for line in args.cases.read_text().splitlines()]
    index = LexicalIndex(settings)
    index.upsert_nodes([TextNode(id_=s["metadata"]["parent_source_id"], text=s["snippet"], metadata={
        "document_id": s["document_id"], "title": s["title"], "source": "synthetic-matrix",
        "knowledge_base_id": row["id"], "content_type": "prose"})
        for row in rows for s in row["candidates"]])
    index.refresh()
    pipeline = RWKVPipeline(settings, index)
    results = []
    try:
        for row in rows:
            task = conversation(row["question"], [ConversationMessage(**m) for m in row["history"]])
            cells = row["matrix"]
            groups, trace = await pipeline.retrieve_groups(SearchRequest(question=row["question"],
                knowledge_base_id=row["id"], retrieval_mode="knowledge_base"),
                [c["question"] for c in cells], settings.candidate_k)
            for cell, group in zip(cells, groups, strict=True):
                candidates = [source_from_hit(h) for h in group[:settings.native_matrix_sources_per_cell]]
                selected, events = await pipeline._resolve(task, [cell["question"]], candidates)
                expected, actual = set(cell["source_ids"]), {s.id for s in selected}
                record = {"case_id": row["id"], "cell_id": cell["id"], "oracle_query": cell["question"],
                    "expected_source_ids": sorted(expected), "selected_source_ids": sorted(actual),
                    "missing": sorted(expected - actual), "extra": sorted(actual - expected),
                    "passed": expected == actual, "sources": [s.model_dump() for s in selected],
                    "candidates": [s.model_dump() for s in candidates], "retrieval": trace, "model_calls": events}
                (args.output / f"{row['id']}-{cell['id']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
                results.append(record)
            print(row["id"], sum(r["passed"] for r in results if r["case_id"] == row["id"]), len(cells), flush=True)
    finally:
        await pipeline.aclose()
        index.close()
    summary = {"cases_sha256": cases_hash, "oracle_queries": True, "planner_tested": False,
        "writer_tested": False, "reader_state": args.reader_state, "index": settings.opensearch_index,
        "cells": len(results), "passed": sum(r["passed"] for r in results),
        "missing": sum(len(r["missing"]) for r in results), "extra": sum(len(r["extra"]) for r in results)}
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
