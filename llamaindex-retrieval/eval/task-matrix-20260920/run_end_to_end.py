"""Real retrieval on a dedicated synthetic index; no writes to business knowledge."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from llama_index.core.schema import TextNode
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.lexical_index import LexicalIndex
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline
from llamaindex_retrieval.schemas import SearchRequest


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--planner-state", default="planner-matrix-v2")
    parser.add_argument("--reader-state", default="reader-matrix-v1")
    parser.add_argument("--writer-state", default="writer-matrix-v2")
    parser.add_argument("--review-state", default="reader-matrix-review-v2")
    parser.add_argument("--assessment-state", default="planner-matrix-v1")
    parser.add_argument("--followup-state", default="planner-matrix-v1")
    parser.add_argument("--concurrency", type=int, choices=[1, 2], default=1)
    parser.add_argument("--answer-repairs", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases_hash = hashlib.sha256(args.cases.read_bytes()).hexdigest()
    config = json.loads(args.settings.read_text())
    config.update(native_base_url=args.base_url, native_task_matrix_enabled=True,
        native_matrix_answer_repairs=args.answer_repairs,
        rwkvos_matrix_state_ids={"plan": args.planner_state, "reader": args.reader_state, "writer": args.writer_state,
            "assessment": args.assessment_state, "followup": args.followup_state, "review": args.review_state},
        opensearch_index=f"rwkvrag-task-matrix-eval-20260920-{cases_hash[:8]}")
    settings = Settings(**config)
    rows = [json.loads(line) for line in args.cases.read_text().splitlines()]
    index = LexicalIndex(settings)
    nodes = [TextNode(id_=s["metadata"].get("parent_source_id", s["id"]), text=s["snippet"], metadata={
        "document_id": s["document_id"], "title": s["title"], "source": "synthetic-matrix",
        "knowledge_base_id": case["id"], "content_type": "prose"})
        for case in rows for s in case.get("candidates", case.get("materials", []))]
    index.upsert_nodes(nodes)
    index.refresh()
    (args.output / "MANIFEST.json").write_text(json.dumps({"cases_sha256": cases_hash,
        "index_version": index.versions.current(), "states": [args.planner_state, args.reader_state, args.writer_state, args.review_state,
            args.assessment_state, args.followup_state],
        "synthetic": True, "business_index_modified": False, "native_task_matrix_enabled": True,
        "concurrency": args.concurrency, "answer_repairs": args.answer_repairs}, indent=2))
    pipeline = RWKVPipeline(settings, index)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def run(row):
        async with semaphore:
            result = await pipeline.ask(SearchRequest(question=row["question"], history=row["history"],
                retrieval_mode="knowledge_base", knowledge_base_id=row["id"]))
            (args.output / f"{row['id']}.json").write_text(result.model_dump_json(indent=2))
            print(row["id"], result.generation["status"], len(result.sources), flush=True)

    try:
        await asyncio.gather(*(run(row) for row in rows))
    finally:
        await pipeline.aclose()
        index.close()


if __name__ == "__main__":
    asyncio.run(main())
