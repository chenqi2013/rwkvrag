"""Isolated real OpenSearch + Planner/Reader/Writer comparison, no business writes."""
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variable", choices=["native_preserve_original_question",
        "native_planner_format_repair", "reader_without_state"], default="native_preserve_original_question")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw_settings = args.settings.read_bytes()
    config = json.loads(raw_settings)
    config["opensearch_index"] = "rwkvrag-comparison-eval-20260919"
    rows = [json.loads(line) for line in (Path(__file__).parent / "development.jsonl").read_text().splitlines()][:2]
    settings = Settings(**config)
    index = LexicalIndex(settings)
    nodes = [TextNode(id_=f"{row['id']}-{m['id']}", text=m["snippet"], metadata={
        "document_id": m["document_id"], "title": m["title"], "source": "comparison-eval",
        "knowledge_base_id": row["id"], "content_type": "prose"})
        for row in rows for m in row["materials"]]
    index.upsert_nodes(nodes)
    index.client.indices.refresh(index=index.index_name)
    manifest = {"production_settings_sha256": hashlib.sha256(raw_settings).hexdigest(),
        "index": settings.opensearch_index, "materials": rows,
        "variable": args.variable, "business_index_modified": False,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    try:
        for enabled in (False, True):
            overrides = ({"rwkvos_binary_reader_state_id": None} if enabled else {}) if args.variable == "reader_without_state" else {args.variable: enabled}
            pipeline = RWKVPipeline(settings.model_copy(update=overrides), index)
            try:
                for row in rows:
                    result = await pipeline.ask(SearchRequest(question=row["question"],
                        retrieval_mode="knowledge_base", knowledge_base_id=row["id"]))
                    path = args.output / f"{row['id']}-{args.variable}-{str(enabled).lower()}.json"
                    path.write_text(result.model_dump_json(indent=2))
                    print(path.name, result.generation["status"], flush=True)
            finally:
                await pipeline.aclose()
    finally:
        index.close()


if __name__ == "__main__":
    asyncio.run(main())
