import asyncio
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "llamaindex-retrieval/src"))
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline
from llamaindex_retrieval.schemas import SourceItem, ConversationMessage


async def main():
    here = Path(__file__).resolve().parent
    for path, digest in json.loads((here / "PINS.json").read_text()).items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    out = ROOT / "data/quality-runs/typed-funnel-v10-integration-20260921/run1"
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((here.parent / "github-natural-comparison-20260921/CONFIG.json").read_text())
    config.update(native_writer_pipeline="typed_funnel_v10", native_funnel_max_calls=512)
    pipeline = RWKVPipeline(Settings(**config), None)
    records = []
    try:
        for i, row in enumerate(json.loads((here / "INPUTS.json").read_text())):
            response = await pipeline.ask_materials(row["question"], [SourceItem(**s) for s in row["sources"]],
                history=[ConversationMessage(**h) for h in row.get("history", [])])
            with (out / f"{i:03d}.json").open("x") as stream:
                json.dump({"id": row["id"], "ordinal": i, "response": response.model_dump()}, stream, ensure_ascii=False, indent=2)
            graph = response.retrieval.get("funnel", {})
            record = {"ordinal": i, "id": row["id"], "status": response.generation["status"],
                "node_failures": len(graph.get("failures", [])), "calls": len(response.generation["model_calls"])}
            records.append(record)
            print(record, flush=True)
    finally:
        await pipeline.aclose()
        with (out / "SUMMARY.json").open("x") as stream:
            json.dump({"records": records, "total": len(records), "semantic_review": "pending"}, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
