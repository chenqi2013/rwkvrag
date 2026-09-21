import asyncio
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "llamaindex-retrieval/src"))
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline
from llamaindex_retrieval.schemas import SourceItem
from llamaindex_retrieval.typed_funnel_v6 import Runner, extract_atomic


async def main():
    here = Path(__file__).resolve().parent
    for path, digest in json.loads((here / "PINS.json").read_text()).items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    out = ROOT / "data/quality-runs/typed-funnel-v6-20260921/atoms1"
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((here.parent / "github-natural-comparison-20260921/CONFIG.json").read_text())
    config.update(native_writer_pipeline="typed_funnel_v5", native_max_concurrency=1)
    pipeline = RWKVPipeline(Settings(**config), None)
    records = []
    try:
        for i, row in enumerate(json.loads((here / "ATOMS.json").read_text())):
            source = SourceItem(**row["source"])
            baseline = await pipeline._call(row["baseline_prompt"], stage="resolver", max_tokens=512,
                sources=[source], structured_schema=row["baseline_schema"])
            runner = Runner(pipeline)
            value = await extract_atomic(runner, source, row["object"], row["field"])
            gold = row["gold"]
            correct = value is not None and all(value[key] == expected and type(value[key]) is type(expected)
                for key, expected in gold.items())
            record = {"id": row["id"], "ordinal": i, "input": row,
                "baseline": {"status": baseline.status, "raw_text": baseline.raw_text, "trace": baseline.trace},
                "candidate": {"parsed": value, "calls": runner.calls, "failures": runner.failures},
                "candidate_exact": correct}
            with (out / f"{i:03d}.json").open("x") as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2)
            records.append({"ordinal": i, "correct": correct, "value": value})
            print(i, correct, value, flush=True)
    finally:
        await pipeline.aclose()
    with (out / "SUMMARY.json").open("x") as stream:
        json.dump({"correct": sum(r["correct"] for r in records), "total": len(records), "records": records}, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
