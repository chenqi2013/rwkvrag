import asyncio
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "llamaindex-retrieval/src"))
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline, conversation
from llamaindex_retrieval.schemas import ConversationMessage
from llamaindex_retrieval.typed_funnel_v9 import Runner, empty_evidence_prompt
from llamaindex_retrieval.funnel_task_v4 import build_task


async def main():
    here = Path(__file__).resolve().parent
    for path, digest in json.loads((here / "PINS.json").read_text()).items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    baseline = ROOT / "data/quality-runs/typed-funnel-integration-20260921/run1"
    assert len(list(baseline.glob("[0-9][0-9][0-9].json"))) == 36
    out = ROOT / "data/quality-runs/funnel-task-query-last-20260921/run1"
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((here.parent / "github-natural-comparison-20260921/CONFIG.json").read_text())
    config.update(native_writer_pipeline="typed_funnel_v8", native_max_concurrency=1)
    pipeline = RWKVPipeline(Settings(**config), None)
    rows = json.loads((here / "INPUTS.json").read_text())
    try:
        for i, row in enumerate(rows):
            runner = Runner(pipeline)
            users = [h["content"] for h in row["history"] if h["role"] == "user"] + [row["question"]]
            task = await build_task(runner, {f"U{j}": text for j, text in enumerate(users, 1)})
            with (out / f"{i:03d}.json").open("x") as stream:
                json.dump({"id": row["id"], "task": task, "calls": runner.calls, "failures": runner.failures}, stream, ensure_ascii=False, indent=2)
            print(i, row["id"], "task_valid", task is not None, "failures", len(runner.failures), flush=True)
    finally:
        await pipeline.aclose()


if __name__ == "__main__":
    asyncio.run(main())
