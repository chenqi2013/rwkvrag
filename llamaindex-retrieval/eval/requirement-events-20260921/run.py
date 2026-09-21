import asyncio
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'llamaindex-retrieval/src'))
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline
from llamaindex_retrieval.typed_funnel_v9 import Runner
from llamaindex_retrieval.requirement_events import interpret_event


async def main():
    here = Path(__file__).resolve().parent
    for path, digest in json.loads((here / 'PINS.json').read_text()).items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    out = ROOT / 'data/quality-runs/requirement-events-20260921/run1'
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((here.parent / 'github-natural-comparison-20260921/CONFIG.json').read_text())
    config.update(native_writer_pipeline='typed_funnel_v8', native_max_concurrency=1)
    pipeline = RWKVPipeline(Settings(**config), None)
    try:
        for i, row in enumerate(json.loads((here / 'INPUTS.json').read_text())):
            runner = Runner(pipeline)
            event = await interpret_event(runner, row['quote'])
            correct = event is not None and event['action'] in row['allowed_actions']
            with (out / f'{i:03d}.json').open('x') as stream:
                json.dump({'input': row, 'event': event, 'action_correct': correct, 'calls': runner.calls,
                           'failures': runner.failures}, stream, ensure_ascii=False, indent=2)
            print(i, correct, event, flush=True)
    finally:
        await pipeline.aclose()


if __name__ == '__main__':
    asyncio.run(main())
