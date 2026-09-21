import asyncio, hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'llamaindex-retrieval/src'))
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline


async def main():
    here=Path(__file__).resolve().parent
    for path,digest in json.loads((here/'PINS.json').read_text()).items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    out=ROOT/'data/quality-runs/requirement-grammar-probe-20260921/run1';out.mkdir(parents=True,exist_ok=False)
    config=json.loads((here.parent/'github-natural-comparison-20260921/CONFIG.json').read_text())
    config.update(native_writer_pipeline='typed_funnel_v8',native_max_concurrency=1)
    pipeline=RWKVPipeline(Settings(**config),None)
    try:
        for row in json.loads((here/'INPUTS.json').read_text()):
            result=await pipeline._call(row['messages'][0]['content'],stage='resolver',max_tokens=256)
            record={**row,'status':result.status,'raw_text':result.raw_text,'trace':result.trace}
            with (out/f"{row['ordinal']:03d}.json").open('x') as stream:json.dump(record,stream,ensure_ascii=False,indent=2)
            print(row['ordinal'],result.status,result.raw_text,flush=True)
    finally:await pipeline.aclose()
if __name__=='__main__':asyncio.run(main())
