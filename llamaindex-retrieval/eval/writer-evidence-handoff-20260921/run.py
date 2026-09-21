import asyncio, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'llamaindex-retrieval/src'))
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline
from llamaindex_retrieval.schemas import SourceItem

async def main():
    here=Path(__file__).resolve().parent
    for path,digest in json.loads((here/'PINS.json').read_text()).items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    out=ROOT/'data/quality-runs/writer-evidence-handoff-20260921/run1';out.mkdir(parents=True,exist_ok=False)
    pipeline=RWKVPipeline(Settings(**json.loads((here/'CONFIG.json').read_text())),None)
    rows=json.loads((here/'INPUTS.json').read_text());records=[]
    try:
        for rnd in [1,2]:
            for row in rows:
                for arm in (['baseline','candidate'] if rnd==1 else ['candidate','baseline']):
                    result=await pipeline._call(row[arm+'_prompt'],stage='writer',max_tokens=2048,
                        sources=[SourceItem(**s)for s in row['sources']])
                    same_input=arm=='baseline' or row['baseline_prompt']==row['candidate_prompt']
                    aligned=(result.trace.get('prompt_sha256')==row['previous_wire_prompt_sha256']
                        and result.trace.get('input_token_ids')==row['previous_input_token_ids']) if same_input else None
                    record={'ordinal':row['ordinal'],'id':row['id'],'round':rnd,'arm':arm,
                        'status':result.status,'raw_text':result.raw_text,'trace':result.trace,
                        'baseline_input_aligned':aligned,'flow':row['flow']}
                    with (out/f"{row['ordinal']:03d}-{arm}-r{rnd}.json").open('x')as f:json.dump(record,f,ensure_ascii=False,indent=2)
                    records.append({k:record[k]for k in ['ordinal','id','round','arm','status','baseline_input_aligned']})
                    print(records[-1],flush=True)
                    if aligned is False:raise RuntimeError('Frozen baseline prompt/token mismatch; record retained')
    finally:
        await pipeline.aclose()
        with (out/'SUMMARY.json').open('x')as f:json.dump({'recorded':len(records),'planned':144,'records':records,'semantic_review':'pending'},f,ensure_ascii=False,indent=2)
if __name__=='__main__':asyncio.run(main())
