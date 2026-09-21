import asyncio,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'llamaindex-retrieval/src'))
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline
from llamaindex_retrieval.funnel_writer_v2 import write_funnel
class AtomicPipeline(RWKVPipeline):
    async def _write(self, task, sources, fields):
        return await write_funnel(self, task, sources)
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.schemas import SourceItem,ConversationMessage
async def main():
    here=Path(__file__).resolve().parent
    for name,digest in json.loads((here/'PINS.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    out=ROOT/'data/quality-runs/funnel-atomic-v2-20260921/run1';out.mkdir(parents=True,exist_ok=False)
    config=json.loads((here.parent/'github-natural-comparison-20260921/CONFIG.json').read_text());config.update(native_writer_pipeline='funnel_v1',native_writer_budget_policy='disabled',native_funnel_max_calls=512)
    pipeline=AtomicPipeline(Settings(**config),None)
    rows=[]
    for n in range(8):
        p=ROOT/'data/quality-runs/github-natural-comparison-20260921/run1'/f'{n:02d}.json';r=json.loads(p.read_text());rows.append({'id':f'natural/{n:02d}',**r['request'],'sources':r['response']['sources'],'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    for row in json.loads((here.parent/'model-size-paired-20260921/INPUTS.json').read_text())[:28]:
        rows.append({'id':row['id'],'question':row['question'],'history':row['history'],'sources':[SourceItem(id=f's{i}',document_id=f'd{i}',source='fixed-regression',title=e.get('title',''),snippet=e['text'],score=1,metadata={}).model_dump() for i,e in enumerate(row['evidence'],1)]})
    (out/'INPUTS.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    try:
        for n,row in enumerate(rows):
            result=await pipeline.ask_materials(row['question'],[SourceItem(**s) for s in row['sources']],history=[ConversationMessage(**h) for h in row.get('history',[])])
            with (out/f'{n:03d}.json').open('x') as f:json.dump({'id':row['id'],'ordinal':n,'response':result.model_dump()},f,ensure_ascii=False,indent=2)
            graph=result.retrieval.get('funnel',{});print(n,row['id'],result.generation['status'],len(graph.get('failures',[])),len(result.generation['model_calls']),flush=True)
    finally:await pipeline.aclose()
if __name__=='__main__':asyncio.run(main())
