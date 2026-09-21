import asyncio,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'llamaindex-retrieval/src'))
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline
from llamaindex_retrieval.funnel_writer_v4 import write_funnel
from llamaindex_retrieval.funnel_final_prompt import grounded_baseline_prompt
from llamaindex_retrieval.structured_native import StructuredNativeRWKVClient
from llamaindex_retrieval.model_client import model_client_options
class StructuredPipeline(RWKVPipeline):
    async def _write(self, task, sources, fields):
        return await write_funnel(self, task, sources)
    async def _call(self, prompt, *, structured_schema=None, **kwargs):
        if structured_schema is None:
            return await super()._call(prompt, **kwargs)
        stage=kwargs['stage'];sources=kwargs.get('sources',())
        return await self.model.complete([{'role':'user','content':prompt}],max_tokens=kwargs['max_tokens'],
            stage=stage,evidence_ids=[s.id for s in sources],assistant_prefill=self.settings.native_resolver_prefill,
            temperature=1,top_p=1,top_k=1,seed=11,structured_schema=structured_schema)
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.schemas import SourceItem,ConversationMessage
async def main():
    here=Path(__file__).resolve().parent
    for name,digest in json.loads((here/'PINS.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    out=ROOT/'data/quality-runs/funnel-structured-v4-20260921/run1';out.mkdir(parents=True,exist_ok=False)
    config=json.loads((here.parent/'github-natural-comparison-20260921/CONFIG.json').read_text());config.update(native_writer_pipeline='funnel_v1',native_writer_budget_policy='disabled',native_funnel_max_calls=512)
    settings=Settings(**config);pipeline=StructuredPipeline(settings,None,model=StructuredNativeRWKVClient(**model_client_options(settings)))
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
            if graph.get('task'):
                from llamaindex_retrieval.rwkv_pipeline import conversation
                spec=graph['task'];objects={f'O{i}':name for i,name in enumerate(spec['objects'],1)};fields={f'F{i}':name for i,name in enumerate(spec['fields'],1)}
                ids={identity for cell in graph['cells'] for identity in cell.get('fact_ids',[])}
                facts=[fact for fact in graph['facts'] if fact['id'] in ids]
                labels={source.id:f'资料 {i}' for i,source in enumerate(result.sources,1)}
                task=conversation(row['question'],[ConversationMessage(**h) for h in row.get('history',[])])
                prompt=grounded_baseline_prompt(task,spec,objects,fields,graph['cells'],graph['field_summaries'],facts,labels)
                replay=await pipeline._call(prompt,stage='writer',max_tokens=2048,sources=result.sources)
                record={'id':row['id'],'status':replay.status,'raw_text':replay.raw_text,'trace':replay.trace,'sources':[s.model_dump() for s in result.sources]}
                with (out/f'{n:03d}.writer-replay.json').open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2)
                print('Writer replay',n,replay.status,flush=True)
    finally:await pipeline.aclose()
if __name__=='__main__':asyncio.run(main())
