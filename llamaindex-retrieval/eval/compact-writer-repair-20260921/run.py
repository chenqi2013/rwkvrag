import asyncio,json,hashlib,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'llamaindex-retrieval/src'))
from llamaindex_retrieval.native_rwkv import NativeRWKVClient
from llamaindex_retrieval.compact_writer import compact_writer_prompt
from llamaindex_retrieval.citation_audit import audit_citations
async def main():
    here=Path(__file__).resolve().parent
    for name,digest in json.loads((here/'PINS.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    out=ROOT/'data/quality-runs/compact-writer-repair-20260921/run1';out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for n in range(8):
        source=ROOT/'data/quality-runs/writer-budget-repair-20260921/run1'/f'{n:02d}.json'
        budget=json.loads(source.read_text());original=json.loads((ROOT/'data/quality-runs/github-natural-comparison-20260921/run1'/f'{n:02d}.json').read_text())
        request=original['request'];task=json.dumps({'history':request.get('history',[]),'latest_question':request['question']},ensure_ascii=False)
        ev=[{'label':f'资料 {i}','title':s['title'],'text':s['snippet'],'context_spans':s['metadata'].get('context_spans',[])} for i,s in enumerate(budget['sources'],1)]
        rows.append({'id':f'natural/{n:02d}','task':task,'evidence':ev,'baseline_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    for row in json.loads((here.parent/'model-size-paired-20260921/INPUTS.json').read_text()):
        rows.append({'id':row['id'],'task':json.dumps({'history':row['history'],'latest_question':row['question']},ensure_ascii=False),'evidence':row['evidence']})
    async with NativeRWKVClient(base_url='http://127.0.0.1:18426/v1',model='paired-eval',prompt_protocol='g1j_plain',timeout_seconds=180,max_concurrency=1) as client:
        for n,row in enumerate(rows):
            tick=time.monotonic();prompt=compact_writer_prompt(row['task'],row['evidence'])
            result=await client.complete([{'role':'user','content':prompt}],max_tokens=2048,assistant_prefill='<think></think',temperature=1,top_p=1,top_k=1,seed=11,stage='writer',evidence_ids=[e['label'] for e in row['evidence']])
            record={**row,'ordinal':n,'status':result.status,'raw_text':result.raw_text,'trace':result.trace,'elapsed_s':time.monotonic()-tick,'citation_audit':audit_citations(result.raw_text or '',[{'snippet':e['text']} for e in row['evidence']])}
            with (out/f'{n:03d}.json').open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2)
            print(n,row['id'],result.status,flush=True)
if __name__=='__main__':asyncio.run(main())
