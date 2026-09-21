import asyncio,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'llamaindex-retrieval/src'))
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline,conversation
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.schemas import SourceItem,ConversationMessage
async def main():
    here=Path(__file__).resolve().parent
    for name,digest in json.loads((here/'PINS.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    out=ROOT/'data/quality-runs/writer-budget-repair-20260921/run1';out.mkdir(parents=True,exist_ok=False)
    config=json.loads((here.parent/'github-natural-comparison-20260921/CONFIG.json').read_text());config['native_writer_budget_policy']='whole_sources'
    pipeline=RWKVPipeline(Settings(**config),None)
    try:
        old=ROOT/'data/quality-runs/github-natural-comparison-20260921/run1'
        for n in range(8):
            p=old/f'{n:02d}.json';record=json.loads(p.read_text());response=record['response'];request=record['request']
            task=conversation(request['question'],[ConversationMessage(**h) for h in request.get('history',[])])
            sources=[SourceItem(**s) for s in response['sources']];fields=response['retrieval']['active_tasks']
            original=next(e for e in reversed(response['generation']['model_calls']) if e['stage']=='writer')
            assert pipeline._writer_prompt(task,sources,fields)==original['messages'][0]['content']
            result=await pipeline._write(task,sources,fields)
            selected={s.id:s for s in sources};actual=[selected[x] for x in result.trace['evidence_budget']['included_source_ids']]
            row={'ordinal':n,'source_record_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'status':result.status,'raw_text':result.raw_text,'trace':result.trace,'sources':[s.model_dump() for s in actual]}
            (out/f'{n:02d}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2));print(n,result.status,len(actual),flush=True)
        ordinary=json.loads((here.parent/'model-size-paired-20260921/INPUTS.json').read_text());checks=[]
        for row in ordinary:
            # These frozen prompts include the full g1j transport envelope.
            prompt=row['prompt'];prefix='User: ';suffix='\n\nAssistant: <think></think>\n'
            assert prompt.startswith(prefix) and prompt.endswith(suffix)
            result=await pipeline.model.complete([{'role':'user','content':prompt[len(prefix):-len(suffix)]}],assistant_prefill='<think></think',max_tokens=2048,temperature=1,top_p=1,top_k=1,seed=11,stage='writer_budget',check_only=True)
            assert result.trace['prompt']==prompt
            checks.append({'id':row['id'],'status':result.status,'trace':result.trace})
        (out/'ORDINARY-BUDGET-CHECKS.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2));print('Ordinary',len(checks),flush=True)
    finally:await pipeline.aclose()
if __name__=='__main__':asyncio.run(main())
