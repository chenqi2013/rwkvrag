"""Second-pass label audit; freezes a complete result snapshot before any API call."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import importlib.util

spec=importlib.util.spec_from_file_location('progression_generator',Path(__file__).with_name('generate.py'))
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)

async def main(args):
    config=json.loads((g.FROZEN/'CONFIG.json').read_text())
    config.update(concurrency=4,review_max_tokens=8192)
    credentials=json.loads(args.credentials.read_text())
    if args.credentials.stat().st_mode & 0o077 or credentials['base_url']!=config['teacher_base_url'] or credentials['model']!=config['teacher_model']:
        raise ValueError('Private service identity mismatch')
    source=[]
    for phase in args.phases:
        directory=args.input/phase
        summary=json.loads((directory/'SUMMARY.json').read_text())
        if summary['jobs']!=summary['planned_jobs']:
            raise ValueError('Input phase incomplete')
        for path in sorted((directory/'results').glob('*.json')):
            raw=path.read_bytes();result=json.loads(raw)
            if result['accepted']:
                source.append((path,hashlib.sha256(raw).hexdigest(),result))
    args.out.mkdir(parents=True,exist_ok=False)
    for name in ('calls','results'):(args.out/name).mkdir()
    prompt=Path(__file__).with_name('critical_review.txt').read_text()
    g.write(args.out/'INPUTS.json',{'sources':[{'path':str(p),'sha256':h} for p,h,r in source],
         'review_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'independent_review':False})
    teacher=g.bounded.Teacher(credentials,config,args.out,0.)
    queue=asyncio.Queue()
    for row in source:queue.put_nowait(row)
    totals=Counter()
    async def worker():
        while not queue.empty() and not teacher.stopped.is_set():
            try:path,digest,result=queue.get_nowait()
            except asyncio.QueueEmpty:return
            output={'job_id':result['job_id'],'input_sha256':digest,'accepted_ids':[],'status':'failed'}
            try:
                items=result['accepted']
                inputs=[{'index':i,'id':x['item_id'],'stage':x['stage'],'parent_id':x['parent_id'],
                         'prompt':x['prompt_body'],'target':x['target']} for i,x in enumerate(items)]
                body={'model':config['teacher_model'],'messages':[{'role':'system','content':prompt},
                     {'role':'user','content':json.dumps({'items':inputs},ensure_ascii=False)}],
                     'response_format':{'type':'json_object'},'thinking':{'type':'disabled'},
                     'temperature':0.,'max_tokens':8192,'stream':False}
                raw=await teacher.call(result['job_id'],'critical-review',body)
                output['review']=raw
                decisions=g.review_map(raw,len(items))
                ids=set()
                for i,x in enumerate(items):
                    if decisions[i]['accept'] and (x['parent_id'] is None or x['parent_id'] in ids):
                        output['accepted_ids'].append(x['id']);ids.add(x['item_id'])
                output['status']='reviewed'
            except Exception as exc:
                output['error']=str(exc) if isinstance(exc,(ValueError,RuntimeError,KeyError)) else type(exc).__name__
                totals['failed_jobs']+=1
            totals['jobs']+=1;totals['accepted']+=len(output['accepted_ids'])
            g.write(args.out/'results'/f"{result['job_id']}.json",output)
            print(json.dumps(dict(totals,cost=round(teacher.spent,4))),flush=True)
    try:await asyncio.gather(*(worker() for _ in range(config['concurrency'])))
    finally:
        await teacher.close()
        g.write(args.out/'SUMMARY.json',dict(totals,planned_jobs=len(source),conservative_cost_usd=teacher.spent,
            stopped=teacher.stopped.is_set(),independent_review=False,training_exported=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--phases',nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--credentials',type=Path,default=Path.home()/'.config/rwkvrag/teacher-deepseek-20260922.json')
    asyncio.run(main(p.parse_args()))
