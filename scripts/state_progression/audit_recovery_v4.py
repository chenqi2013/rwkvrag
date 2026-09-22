"""Explicit recovery audit: review jobs with no prior successful audit, two targets per call."""
import argparse
import asyncio
from collections import Counter
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path

spec=importlib.util.spec_from_file_location('progression_generator',Path(__file__).with_name('generate.py'))
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)

def decisions(raw,size):
    result={}
    for row in raw['reviews']:
        index=row['index']
        if type(index) is not int or not 0<=index<size or index in result or type(row['accept']) is not bool:
            raise ValueError('Invalid review identity or decision')
        if not isinstance(row.get('issues'),list) or not isinstance(row.get('reason'),str) or not row['reason'].strip():
            raise ValueError('Missing review justification')
        if row['accept'] and row['issues']:raise ValueError('Unresolved issue marked accepted')
        result[index]=row
    if len(result)!=size:raise ValueError('Incomplete review')
    return result

async def main(args):
    config=json.loads((g.FROZEN/'CONFIG.json').read_text());config.update(concurrency=8,timeout_s=240,cost_ceiling_usd=40)
    credentials=json.loads(args.credentials.read_text())
    exclusions=json.loads(args.exclusions.read_text())
    if args.credentials.stat().st_mode & 0o077 or credentials['base_url']!=config['teacher_base_url'] or credentials['model']!=config['teacher_model']:
        raise ValueError('Private configuration mismatch')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    guard=(args.out.parent/'audit.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=0.;claimed=set()
    for p in args.out.parent.glob('*/results/*.json'):
        row=json.loads(p.read_text())
        if row['status']=='reviewed':claimed.add(row['job_id'])
    for p in args.out.parent.glob('*/calls/*.request.json'):
        receipt=p.with_name(p.name.replace('.request.json','.response.json'))
        prior+=json.loads(receipt.read_text())['conservative_cost_usd'] if receipt.exists() else json.loads(p.read_text())['reserved_upper_usd']
    sources=[]
    for root in args.input:
        for p in sorted(root.glob('*/results/*.json')):
            data=p.read_bytes()
            try:r=json.loads(data)
            except json.JSONDecodeError:continue  # Writer has not closed this new file yet; not snapshotted.
            if r['job_id'] in claimed or not r['accepted'] or (args.job_ids and r['job_id'] not in args.job_ids):continue
            excluded=[x['id'] in exclusions for x in r['accepted']]
            if any(excluded) and not all(excluded):raise ValueError('Partial job quarantine: '+r['job_id'])
            if all(excluded):continue
            sources.append((p,hashlib.sha256(data).hexdigest(),r))
    sources=sorted(sources,key=lambda row:row[2]['job_id'])[:args.max_jobs]
    if not sources:raise ValueError('No new completed candidate jobs')
    args.out.mkdir(exist_ok=False)
    for name in ('calls','results'):(args.out/name).mkdir()
    prompt=Path(__file__).with_name('critical_review_v3.txt').read_text()
    g.write(args.out/'INPUTS.json',{'sources':[{'path':str(p),'sha256':h,'job_id':r['job_id']} for p,h,r in sources],
        'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'thinking':'enabled','reasoning_effort':'high','max_tokens':16384,'chunk_size':2,'prior_cost':prior,'cost_ceiling_usd':40,'independent_review':False,
        'recovery_policy':'Retry prior failed jobs once in this run; review interrupted and newly completed jobs. Earlier raw failures remain unchanged.',
        'exclusions_sha256':hashlib.sha256(args.exclusions.read_bytes()).hexdigest()})
    teacher=g.bounded.Teacher(credentials,config,args.out,prior)
    queue=asyncio.Queue()
    for row in sources:queue.put_nowait(row)
    totals=Counter()
    async def worker():
        while not queue.empty() and not teacher.stopped.is_set():
            try:path,digest,r=queue.get_nowait()
            except asyncio.QueueEmpty:return
            output={'job_id':r['job_id'],'input_sha256':digest,'accepted_ids':[],'reviews':[],'status':'failed'}
            try:
                items=r['accepted'];by_id={x['item_id']:x for x in items};approved=set()
                for start in range(0,len(items),2):
                    part=items[start:start+2]
                    context=[]
                    for x in part:
                        if x['parent_id']:
                            parent=by_id[x['parent_id']]
                            context.append({'id':parent['item_id'],'prompt':parent['prompt_body'],'target':parent['target']})
                    targets=[{'index':i,'id':x['item_id'],'stage':x['stage'],'parent_id':x['parent_id'],
                         'prompt':x['prompt_body'],'target':x['target']} for i,x in enumerate(part)]
                    body={'model':config['teacher_model'],'messages':[{'role':'system','content':prompt},
                        {'role':'user','content':json.dumps({'targets':targets,'parent_context_only':context},ensure_ascii=False)}],
                        'thinking':{'type':'enabled'},'reasoning_effort':'high','response_format':{'type':'json_object'},
                        'max_tokens':16384,'stream':False}
                    raw=await teacher.call(r['job_id'],f'chunk-{start}',body)
                    reviewed=decisions(raw,len(part))
                    output['reviews'].append({'start':start,'raw':raw})
                    for i,x in enumerate(part):
                        if reviewed[i]['accept']:approved.add(x['item_id'])
                admitted=set()
                for x in items:
                    if x['item_id'] in approved and (x['parent_id'] is None or x['parent_id'] in admitted):
                        output['accepted_ids'].append(x['id']);admitted.add(x['item_id'])
                output['status']='reviewed'
            except Exception as exc:
                output['error']=str(exc) if isinstance(exc,(RuntimeError,ValueError,KeyError)) else type(exc).__name__
                totals['failed_jobs']+=1
            g.write(args.out/'results'/f"{r['job_id']}.json",output)
            totals['jobs']+=1;totals['accepted']+=len(output['accepted_ids'])
            print(json.dumps(dict(totals,cost=round(teacher.spent,4))),flush=True)
    try:await asyncio.gather(*(worker() for _ in range(config['concurrency'])))
    finally:
        await teacher.close()
        g.write(args.out/'SUMMARY.json',dict(totals,planned_jobs=len(sources),conservative_cost_usd=teacher.spent,
            stopped=teacher.stopped.is_set(),independent_review=False,training_exported=False))
        guard.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--max-jobs',type=int,default=1000)
    p.add_argument('--job-ids',nargs='*')
    p.add_argument('--exclusions',type=Path,required=True)
    p.add_argument('--credentials',type=Path,default=Path.home()/'.config/rwkvrag/teacher-deepseek-20260922.json')
    asyncio.run(main(p.parse_args()))
