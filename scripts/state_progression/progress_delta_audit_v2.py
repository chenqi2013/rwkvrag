"""Fail-closed, immutable second review of real parent-to-child progress."""
import argparse
import asyncio
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('progression_generator',Path(__file__).with_name('generate.py'))
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)


def digest(raw):return hashlib.sha256(raw).hexdigest()


def load(release):
    summary=json.loads((release/'SUMMARY.json').read_text())
    rows=[];pins={}
    for split in ('train','dev','holdout'):
        path=release/(split+'.jsonl');raw=path.read_bytes()
        if digest(raw)!=summary['files'][path.name]:raise ValueError('Release file pin changed')
        pins[path.name]=digest(raw)
        rows.extend(json.loads(line) for line in raw.splitlines())
    by_id={r['id']:r for r in rows}
    if len(by_id)!=len(rows):raise ValueError('Duplicate release identity')
    targets=[]
    for child in rows:
        if child['stage']!='progress':continue
        if not re.fullmatch(r'[A-Za-z0-9:_-]+',child['id']):raise ValueError('Unsafe target identity')
        parent=by_id.get(child['job_id']+':'+str(child['parent_id']))
        if parent is None or parent['split']!=child['split']:
            raise ValueError('Progress parent missing from same split: '+child['id'])
        targets.append((child,parent))
    return sorted(targets,key=lambda pair:pair[0]['id']),pins


def verdict(raw):
    if set(raw)!={'accept','new_relevant_fact','corrects_prior_error','answers_new_requirement','reason'}:
        raise ValueError('Invalid delta-review fields')
    if any(type(raw[k]) is not bool for k in ('accept','new_relevant_fact','corrects_prior_error','answers_new_requirement')):
        raise ValueError('Invalid delta-review booleans')
    if not isinstance(raw['reason'],str) or len(raw['reason'].strip())<12:
        raise ValueError('Missing specific delta-review reason')
    if raw['accept'] and not any(raw[k] for k in ('new_relevant_fact','corrects_prior_error','answers_new_requirement')):
        raise ValueError('Progress accepted without any delta')
    return raw


async def main(args):
    config=json.loads((g.FROZEN/'CONFIG.json').read_text())
    config.update(concurrency=4,timeout_s=240,cost_ceiling_usd=15)
    credentials=json.loads(args.credentials.read_text())
    if (args.credentials.stat().st_mode&0o077 or credentials['base_url']!=config['teacher_base_url']
            or credentials['model']!=config['teacher_model']):
        raise ValueError('Private teacher configuration mismatch')
    targets,pins=load(args.release)
    if args.ids:
        requested=set(args.ids)
        present={child['id'] for child,_ in targets}
        if requested-present:raise ValueError('Requested progress IDs absent from release')
        targets=[pair for pair in targets if pair[0]['id'] in requested]
    targets=targets[:args.max_items]
    if not targets:raise ValueError('No progress targets')
    args.out.mkdir(parents=True,exist_ok=False)
    for name in ('calls','results'):(args.out/name).mkdir()
    prompt=Path(__file__).with_name('progress_delta_review_v1.txt').read_text()
    g.write(args.out/'INPUTS.json',{'release':str(args.release),'release_pins':pins,
        'target_ids':[child['id'] for child,_ in targets],
        'script_sha256':digest(Path(__file__).read_bytes()),'prompt_sha256':digest(prompt.encode()),
        'review_model':config['teacher_model'],'thinking':'enabled','reasoning_effort':'high',
        'max_tokens':8192,'same_teacher_not_independent':True,'fail_closed':True})
    teacher=g.bounded.Teacher(credentials,config,args.out,0.)
    queue=asyncio.Queue()
    for pair in targets:queue.put_nowait(pair)
    totals=Counter()
    async def worker():
        while not queue.empty() and not teacher.stopped.is_set():
            try:child,parent=queue.get_nowait()
            except asyncio.QueueEmpty:return
            body={'model':config['teacher_model'],'messages':[
                {'role':'system','content':prompt},
                {'role':'user','content':json.dumps({'parent':{'id':parent['id'],'prompt':parent['prompt'],
                    'target':parent['target'],'question':parent['raw_item'].get('question')},
                    'child':{'id':child['id'],'prompt':child['prompt'],'target':child['target'],
                    'question':child['raw_item'].get('question')}},ensure_ascii=False)}],
                'thinking':{'type':'enabled'},'reasoning_effort':'high',
                'response_format':{'type':'json_object'},'max_tokens':8192,'stream':False}
            result={'id':child['id'],'row_sha256':digest(json.dumps(child,ensure_ascii=False,sort_keys=True).encode()),
                    'parent_id':parent['id'],'status':'failed','accept':False}
            try:
                result['verdict']=verdict(await teacher.call(child['id'],'delta',body))
                result['accept']=result['verdict']['accept'];result['status']='reviewed'
            except Exception as exc:
                result['error']=str(exc) if isinstance(exc,(ValueError,RuntimeError,KeyError)) else type(exc).__name__
            g.write(args.out/'results'/(child['id']+'.json'),result)
            totals['reviewed' if result['status']=='reviewed' else 'failed']+=1
            totals['accepted' if result['accept'] else 'rejected_or_failed']+=1
            print(json.dumps(dict(totals,cost=round(teacher.spent,4))),flush=True)
    try:await asyncio.gather(*(worker() for _ in range(config['concurrency'])))
    finally:
        await teacher.close()
        g.write(args.out/'SUMMARY.json',dict(totals,planned=len(targets),
            conservative_cost_usd=teacher.spent,stopped=teacher.stopped.is_set(),
            independent_review=False,training_exported=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-items',type=int,default=100000)
    p.add_argument('--ids',nargs='*')
    p.add_argument('--credentials',type=Path,default=Path.home()/'.config/rwkvrag/teacher-deepseek-20260922.json')
    asyncio.run(main(p.parse_args()))
