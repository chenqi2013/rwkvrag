"""V7 relabeling on newly fictional training-only V6 materials."""
import argparse
import asyncio
from collections import Counter
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/statetune_broad'))
from contracts import compile_item, review_map
spec = importlib.util.spec_from_file_location('bounded_teacher', ROOT/'scripts/statetune_broad/generate.py')
bounded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bounded)
FROZEN = ROOT/'llamaindex-retrieval/statetune/progression-relabel-v7-20260922'
write = bounded.write


def material_map(job, draft=None):
    sources = job['sources'] if job['material_mode'] == 'real' else draft['sources']
    if not 1 <= len(sources) <= 6:
        raise ValueError('Invalid number of sources')
    result = {}
    for source in sources:
        if (not isinstance(source, dict) or not all(isinstance(source.get(k), str) and source[k].strip() for k in ('id','title','text'))
                or source['id'] in result):
            raise ValueError('Invalid source identity/text')
        if job['material_mode'] == 'synthetic' and not 40 <= len(source['text']) <= 4000:
            raise ValueError('Invalid synthetic source length')
        result[source['id']] = source
    if job['material_mode'] == 'synthetic' and len(result) < 3:
        raise ValueError('Synthetic material requires multiple independent documents')
    return result


def compile_batch(draft, sources):
    if set(draft) != {'items'} or not isinstance(draft['items'], list) or not 6 <= len(draft['items']) <= 24:
        raise ValueError('Invalid batch shape')
    items = draft['items']
    ids = [item.get('id') for item in items]
    if any(not isinstance(x,str) or not x.strip() for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('Invalid/duplicate item identity')
    if {i.get('stage') for i in items} != {'normal','defect','progress'}:
        raise ValueError('Missing normal, defect or progression stage')
    compiled, checks = {}, []
    for index,item in enumerate(items):
        try:
            parent = item['parent_id']
            if parent is not None and parent not in ids[:index]:
                raise ValueError('Parent must precede child')
            if item['stage'] == 'progress' and parent is None:
                raise ValueError('Progress needs a prior task')
            available = item['available_sources']
            if not isinstance(available,list) or len(set(available)) != len(available) or not set(available) <= set(sources):
                raise ValueError('Invalid visible source set')
            visible = {key:sources[key] for key in available}
            if item['kind'] == 'status' and available:
                raise ValueError('Status cannot see source content')
            student = compile_item(item, visible)
            student.update(item_id=item['id'], stage=item['stage'], parent_id=parent,
                           available_sources=available)
            compiled[index] = student
            checks.append({'index':index,'pass':True})
        except (ValueError,KeyError,TypeError) as exc:
            checks.append({'index':index,'pass':False,'error':str(exc)})
    return compiled, checks


def accepted_with_parents(items, compiled, decisions):
    accepted, ids = [], set()
    for index,item in enumerate(items):
        if index not in compiled or not decisions[index]['accept']:
            continue
        if item['parent_id'] is not None and item['parent_id'] not in ids:
            continue
        accepted.append(dict(compiled[index], index=index, teacher_review=decisions[index], skill=item['skill']))
        ids.add(item['id'])
    return accepted


async def main(args):
    pins = json.loads((FROZEN/'PINS.json').read_text())
    for path,digest in pins.items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen source mismatch: '+path)
    config = json.loads((FROZEN/'CONFIG.json').read_text())
    credentials = json.loads(args.credentials.read_text())
    if (args.credentials.stat().st_mode & 0o077 or credentials['base_url'] != config['teacher_base_url']
            or credentials['model'] != config['teacher_model']):
        raise ValueError('Credential permissions or service identity mismatch')
    jobs = [json.loads(line) for line in (ROOT/config['jobs_path']).read_text().splitlines()]
    if not 0 <= args.start < args.stop <= len(jobs):
        raise ValueError('Invalid job range')
    selected = jobs[args.start:args.stop]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    guard=(args.out.parent/'teacher.lock').open('a')
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=0.
    for path in args.out.parent.glob('*/calls/*.request.json'):
        receipt=path.with_name(path.name.replace('.request.json','.response.json'))
        prior += json.loads(receipt.read_text())['conservative_cost_usd'] if receipt.exists() else json.loads(path.read_text())['reserved_upper_usd']
    args.out.mkdir(exist_ok=False)
    for name in ('calls','results'): (args.out/name).mkdir()
    write(args.out/'RUN.json',{'started_at':bounded.now(),'config':config,'jobs':[j['id'] for j in selected],
         'pins_sha256':hashlib.sha256((FROZEN/'PINS.json').read_bytes()).hexdigest(),'prior_cost':prior})
    teacher=bounded.Teacher(credentials,config,args.out,prior)
    queue=asyncio.Queue()
    for job in selected:queue.put_nowait(job)
    totals=Counter()

    def body(phase,value):
        prompt=(FROZEN/('generate.txt' if phase=='generation' else phase+'.txt')).read_text()
        return {'model':config['teacher_model'],'messages':[{'role':'system','content':prompt},
            {'role':'user','content':json.dumps(value,ensure_ascii=False)+'\n请遵循系统约定，仅返回本步骤的JSON结构。'}],
            'response_format':{'type':'json_object'},'thinking':{'type':'disabled'},
            'temperature':0.0 if phase=='review' else 0.2,'max_tokens':config[phase+'_max_tokens'],'stream':False}

    async def worker():
        while not queue.empty() and not teacher.stopped.is_set():
            try:job=queue.get_nowait()
            except asyncio.QueueEmpty:return
            result={'job_id':job['id'],'split':job['split'],'material_mode':job['material_mode'],
                    'source_families':job['source_families'],'accepted':[],'status':'failed'}
            try:
                if job['material_mode']=='synthetic':
                    theme=[{k:s[k] for k in ('title','text')} for s in job['sources']]
                    material=await teacher.call(job['id'],'material',body('material',{'theme':theme,'task':'创造新的明确虚构材料，不复制主题文章'}))
                    result['material_draft']=material
                    sources=material_map(job,material)
                else:sources=material_map(job)
                result['sources']=list(sources.values())
                draft=await teacher.call(job['id'],'generation',body('generation',{
                    'sources':[{k:s[k] for k in ('id','title','text')} for s in sources.values()],
                    'task':job['focus']+' 只写items，不写sources；每条target必须短、准确、逐句引用可见原文。'}))
                result['draft']=draft
                compiled,checks=compile_batch(draft,sources)
                result['structural_checks']=checks
                review_input=[]
                for index,item in enumerate(draft['items']):
                    student=compiled.get(index)
                    review_input.append({'index':index,'id':item['id'],'stage':item['stage'],'parent_id':item['parent_id'],
                        'prompt':student['prompt_body'] if student else '结构无效，请拒绝此项',
                        'target':student['target'] if student else '', 'structurally_valid':student is not None})
                review=await teacher.call(job['id'],'review',body('review',{'items':review_input}))
                result['review']=review
                decisions=review_map(review,len(draft['items']))
                result['accepted']=accepted_with_parents(draft['items'],compiled,decisions)
                result['status']='reviewed'
                totals['reviewed_jobs']+=1
            except Exception as exc:
                result['error_type']=type(exc).__name__
                result['error']=str(exc) if isinstance(exc,(ValueError,KeyError,RuntimeError)) else 'Request/data processing failed; see receipt'
                totals['failed_jobs']+=1
            for item in result['accepted']:
                item['id']=job['id']+':'+item['item_id']
                totals['accepted_'+item['state_role']]+=1
                totals['stage_'+item['stage']]+=1
            write(args.out/'results'/f"{job['id']}.json",result)
            totals['jobs']+=1
            totals['accepted']+=len(result['accepted'])
            print(json.dumps(dict(totals,cost=round(teacher.spent,4))),flush=True)
    try:await asyncio.gather(*(worker() for _ in range(config['concurrency'])))
    finally:
        await teacher.close()
        write(args.out/'SUMMARY.json',dict(totals,ended_at=bounded.now(),planned_jobs=len(selected),
              stopped=teacher.stopped.is_set(),conservative_cost_usd=teacher.spent,training_exported=False,independent_review=False))
        guard.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--start',type=int,required=True)
    parser.add_argument('--stop',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--credentials',type=Path,default=Path.home()/'.config/rwkvrag/teacher-deepseek-20260922.json')
    asyncio.run(main(parser.parse_args()))
