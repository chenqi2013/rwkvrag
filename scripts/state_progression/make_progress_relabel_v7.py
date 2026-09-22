"""Freeze new tasks over V6 training-only fictional source documents."""
import argparse
import hashlib
import json
from pathlib import Path


FOCUS = {
    'comparison': '围绕同一问题比较多个对象、版本或条件。先用较早或较少资料给出可证的部分比较和具体缺口，再引入新增相关记录补足遗漏对象或条件。问题使用“按当前可见记录”等开放时间表述，不能反过来改变原问题的截止时点。',
    'lifecycle': '围绕计划、测试、批准、生效、完成或更正的状态变化提出另一项独立任务。先准确说明目前阶段和缺失的后续结果，再用新增的相关记录更新事实；批准不等于发布，发布不等于部署，计划不等于实测。',
}


def main(args):
    records=[]
    for path in sorted(args.source.glob('*/results/*.json')):
        raw=path.read_bytes()
        row=json.loads(raw)
        if row['status']!='reviewed' or not row.get('sources'):
            continue
        if row['material_mode']!='synthetic' or row['split']!='train':
            raise ValueError('V7 must use training-only V6 synthetic materials')
        if len(row['source_families'])!=1 or not row['source_families'][0].startswith('progress-repair-v6/'):
            raise ValueError('Unexpected V6 source family')
        records.append((path,hashlib.sha256(raw).hexdigest(),row))
    if len(records)!=57 or len({r['job_id'] for _,_,r in records})!=57:
        raise ValueError('Expected 57 distinct reviewed V6 source groups')
    records.sort(key=lambda x:hashlib.sha256(x[2]['job_id'].encode()).hexdigest())
    jobs=[];manifest=[]
    for path,digest,row in records:
        for mode,focus in FOCUS.items():
            identity='relabel-v7-'+hashlib.sha256((row['job_id']+'\n'+mode).encode()).hexdigest()[:20]
            jobs.append({'id':identity,'split':'train','material_mode':'real',
                         'source_families':row['source_families'],'sources':row['sources'],
                         'focus':focus,'focus_mode':mode})
            manifest.append({'job_id':identity,'mode':mode,'source_job_id':row['job_id'],
                             'source_result_path':str(path),'source_result_sha256':digest,
                             'source_families':row['source_families']})
    if len(jobs)!=114 or len({j['id'] for j in jobs})!=114:
        raise ValueError('Expected 114 distinct new tasks')
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'JOBS.jsonl').exists() or (args.out/'SOURCE-SELECTION.json').exists():
        raise FileExistsError('V7 task selection is already frozen')
    payload=''.join(json.dumps(job,ensure_ascii=False)+'\n' for job in jobs)
    (args.out/'JOBS.jsonl').write_text(payload)
    summary={'jobs':len(jobs),'source_groups':len(records),'focus_modes':list(FOCUS),
             'split':'train','historical_question_inputs':0,'historical_answer_inputs':0,
             'source_type':'new_v6_fictional_training_only',
             'jobs_sha256':hashlib.sha256(payload.encode()).hexdigest(),
             'source_receipts':manifest}
    (args.out/'SOURCE-SELECTION.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='source_receipts'},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
