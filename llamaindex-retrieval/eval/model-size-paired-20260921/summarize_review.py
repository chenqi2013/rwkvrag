"""Bind implementer review to raw outputs; inherit only byte-identical repeats."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,d):
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2)

def main(run,annotations,out):
    out.mkdir(parents=True,exist_ok=False)
    inputs=json.loads((run/'TOKENIZED-INPUTS.json').read_text())
    notes=json.loads(annotations.read_text())
    records=[];by={}
    for arm in ['2.9b','7.2b']:
        for rnd in [1,2]:
            for i in range(188):
                p=run/f'{arm}-round{rnd}/{i:04d}.json';raw=json.loads(p.read_text())
                assert raw['status']=='recorded'
                key=f'{arm}/{rnd}/{i}'
                inherited=None
                if key in notes:
                    review=notes[key].copy()
                else:
                    assert rnd==2,key
                    first=json.loads((run/f'{arm}-round1/{i:04d}.json').read_text())
                    assert (raw['raw_text'],raw['finish_reason'],raw['output_token_ids']) == (first['raw_text'],first['finish_reason'],first['output_token_ids']),key
                    inherited=f'{arm}/1/{i}';review=notes[inherited].copy()
                strict=bool(review['facts_complete'] and review['citations_correct']
                            and raw['finish_reason']=='stop' and not review['repetition_confirmed'])
                row={'key':key,'arm':arm,'round':rnd,'ordinal':i,'id':raw['id'],
                     'suite':raw['suite'],'category':raw['category'],
                     'input_sha256':inputs[i]['prompt_sha256'],'raw_record_sha256':sha(p.read_bytes()),
                     'raw_answer_sha256':raw['raw_text_sha256'],'finish_reason':raw['finish_reason'],
                     'review':review,'strict_pass':strict,'review_inherited_from':inherited}
                records.append(row);by[key]=row
    groups={
        'all188':list(range(188)), 'old28':list(range(28)), 'full160':list(range(28,188)),
        'ordinary4':[r['ordinal'] for r in inputs if r['category']=='ordinary'],
        'old_regression8':[r['ordinal'] for r in inputs if r['suite']=='old28' and r['category']=='regression'],
        'comparison44':[r['ordinal'] for r in inputs if r['ordinal']>=28 and r['category']=='comparison'],
        'choice48':[r['ordinal'] for r in inputs if r['category']=='choice'],
        'mixed20':[r['ordinal'] for r in inputs if r['category']=='mixed']}
    seen=set();groups['distinct_prompts180']=[]
    for row in inputs:
        if row['prompt_sha256'] not in seen:
            groups['distinct_prompts180'].append(row['ordinal']);seen.add(row['prompt_sha256'])
    totals={}
    for arm in ['2.9b','7.2b']:
        for rnd in [1,2]:
            group_totals={}
            for label,ids in groups.items():
                rs=[by[f'{arm}/{rnd}/{i}'] for i in ids]
                group_totals[label]={'count':len(rs),
                    'facts_complete':sum(r['review']['facts_complete'] for r in rs),
                    'citations_correct':sum(r['review']['citations_correct'] for r in rs),
                    'strict_pass':sum(r['strict_pass'] for r in rs),
                    'repetition_confirmed':sum(r['review']['repetition_confirmed'] for r in rs),
                    'length':sum(r['finish_reason']=='length' for r in rs),
                    'hard_condition_error':sum(r['review'].get('hard_condition_error',False) for r in rs),
                    'hard_condition_wrong_recommendation':sum(r['review'].get('hard_condition_wrong_recommendation',False) for r in rs)}
            totals[f'{arm}-round{rnd}']=group_totals
    pairs={}
    for rnd in [1,2]:
        better=[];worse=[]
        for i in range(188):
            a=by[f'2.9b/{rnd}/{i}']['review']['facts_complete'];b=by[f'7.2b/{rnd}/{i}']['review']['facts_complete']
            if b and not a:better.append(i)
            if a and not b:worse.append(i)
        pairs[str(rnd)]={'facts_improved':better,'facts_regressed':worse,
                         'unchanged_count':188-len(better)-len(worse)}
    summary={'reviewer':'Codex task implementer, non-independent model-assisted semantic reading; not independent human assessment',
             'scope':'seen fixed-material direct answering; not live retrieval or production frontend',
             'records':len(records),'explicit_review_records':sum(r['review_inherited_from'] is None for r in records),
             'inherited_identical_repeats':sum(r['review_inherited_from'] is not None for r in records),
             'annotations_sha256':sha(annotations.read_bytes()),'groups':totals,'pairs':pairs,
             'promotion':'NOT_PROMOTED: hard-condition failures, unsupported assertions, bad citations and repetition remain; no broad commercial-quality claim'}
    save(out/'REVIEW.json',records);save(out/'SUMMARY.json',summary)
    lines=['# 逐题审读：2.9B / 7.2B 配对复验','',
           '由Codex任务实现者按问题、历史与原文审读，未经独立人工复核。完全相同的第二轮输出继承第一次审读，变化项另行阅读。原始答案没有修改。','',
           '事实完整性、引用、持续复读分别判断；正常结束不等于答对。未要求差值的比较题允许按对象/模式并列正确数值；选择可用唯一明确合格对象表达，不要求固定关键词。','']
    for c in inputs:
        i=c['ordinal'];lines += [f"## {i:03d} — {c['question']}",'',
           '| 模型/轮次 | 事实完整正确 | 引用正确 | 持续复读 | 结束 | 严格通过 |',
           '|---|---|---|---|---|---|']
        for arm in ['2.9b','7.2b']:
            for rnd in [1,2]:
                r=by[f'{arm}/{rnd}/{i}'];v=r['review']
                yes=lambda b:'是' if b else '否'
                lines.append(f"| {arm}/{rnd} | {yes(v['facts_complete'])} | {yes(v['citations_correct'])} | {yes(v['repetition_confirmed'])} | {r['finish_reason']} | {yes(r['strict_pass'])} |")
        lines += ['']
        for arm in ['2.9b','7.2b']:
            for rnd in [1,2]:
                r=by[f'{arm}/{rnd}/{i}']
                if r['review_inherited_from']:continue
                lines += [f"- {arm}/{rnd}：{r['review']['notes']}"]
        lines += ['']
    (out/'REVIEW.md').write_text('\n'.join(lines))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--annotations',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();main(a.run,a.annotations,a.output)
