"""Read-only audit/export. Structural statistics never stand in for semantics."""
import argparse
import base64
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics

def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def main(run,out):
    out.mkdir(parents=True,exist_ok=False)
    inputs=json.loads((run/'TOKENIZED-INPUTS.json').read_text())
    frozen=json.loads((Path(__file__).resolve().parent/'INPUTS.json').read_text())
    assert [{k:v for k,v in r.items() if k!='prompt_token_ids'} for r in inputs]==frozen
    rows=[];by={}
    for arm in ['2.9b','7.2b']:
        for rnd in [1,2]:
            for p in sorted((run/f'{arm}-round{rnd}').glob('*.json')):
                if p.name.endswith('.request.json'):continue
                r=json.loads(p.read_text());i=r['ordinal'];case=inputs[i]
                assert r['status']=='recorded'
                assert sha(r['raw_text'].encode())==r['raw_text_sha256']
                response=base64.b64decode(r['response_body_base64'])
                assert sha(response)==r['response_sha256']
                choice=json.loads(response)['choices'][0]
                assert choice['text']==r['raw_text']
                assert choice['prompt_token_ids']==case['prompt_token_ids']
                assert r['request']['prompt']==case['prompt']
                wire=json.dumps(r['request'],ensure_ascii=False,separators=(',',':')).encode()
                assert sha(wire)==r['request_sha256']
                r['raw_record_sha256']=sha(p.read_bytes())
                rows.append(r);by[arm,rnd,i]=r
    assert len(rows)==752
    summary={'calls':len(rows),'case_memberships':188,
             'distinct_prompts':len({r['prompt_sha256'] for r in inputs}),
             'all_raw_response_and_input_hashes_verified':True,
             'semantic_scoring':'requires manual review; no accuracy inferred', 'groups':{},'drift':{},'output_token_drift':{}}
    for arm in ['2.9b','7.2b']:
        for rnd in [1,2]:
            rs=[r for r in rows if r['arm']==arm and r['round']==rnd]
            summary['groups'][f'{arm}-round{rnd}']={
                'count':len(rs),'finish_reasons':dict(Counter(r['finish_reason'] for r in rs)),
                'repetition_candidates':[r['ordinal'] for r in rs if r['repetition']['automatic_candidate_only']],
                'output_tokens':sum(r['usage']['completion_tokens'] for r in rs),
                'elapsed_ms_median':statistics.median(r['elapsed_ms'] for r in rs)}
        summary['drift'][arm]=[i for i in range(188) if
            (by[arm,1,i]['raw_text'],by[arm,1,i]['finish_reason'])!=
            (by[arm,2,i]['raw_text'],by[arm,2,i]['finish_reason'])]
        summary['output_token_drift'][arm]=[i for i in range(188) if
            by[arm,1,i]['output_token_ids'] != by[arm,2,i]['output_token_ids']]
    for i in range(188):
        assert len({by[a,n,i]['request_sha256'] for a in ['2.9b','7.2b'] for n in [1,2]})==1
    summary['paired_request_bytes_equal']=True
    save(out/'AUTOMATIC-SUMMARY.json',summary)
    save(out/'ALL-ANSWERS.json',{'inputs':inputs,'answers':rows})
    with (out/'全部752条回答.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['ordinal','id','suite','category','question','model','round','answer','finish','elapsed_ms','source_sha256'])
        for r in rows:
            c=inputs[r['ordinal']]
            w.writerow([r['ordinal'],r['id'],r['suite'],r['category'],c['question'],r['arm'],r['round'],r['raw_text'],r['finish_reason'],r['elapsed_ms'],r['raw_record_sha256']])
    lines=['# 188条题目成员：2.9B与7.2B两轮原始回答','',
           '固定材料直接回答；全部已见开发题，不是在线检索或生产前端验收。所有原始答案保持不变。','']
    for c in inputs:
        i=c['ordinal'];lines += [f"## {i:03d} — {c['id']}",'',c['question'],'',
            '### 历史','', '```json',json.dumps(c['history'],ensure_ascii=False,indent=2),'```','', '### 原始材料','']
        for e in c['evidence']:lines += [f"**{e['label']}**",'', '```text',e['text'],'```','']
        for arm in ['2.9b','7.2b']:
            for rnd in [1,2]:
                r=by[arm,rnd,i]
                lines += [f'### {arm} / 第{rnd}轮','',f"finish={r['finish_reason']}；耗时={r['elapsed_ms']:.1f}ms；原始记录SHA256={r['raw_record_sha256']}",'','```text',r['raw_text'],'```','']
    markdown='\n'.join(lines)
    assert all(r['raw_text'] in markdown for r in rows)
    (out/'全部题目与两模型原始回答.md').write_text(markdown)
    save(out/'HASHES.json',{p.name:sha(p.read_bytes()) for p in out.iterdir() if p.is_file()})
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();main(a.run,a.output)
