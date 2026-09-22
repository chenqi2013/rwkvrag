"""Keep every historical Writer case and bind exact prompts; optionally add new dev/holdout."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'llamaindex-retrieval/src'))
from llamaindex_retrieval.state_tokens import Vocabulary

def sha(raw):return hashlib.sha256(raw).hexdigest()
def main(args):
    rows=[];pins={}
    p=ROOT/'llamaindex-retrieval/eval/model-size-paired-20260921/INPUTS.json'
    pins[str(p.relative_to(ROOT))]=sha(p.read_bytes())
    for r in json.loads(p.read_text()):
        if sha(r['prompt'].encode())!=r['prompt_sha256']:raise ValueError('Historical prompt changed')
        rows.append({'id':'paired188/'+r['id'],'suite':'paired188','category':r['category'],
            'state_role':'writer','prompt':r['prompt'],'question':r['question'],'history':r['history'],
            'expected':r['case'],'evidence':r['evidence'],'evaluation_kind':'seen_fixed_evidence_writer'})
    base=ROOT/'data/quality-runs/restored-retrieval-v2-20260920/run1/calls'
    cases=json.loads((ROOT/'llamaindex-retrieval/eval/restored-retrieval-v2-20260920/cases.json').read_text())
    for index,case in enumerate(cases):
        p=base/f'{index:04d}.json';pins[str(p.relative_to(ROOT))]=sha(p.read_bytes())
        record=json.loads(p.read_text())
        if record['case_id']!=case['id']:raise ValueError('Historical case membership changed')
        try:result=json.loads(record['raw_response']) if record.get('raw_response') else {}
        except json.JSONDecodeError:result={}
        calls=[c for c in result.get('generation',{}).get('model_calls',[]) if c.get('stage')=='writer' and c.get('prompt')]
        row={'id':'restored434/'+case['id'],'suite':'restored434','category':case['category'],'state_role':'writer',
            'question':case['payload']['question'],'history':case['payload'].get('history',[]),'expected':case['original_case'],
            'evidence':result.get('sources',[]),'evaluation_kind':'seen_frozen_retrieval_writer_replay_not_live_retrieval'}
        if calls:
            call=calls[-1]
            if sha(call['prompt'].encode())!=call['prompt_sha256']:raise ValueError('Historical Writer prompt hash mismatch')
            row['prompt']=call['prompt']
        else:row.update(prompt=None,unsupported_reason='original_run_has_no_writer_prompt')
        rows.append(row)
    p=ROOT/'llamaindex-retrieval/eval/writer-evidence-handoff-20260921/INPUTS.json'
    pins[str(p.relative_to(ROOT))]=sha(p.read_bytes())
    for r in json.loads(p.read_text()):
        for variant in ('baseline','candidate'):
            prompt='User: '+r[variant+'_prompt']+'\n\nAssistant: <think></think>\n'
            if variant=='baseline' and sha(prompt.encode())!=r['previous_wire_prompt_sha256']:
                raise ValueError('Historical handoff prompt boundary mismatch')
            rows.append({'id':'handoff36/'+r['id']+'/'+variant,'suite':'handoff36_'+variant,'category':'comparison_and_execution',
                'state_role':'writer','question':r['question'],'history':r['history'],'prompt':prompt,
                'evidence':r['sources'],'flow':r['flow'],'expected':r['rubric'],'evaluation_kind':'seen_fixed_upstream_writer'})
    if args.release:
        for split in ('dev','holdout'):
            p=args.release/(split+'.jsonl');pins[str(p)]=sha(p.read_bytes())
            for line in p.read_text().splitlines():
                r=json.loads(line)
                if r['split']!=split:raise ValueError('Release split mismatch')
                rows.append({'id':split+'/'+r['id'],'suite':split,'state_role':r['state_role'],'prompt':r['prompt'],
                    'question':r['raw_item'].get('question',r['raw_item'].get('field',{}).get('question')),
                    'expected_target':r['target'],'category':r['kind']+'/'+r['stage'],
                    'source_families':r['source_families'],'evaluation_kind':'source_separated_same_teacher_labels_not_independent'})
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate evaluation member ID')
    vocab=Vocabulary(ROOT/'llamaindex-retrieval/statetune/assets/rwkv_vocab_v20230424.txt')
    for r in rows:
        if r['prompt'] is None:continue
        r['input_ids']=vocab.encode(r['prompt']);r['prompt_sha256']=sha(r['prompt'].encode())
        r['max_output_tokens']=512 if r['state_role']=='resolver' else 2048
        if len(r['input_ids'])+r['max_output_tokens']>16384:r['unsupported_reason']='full_input_plus_output_exceeds_16384'
    args.out.mkdir(parents=True,exist_ok=False)
    with (args.out/'cases.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    (args.out/'PINS.json').write_text(json.dumps(pins,indent=2)+'\n')
    summary={'members':len(rows),'by_suite':dict(Counter(r['suite'] for r in rows)),
        'unsupported_members':sum('unsupported_reason' in r for r in rows),
        'unique_supported_prompts':len({r['prompt_sha256'] for r in rows if 'unsupported_reason' not in r}),
        'cases_sha256':sha((args.out/'cases.jsonl').read_bytes()),'all_historical_members_retained':True,
        'rounds':2,'arms':['zero','trained'],'planned_member_records':len(rows)*4,
        'live_retrieval':False,'production_promoted':False}
    (args.out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--release',type=Path)
    main(p.parse_args())
