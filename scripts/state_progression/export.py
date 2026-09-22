"""Admit audited labels, preserve source-family splits, deduplicate and encode without truncation."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'llamaindex-retrieval/src'))
from llamaindex_retrieval.state_tokens import Vocabulary,encode_training

def sha(raw):return hashlib.sha256(raw).hexdigest()
def normalized(value):
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',value)).casefold()

def collect(roots):
    records={};families={}
    for root in roots:
        for p in sorted(root.glob('*/results/*.json')):
            r=json.loads(p.read_text())
            if 'accepted' not in r:continue
            for f in r['source_families']:
                if f in families and families[f]!=r['split']:raise ValueError('Source family crossed split')
                families[f]=r['split']
            for item in r['accepted']:
                if item['id'] in records:raise ValueError('Duplicate input identity')
                raw_item=r['draft']['items'][item['index']]
                records[item['id']]=dict(item,split=r['split'],source_families=r['source_families'],
                    material_mode=r['material_mode'],job_id=r['job_id'],input_path=str(p),input_sha256=sha(p.read_bytes()),
                    raw_item=raw_item)
    return records

def audit_ids(roots):
    allowed={}
    for root in roots:
        for p in sorted(root.glob('*/results/*.json')):
            r=json.loads(p.read_text())
            if r['status']!='reviewed':continue
            for identity in r['accepted_ids']:
                if identity in allowed:raise ValueError('Duplicate audited identity')
                allowed[identity]=r['input_sha256']
    return allowed

def main(args):
    records=collect(args.input);allowed=audit_ids(args.audit)
    for identity,digest in allowed.items():
        if identity not in records or records[identity]['input_sha256']!=digest:raise ValueError('Audit/input identity mismatch')
    rejected={i:'critical_review_not_accepted' for i in records if i not in allowed}
    explicit=json.loads(args.exclusions.read_text()) if args.exclusions else {}
    for identity,reason in explicit.items():
        if identity not in records:raise ValueError('Manual exclusion not found')
        rejected[identity]='explicit_review: '+reason
    candidates=[r for i,r in records.items() if i not in rejected]
    by_prompt=defaultdict(list)
    for r in candidates:by_prompt[sha(r['prompt'].encode())].append(r)
    for group in by_prompt.values():
        if len({r['target'] for r in group})>1 or len({r['split'] for r in group})>1:
            for r in group:rejected[r['id']]='conflicting_target_or_cross_split_prompt'
        else:
            for r in sorted(group,key=lambda x:x['id'])[1:]:rejected[r['id']]='exact_duplicate_prompt_and_target'
    # Content fingerprints ignore fixed protocol boilerplate and superficial spacing.
    by_content=defaultdict(list)
    for r in candidates:
        raw=r['raw_item'];task=raw.get('question',json.dumps(raw.get('field'),ensure_ascii=False))
        key=normalized(task+'\n'+r['target'])
        by_content[key].append(r)
    for group in by_content.values():
        if len({r['split'] for r in group})>1:
            for r in group:rejected[r['id']]='cross_split_normalized_task_target'
        else:
            viable=sorted((r for r in group if r['id'] not in rejected),key=lambda x:x['id'])
            for r in viable[1:]:rejected[r['id']]='normalized_duplicate_task_target'
    vocab=Vocabulary(ROOT/'llamaindex-retrieval/statetune/assets/rwkv_vocab_v20230424.txt')
    encoded={}
    for r in candidates:
        if r['id'] in rejected:continue
        try:encoded[r['id']]=encode_training(r['prompt'],r['target'],vocab,args.max_tokens)
        except ValueError as exc:rejected[r['id']]=str(exc)
    changed=True
    while changed:
        changed=False
        for r in candidates:
            if r['id'] not in rejected and r['parent_id'] is not None and r['job_id']+':'+r['parent_id'] in rejected:
                rejected[r['id']]='parent_not_admitted';changed=True
    admitted=[r for r in candidates if r['id'] not in rejected]
    args.out.mkdir(parents=True,exist_ok=False)
    by_split=Counter();by_role=Counter();by_stage=Counter();atomic=Counter();max_length=0
    for split in ('train','dev','holdout'):
        with (args.out/(split+'.jsonl')).open('x') as f:
            for r in sorted(admitted,key=lambda x:x['id']):
                if r['split']!=split:continue
                row=dict(r,**encoded[r['id']]);f.write(json.dumps(row,ensure_ascii=False)+'\n')
                by_split[split]+=1;by_role[(split,r['state_role'])]+=1;by_stage[(split,r['stage'])]+=1
                max_length=max(max_length,len(row['input_ids']))
                if r['kind']=='atomic':
                    x=r['raw_item'];value=x['answer']['value']
                    atomic[(split,x['field']['value_type'])]+=1
                    if value is None:atomic[(split,'null')]+=1
                    if value is False:atomic[(split,'false')]+=1
                    if value is True:atomic[(split,'true')]+=1
                    if x['answer']['source_scope'] is not None:atomic[(split,'scoped')]+=1
    def counters(c):return {'/'.join(k) if isinstance(k,tuple) else k:v for k,v in sorted(c.items())}
    gates={'minimum_2000_train':by_split['train']>=2000,
           'normal_retention_present':by_stage[('train','normal')]>=400,
           'defect_examples_present':by_stage[('train','defect')]>=200,
           'progression_examples_present':by_stage[('train','progress')]>=200,
           'resolver_missing_values_present':atomic[('train','null')]>=20,
           'resolver_explicit_false_present':atomic[('train','false')]>=20,
           'resolver_scoped_values_present':atomic[('train','scoped')]>=20,
           'dev_and_holdout_present':by_split['dev']>=100 and by_split['holdout']>=100}
    summary={'candidate_count':len(records),'admitted':len(admitted),'by_split':counters(by_split),
             'by_role':counters(by_role),'by_stage':counters(by_stage),'atomic':counters(atomic),
             'source_families':len({f for r in admitted for f in r['source_families']}),
             'max_sequence_tokens':max_length,'gates':gates,'training_admitted':all(gates.values()),
             'independent_review':False,'semantic_dedup_exhaustive':False,
             'rejected_reasons':dict(Counter(rejected.values())),
             'files':{p.name:sha(p.read_bytes()) for p in args.out.glob('*.jsonl')}}
    (args.out/'REJECTIONS.json').write_text(json.dumps(rejected,ensure_ascii=False,indent=2)+'\n')
    (args.out/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,nargs='+',required=True)
    p.add_argument('--audit',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--exclusions',type=Path);p.add_argument('--max-tokens',type=int,default=8192)
    main(p.parse_args())
