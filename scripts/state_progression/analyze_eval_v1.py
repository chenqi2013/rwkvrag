"""Mechanical paired diagnostics; leaves semantic judgment to separate source-grounded review."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repeated_span(value,size=40):
    text=re.sub(r'\s+','',value)
    first={}
    for pos in range(max(0,len(text)-size+1)):
        chunk=text[pos:pos+size]
        old=first.get(chunk)
        if old is not None and pos-old>=size:
            return {'span':chunk,'first':old,'second':pos}
        first.setdefault(chunk,pos)
    return None


def cite_issue(value,evidence):
    labels={f'资料 {i}' for i in range(1,len(evidence)+1)}
    found=re.findall(r'\[资料\s*\d+\]',value)
    bad=[x for x in found if x[1:-1].strip() not in labels]
    malformed=re.findall(r'\[资料[^\]]*\]',value)
    malformed=[x for x in malformed if x not in found]
    return {'citation_count':len(found),'invalid_citations':bad+malformed}


def main(args):
    if sha(args.cases)!=json.loads((args.eval/'RUN.json').read_text())['inputs_sha256']:
        raise ValueError('Evaluation case binding changed')
    cases=[json.loads(x) for x in args.cases.read_text().splitlines()]
    records=[json.loads(p.read_text()) for p in (args.eval/'records').glob('*.json')]
    expected=len(cases)*4
    if len(records)!=expected or len({(r['id'],r['round'],r['arm']) for r in records})!=expected:
        raise ValueError('Incomplete or duplicate paired evaluation')
    record_map={(r['id'],r['round'],r['arm']):r for r in records}
    totals=Counter();pairs=[];issues=[]
    for case in cases:
        for round_no in (1,2):
            outputs={arm:record_map[(case['id'],round_no,arm)] for arm in ('zero','trained')}
            flags={}
            for arm,row in outputs.items():
                citations=cite_issue(row['raw_text'],case.get('evidence',[]))
                repeat=repeated_span(row['raw_text'])
                flags[arm]=dict(citations,repeat=repeat,status=row['status'],output_tokens=len(row['generated_ids']))
                totals[arm+'/'+case['category']+'/'+row['status']]+=1
                if citations['invalid_citations']:totals[arm+'/invalid_citation_records']+=1
                if repeat:totals[arm+'/repeated_span_records']+=1
            if flags['trained']['invalid_citations'] and not flags['zero']['invalid_citations']:
                issues.append({'id':case['id'],'round':round_no,'issue':'new_invalid_citation',
                               'zero':flags['zero'],'trained':flags['trained']})
            if flags['trained']['repeat'] and not flags['zero']['repeat']:
                issues.append({'id':case['id'],'round':round_no,'issue':'new_repetition',
                               'zero':flags['zero'],'trained':flags['trained']})
            if outputs['zero']['status']=='stop' and outputs['trained']['status']!='stop':
                issues.append({'id':case['id'],'round':round_no,'issue':'new_nonstop',
                               'zero':flags['zero'],'trained':flags['trained']})
            pairs.append({'id':case['id'],'suite':case['suite'],'category':case['category'],'round':round_no,
                          'zero':flags['zero'],'trained':flags['trained']})
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'PAIRS.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in pairs))
    (args.out/'ISSUES.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in issues))
    summary={'cases':len(cases),'rounds':2,'records':len(records),'mechanical_counts':dict(totals),
             'new_mechanical_issues':dict(Counter(x['issue'] for x in issues)),
             'semantic_correctness_reviewed':False,'source_support_reviewed':False,
             'limits':'Citation syntax and repeated spans are diagnostics, not answer correctness.'}
    (args.out/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--eval',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
