import argparse
import asyncio
from collections import Counter,defaultdict
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from time import perf_counter

import httpx

HERE=Path(__file__).resolve().parent
URL='http://127.0.0.1:18446'
SETTINGS=Path('data/services/restored-regression-20260920/settings.json')
COVERAGE={r['case_id']:r for r in json.loads((HERE/'COVERAGE.json').read_text())['cases']}


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2))


def diagnostics(case,data):
    answer=data.get('answer','');sources=data.get('sources',[])
    citations=[int(x) for x in re.findall(r'\[资料\s*(\d+)\]',answer)]
    original=case['original_case'];terms=original.get('expected_terms',[])
    expected=original.get('expected_document_id')
    titles=original.get('expected_titles',[])
    return {'empty_answer':not answer.strip(),'source_count':len(sources),
        'has_numeric_citation':bool(citations),'citation_out_of_range':any(i<1 or i>len(sources) for i in citations),
        'generation_status':data.get('generation',{}).get('status','unspecified'),
        'refusal_text_indicator':any(s in answer for s in ('无法确定','未检索到可用于回答','资料不足','未找到相关')),
        'legacy_document_id_returned':expected in [s.get('document_id') for s in sources] if expected else None,
        'restored_expected_document_returned':bool(set(COVERAGE[case['id']]['actual_document_ids']) & {s.get('document_id') for s in sources}),
        'expected_title_returned':any(t in s.get('title','') for t in titles for s in sources) if titles else None,
        'reference_terms_matched':sum(t in answer for t in terms) if terms else None,
        'reference_terms_total':len(terms) if terms else None,
        'semantic_accuracy_verified':False}


async def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);(args.output/'calls').mkdir()
    cases=json.loads((HERE/'cases.json').read_text())
    hashes={str(p):sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.is_file()}
    origins=json.loads((HERE/'ORIGINS.json').read_text())
    assert all(sha256(Path(p).read_bytes()).hexdigest()==h for p,h in origins.items())
    settings_hash=sha256(SETTINGS.read_bytes()).hexdigest()
    save(args.output/'BINDINGS.json',{'files':hashes,'origins':origins,'settings_sha256':settings_hash,
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'started_at':datetime.now(timezone.utc).isoformat(),'planned':len(cases)})
    async with httpx.AsyncClient(timeout=600) as client:
        async def snapshot():
            result={}
            for name,url in [('app',URL+'/health'),('model','http://127.0.0.1:18423/health'),
                ('index','http://127.0.0.1:18438/_alias/rwkvrag-restored-regression-20260920-v1-active')]:
                response=await client.get(url);response.raise_for_status();result[name]=response.json()
            return result
        before=await snapshot();save(args.output/'HEALTH-BEFORE.json',before)
        results=[];semaphore=asyncio.Semaphore(4)
        async def run(index,case):
            async with semaphore:
                began=perf_counter();record={'index':index,'case_id':case['id'],'suite':case['suite'],
                    'category':case['category'],'style':case['style'],'endpoint':case['endpoint'],
                    'payload':case['payload'],'completed':False}
                try:
                    response=await client.post(URL+case['endpoint'],json=case['payload'])
                    record.update(http_status=response.status_code,raw_response=response.text)
                    response.raise_for_status();data=response.json()
                    record.update(completed=True,diagnostics=diagnostics(case,data))
                except Exception as error:
                    record['error']={'type':type(error).__name__,'detail':str(error)}
                finally:
                    record['elapsed_ms']=round((perf_counter()-began)*1000,2)
                    save(args.output/'calls'/f'{index:04d}.json',record);results.append(record)
                    if len(results)%10==0:print(json.dumps({'completed_requests':len(results),'total':len(cases)}),flush=True)
        await asyncio.gather(*(run(i,c) for i,c in enumerate(cases)))
        try:
            after=await snapshot();save(args.output/'HEALTH-AFTER.json',after)
        except Exception as error:
            after={};save(args.output/'HEALTH-AFTER.json',{'error':str(error)})
    results.sort(key=lambda r:r['index']);save(args.output/'ROWS.json',results)
    groups={}
    for key in ('suite','category','style'):
        values=defaultdict(list)
        for r in results:values[r[key]].append(r)
        groups[key]={name:{'n':len(items),'http_completed':sum(r['completed'] for r in items),
            'empty_answers':sum(r.get('diagnostics',{}).get('empty_answer',False) for r in items),
            'with_citations':sum(r.get('diagnostics',{}).get('has_numeric_citation',False) for r in items),
            'refusal_indicators':sum(r.get('diagnostics',{}).get('refusal_text_indicator',False) for r in items),
            'generation_statuses':dict(Counter(r.get('diagnostics',{}).get('generation_status','http_failed') for r in items))}
            for name,items in values.items()}
    stable=sha256(SETTINGS.read_bytes()).hexdigest()==settings_hash and before.get('index')==after.get('index')
    assert all(sha256(Path(p).read_bytes()).hexdigest()==h for p,h in {**hashes,**origins}.items())
    save(args.output/'SUMMARY.json',{'planned':len(cases),'recorded':len(results),
        'http_completed':sum(r['completed'] for r in results),'groups':groups,
        'settings_and_index_alias_unchanged':stable,'historical_files_unchanged':True,
        'automated_diagnostics_only':True,'semantic_accuracy_verified':False,'production_promotion':False})
    print(json.dumps({'complete':True,'recorded':len(results)}),flush=True)


if __name__=='__main__':asyncio.run(main())
