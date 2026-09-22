"""Collect official GitHub README blobs; preserve bytes and split repository families before labels."""
import base64
import hashlib
import json
from pathlib import Path
import urllib.request
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/corpora/progression-engineering-20260922'
FAMILIES=[
 ('workflow','train',['apache/airflow','PrefectHQ/prefect','dagster-io/dagster','kestra-io/kestra']),
 ('vector','train',['qdrant/qdrant','milvus-io/milvus','weaviate/weaviate','chroma-core/chroma']),
 ('observability','train',['prometheus/prometheus','grafana/grafana','VictoriaMetrics/VictoriaMetrics','jaegertracing/jaeger']),
 ('web','train',['fastapi/fastapi','pallets/flask','django/django','litestar-org/litestar']),
 ('packaging','dev',['astral-sh/uv','python-poetry/poetry','pdm-project/pdm','pypa/pipenv']),
 ('analytics','holdout',['apache/superset','metabase/metabase','getredash/redash','lightdash/lightdash']),
]

def chunks(text):
    # Complete paragraphs (including tables/lists) with the previous full paragraph as overlap.
    paragraphs=text.split('\n\n');result=[];current=[];size=0
    for paragraph in paragraphs:
        if size+len(paragraph)+2>5500 and current:
            result.append('\n\n'.join(current));current=current[-1:] if len(current[-1])<1500 else []
            size=sum(len(p)+2 for p in current)
        current.append(paragraph);size+=len(paragraph)+2
    if current:result.append('\n\n'.join(current))
    result=list(dict.fromkeys(x for x in result if 250<=len(x)<=12000))
    for value in result:
        if value not in text:raise ValueError('Chunk is not a literal contiguous README span')
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'readmes').mkdir()
    manifest=[];jobs=[]
    for family,split,repos in FAMILIES:
        documents=[]
        for repo in repos:
            url='https://api.github.com/repos/'+repo+'/readme'
            request=urllib.request.Request(url,headers={'Accept':'application/vnd.github+json','User-Agent':'rwkvrag-training-source-audit'})
            with urllib.request.urlopen(request,timeout=60) as response:record=json.load(response)
            raw=base64.b64decode(record['content']);text=raw.decode('utf-8')
            if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=record['sha']:
                raise ValueError('Git blob identity mismatch')
            path=OUT/'readmes'/(repo.replace('/','--')+'.md');path.write_bytes(raw)
            parts=chunks(text)
            if not parts:raise ValueError('No complete eligible source chunks: '+repo)
            source={'repo':repo,'blob_sha':record['sha'],'sha256':hashlib.sha256(raw).hexdigest(),
                'path':str(path.relative_to(ROOT)),'url':'https://api.github.com/repos/'+repo+'/git/blobs/'+record['sha'],
                'html_url':record['html_url'],'chunks':parts,'family':family,'split':split}
            documents.append(source);manifest.append({k:v for k,v in source.items() if k!='chunks'})
            print(json.dumps({'repo':repo,'characters':len(text),'complete_chunks':len(parts)}),flush=True)
            time.sleep(.5)
        combinations=set()
        for variant in range(16):
            selected=[]
            for index,doc in enumerate(documents):
                chunk=doc['chunks'][(variant//(index+1)+index)%len(doc['chunks'])]
                selected.append({'id':f'S{index+1}','title':doc['repo']+' README','text':chunk,'url':doc['url'],
                    'repo':doc['repo'],'blob_sha':doc['blob_sha'],'sha256':hashlib.sha256(chunk.encode()).hexdigest()})
            identity=hashlib.sha256(json.dumps([s['sha256'] for s in selected]).encode()).hexdigest()
            if identity in combinations:continue
            combinations.add(identity)
            jobs.append({'id':'engineering-'+family+'-'+identity[:12],'split':split,'material_mode':'real',
                'source_families':[hashlib.sha256(('engineering/'+family).encode()).hexdigest()],
                'sources':selected,'status':'unlabelled','defect_evidence':'real_multi_project_comparison'})
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    # Interleave independent families; all variants of a repository remain in one split.
    jobs.sort(key=lambda j:hashlib.sha256(j['id'].encode()).hexdigest())
    (OUT/'JOBS.jsonl').write_text(''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs))
    print(json.dumps({'jobs':len(jobs),'repositories':len(manifest),'families':len(FAMILIES)}))

if __name__=='__main__':main()
