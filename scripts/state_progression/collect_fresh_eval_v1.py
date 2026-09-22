"""Freeze new official README material and human-authored questions before State training."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'llamaindex-retrieval/eval/state-fresh-github-20260922-v1'
FAMILIES={
 'http_clients':['psf/requests','encode/httpx','aio-libs/aiohttp','urllib3/urllib3'],
 'analytics_stack':['pandas-dev/pandas','pola-rs/polars','duckdb/duckdb','apache/arrow'],
 'python_quality':['astral-sh/ruff','psf/black','PyCQA/isort','python/mypy'],
}
QUESTIONS={
 'http_clients':[
  'Requests、HTTPX、aiohttp、urllib3 如果都用来发 HTTP 请求，README 对各自定位和同步／异步支持到底说了什么？请逐个比较并指出证据不足的地方。',
  '我需要一个异步 Python HTTP 客户端，还希望保留同步调用的可能。四个项目的 README 能支持选哪个，不能支持哪些判断？',
  '这四个项目里哪些 README 明确说到 HTTP/2？若我还要求异步，依据现有资料能下结论吗？',
  'urllib3 与 Requests 在 README 里的关系是什么？能否据此断言其中一个在所有场景更快？',
 ],
 'analytics_stack':[
  'pandas、Polars、DuckDB、Arrow 放在一起时，README 分别将自己定位为什么？请比较数据表、SQL 和列式格式这些方向。',
  '我手上有 CSV，想运行 SQL 并继续用 Python 处理。四个项目的 README 哪些能提供直接线索，哪些需求仍要进一步查？',
  '如果重点是 Python DataFrame API，pandas、Polars、DuckDB 和 Arrow 的 README 各有什么明确说明？',
  '这四个项目的 README 是否足以证明某个项目在我自己的数据上最快？如果不足，哪些性能表述只能视为项目主张？',
 ],
 'python_quality':[
  'Ruff、Black、isort、mypy 在格式化、导入排序、静态类型检查上各自做什么？请只依据四份 README 比较。',
  '我想给 Python 项目同时做代码格式化、导入排序和类型检查；四个 README 能支持什么组合，哪里还需要验证？',
  'Ruff 的 README 是否足以证明它能完全替代 Black、isort 和 mypy 的所有功能？请逐项区分。',
  '如果只想先用一个工具改善代码风格，这四个项目的 README 分别给出什么定位？别把速度宣传当成实测结论。',
 ]}

def sha(raw):return hashlib.sha256(raw).hexdigest()

def fetch(repo):
    p=subprocess.run(['gh','api','repos/'+repo+'/readme'],capture_output=True,text=True,timeout=60)
    if p.returncode:raise RuntimeError('Official README request failed: '+repo)
    r=json.loads(p.stdout);raw=base64.b64decode(r['content'])
    if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=r['sha']:
        raise ValueError('Git blob mismatch: '+repo)
    return r,raw

def main():
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'readmes').mkdir()
    manifest=[];cases=[]
    for family,repos in FAMILIES.items():
        evidence=[]
        for i,repo in enumerate(repos,1):
            r,raw=fetch(repo)
            text=raw.decode('utf-8')
            # Keep a single literal prefix: this is fixed evidence, not an edited summary.
            snippet=text[:5000]
            path=OUT/'readmes'/(repo.replace('/','--')+'.md');path.write_bytes(raw)
            manifest.append({'repo':repo,'family':family,'blob_sha':r['sha'],'sha256':sha(raw),
                             'readme_path':str(path.relative_to(ROOT)),'snippet_sha256':sha(snippet.encode()),
                             'snippet_characters':len(snippet),'blob_uri':'https://api.github.com/repos/'+repo+'/git/blobs/'+r['sha']})
            evidence.append({'label':f'资料 {i}','title':repo+' README','text':snippet})
            print(json.dumps({'repo':repo,'characters':len(text),'evidence_characters':len(snippet)}),flush=True)
        for i,question in enumerate(QUESTIONS[family],1):
            cases.append({'id':f'{family}-comparison-{i:02d}','suite':'fresh_github','category':'comparison',
                          'state_role':'writer','question':question,'history':[],
                          'evidence':evidence,'evidence_format':'title_and_literal_readme_prefix',
                          'evaluation_kind':'new_source_separated_manual_questions_no_teacher_labels',
                          'source_family':family})
        for i,repo in enumerate(repos,1):
            question=f'{repo} 的 README 如何介绍这个项目的定位和主要用途？请标明来源，没写清楚的不要推断。'
            cases.append({'id':f'{family}-ordinary-{i:02d}','suite':'fresh_github','category':'ordinary',
                          'state_role':'writer','question':question,'history':[],
                          'evidence':[evidence[i-1]],'evidence_format':'title_and_literal_readme_prefix',
                          'evaluation_kind':'new_source_separated_manual_questions_no_teacher_labels',
                          'source_family':family})
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (OUT/'CASES.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases))
    summary={'repositories':len(manifest),'cases':len(cases),'ordinary':12,'comparison':12,
             'training_source_repositories_overlap':0,'question_author':'human-authored by project agent, not teacher',
             'raw_answers_or_gold_present':False,'material_frozen_before_training':True,
             'cases_sha256':sha((OUT/'CASES.jsonl').read_bytes())}
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
