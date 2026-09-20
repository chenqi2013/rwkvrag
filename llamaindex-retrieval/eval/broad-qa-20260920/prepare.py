import json
from hashlib import sha256
from pathlib import Path
import random

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def load(path):
    return ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if path.suffix=='.jsonl' else json.loads(path.read_text()))


def main():
    groups=[('wiki_verified','wiki_qa_200_diverse_cases_verified.json','ask'),
        ('wiki_batch2','wiki_qa_200_diverse_cases_batch2.json','ask'),
        ('basic','cases.jsonl','ask'),('diverse','diverse_cases.jsonl','ask'),
        ('native_wiki','native-wiki/fixtures.jsonl','ask'),
        ('writer_holdout','writer-p0-20260916/holdout.jsonl','material'),
        ('matrix_validation','task-matrix-20260920/data-v4/validation.cases.jsonl','material'),
        ('matrix_holdout','task-matrix-20260920/data-v4/holdout.cases.jsonl','material'),
        ('atomic_fixtures','atomic-evidence-20260920/fixtures.jsonl','atomic'),
        ('atomic_holdout','atomic-evidence-20260920/holdout.jsonl','atomic')]
    rows=[];sources={}
    for suite,name,kind in groups:
        path=ROOT/name;sources[str(path)]=sha256(path.read_bytes()).hexdigest()
        for index,case in enumerate(load(path)):
            item={'id':f'{suite}-{index:04d}','suite':suite,'origin':str(path),
                  'origin_index':index,'original_case':case,
                  'category':case.get('question_type',case.get('case_type',case.get('kind',suite))),
                  'style':case.get('style','unspecified')}
            if kind=='ask':
                item.update(endpoint='/v1/ask',payload={'question':case['question'],
                    'history':case.get('history',[]),'knowledge_base_id':'default',
                    'retrieval_mode':'knowledge_base','top_k':3})
            else:
                if kind=='atomic':
                    r=case['request'];question=f"{r['object']}的{r['attribute']}是什么？"
                    if r.get('conditions'):question+=f"所问条件：{r['conditions']}。"
                    materials=[{'id':f"{item['id']}-s{i}",'document_id':f"{item['id']}-d{i}",
                        'source':'synthetic-regression','title':'原始测试材料','snippet':text,
                        'score':1.0,'metadata':{'synthetic':True}} for i,text in enumerate(case['texts'])]
                else:
                    question=case['question'];materials=case.get('materials',case.get('candidates'))
                assert isinstance(materials,list) and len(materials)<=20
                item.update(endpoint='/v1/material-ask',payload={'question':question,
                    'history':case.get('history',[]),'materials':materials})
            rows.append(item)
    assert len(rows)==478
    random.Random(20260920).shuffle(rows)
    for name,value in [('cases.json',rows),('ORIGINS.json',sources)]:
        with (HERE/name).open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2))
    print(json.dumps({'requests':len(rows),'retrieval':sum(r['endpoint']=='/v1/ask' for r in rows),
        'fixed_material':sum(r['endpoint']=='/v1/material-ask' for r in rows)}))


if __name__=='__main__':main()
