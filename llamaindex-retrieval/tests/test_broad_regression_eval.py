from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from collections import Counter

ROOT=Path(__file__).parents[1]/'eval'
HERE=ROOT/'broad-reader-20260920'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def test_all_historical_rows_and_empty_application_fixtures_retained():
    rows=json.loads((HERE/'cases.json').read_text())
    old=[r for r in rows if r['case']['cohort']!='new960']
    assert len(old)==264
    assert len({r['original_case']['id'] for r in old})==224
    for r in old:
        lines=Path(r['origin']).read_text().splitlines()
        assert r['original_case']==json.loads(lines[r['origin_index']])
        assert r['case']['expected']==r['original_case']['expected']
        assert r['case']['text']==r['original_case']['text']
        assert r['case']['question']==r['original_case']['question']
    ledger=json.loads((HERE/'HISTORY-RETENTION.json').read_text())['files']
    assert len(ledger)==352
    assert all(Path(p).exists() and sha256(Path(p).read_bytes()).hexdigest()==v['sha256'] for p,v in ledger.items())
    qa=json.loads((ROOT/'broad-qa-20260920/cases.json').read_text())
    assert len(qa)==478
    assert sum(r['endpoint']=='/v1/ask' for r in qa)==434
    assert sum(r['endpoint']=='/v1/material-ask' and not r['payload']['materials'] for r in qa)==3
    assert all(r['payload']['question']==r['original_case']['question'] for r in qa if 'question' in r['original_case'])


def test_new_family_coverage_and_exact_old_model_inputs():
    rows=json.loads((HERE/'cases.json').read_text());new=[r['case'] for r in rows if r['case']['cohort']=='new960']
    assert len(new)==960
    assert len({c['family'] for c in new})==40
    pairs=Counter(c['pair'] for c in new);assert len(pairs)==480 and set(pairs.values())=={2}
    assert sum(c['expected'] for c in new)==480
    inputs=json.loads((HERE/'inputs.json').read_text())
    schedule=json.loads((HERE/'schedule.json').read_text())
    assert len(schedule)==4896
    assert len({(r['case_id'],r['arm'],r['round']) for r in schedule})==4896
    old={v['case']['id']:v['variants'] for v in json.loads((ROOT/'reader-presentation-20260920/inputs.json').read_text())}
    for v in inputs:
        if v['case']['cohort']=='historical160':assert v['variants']==old[v['case']['id'].split(':',1)[1]]
        assert all(len(arm['prompt_token_ids'])+32<=4096 for arm in v['variants'].values())


def test_previous_successes_and_invalid_outputs_cannot_be_hidden(monkeypatch):
    protocol=load(HERE/'protocol.py','broad_protocol');monkeypatch.setitem(sys.modules,'protocol',protocol)
    analysis=load(HERE/'analysis.py','broad_analysis')
    inputs=json.loads((HERE/'inputs.json').read_text())
    rows=[];prior=[]
    for item in inputs:
        c=item['case']
        for arm in ('json','multiline'):
            for n in (1,2):
                rows.append({'case_id':c['id'],'arm':arm,'round':n,'seed':11,'expected':c['expected'],
                    'prediction':c['expected'],'status':'valid','family':c['family'],'category':c['category'],
                    'pair':c['pair'],'raw_text':str(c['expected']),'token_ids':[int(c['expected'])],
                    'finish_reason':'stop'})
        if c['cohort']=='historical160':prior.append({**rows[-1],'case_id':c['id'].split(':',1)[1],'arm':'top1','seed':11})
    loss=next(r for r in rows if r['case_id'].startswith('historical40:') and r['arm']=='multiline' and r['round']==1)
    loss['prediction']=not loss['expected']
    invalid=next(r for r in rows if r['case_id'].startswith('new:') and r['arm']=='multiline' and r['round']==1)
    invalid.update(status='invalid',prediction=None)
    summary=analysis.summarize(rows,inputs,prior)
    assert not summary['criteria']['no_historical_correct_case_lost']
    assert not summary['criteria']['no_cohort_category_invalid_increase']
    assert summary['groups']['cohort']['new960']['multiline-1']['invalid']==1
    assert not summary['limited_regression_improvement_gate']


def test_qa_diagnostics_do_not_claim_semantic_correctness():
    runner=load(ROOT/'broad-qa-20260920/run.py','broad_qa_runner')
    result=runner.diagnostics({'original_case':{'expected_document_id':'wanted'}},
        {'answer':'一个错误的结论[资料9]','sources':[{'document_id':'other'}],'generation':{'status':'completed'}})
    assert result['citation_out_of_range'] and not result['expected_document_returned']
    assert not result['semantic_accuracy_verified']
