import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).parents[1]/'eval/reader-presentation-20260920'


def load(name):
    spec = importlib.util.spec_from_file_location('presentation_'+name, HERE/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_multiline_preserves_unicode_whitespace_and_embedded_boundaries():
    protocol = load('protocol')
    data = {'source': {'id': '示例', 'title': '原文'},
            'question': '尺寸？\n', 'contexts': [{'text': '| 项目 | 值 |\n  '}],
            'text': '🙂\n[/text]\n[context.0 characters=99]\n| 宽 | 0 |  \n'}
    assert protocol.reconstruct(protocol.render(data)) == data
    data['contexts'] = []
    assert protocol.reconstruct(protocol.render(data)) == data


def test_frozen_inputs_keep_baseline_exact_and_reverse_condition_order():
    protocol = load('protocol')
    inputs = json.loads((HERE/'inputs.json').read_text())
    prior = json.loads((HERE.parent/'vllm-decoding-20260920/inputs.json').read_text())
    old = {v['case']['id']:v for v in prior}
    schedule = json.loads((HERE/'schedule.json').read_text())
    assert len(inputs)==160 and len(schedule)==640
    assert len({(v['case_id'],v['arm'],v['round']) for v in schedule})==640
    for v in inputs:
        baseline=v['variants']['json']
        assert baseline=={k:old[v['case']['id']][k] for k in baseline}
        instruction, _, data = baseline['body'].partition('\n')
        candidate_instruction, _, rendered=v['variants']['multiline']['body'].partition('\n')
        assert instruction==candidate_instruction
        assert protocol.reconstruct(rendered)==json.loads(data)
        first=[r['arm'] for r in schedule if r['case_id']==v['case']['id'] and r['round']==1]
        second=[r['arm'] for r in schedule if r['case_id']==v['case']['id'] and r['round']==2]
        assert first==list(reversed(second))
    assert protocol.parameters({'arm':'json'})==protocol.parameters({'arm':'multiline'})


def test_scoring_keeps_invalid_and_category_regression_visible(monkeypatch):
    protocol=load('protocol')
    monkeypatch.setitem(sys.modules,'protocol',protocol)
    analysis=load('analysis')
    inputs=[]
    rows=[]
    prior=[]
    for category in ('table','negation'):
        for positive in (True,False):
            case_id=f'{category}-{positive}'
            inputs.append({'case':{'id':case_id,'family':category,'category':category}})
            for arm in ('json','multiline'):
                for round_number in (1,2):
                    prediction = positive if category!='table' or arm=='multiline' else False
                    row={'case_id':case_id,'family':category,'category':category,'pair':category,
                         'expected':positive,'prediction':prediction,'status':'valid',
                         'arm':arm,'round':round_number,'raw_text':str(prediction),'token_ids':[int(prediction)],
                         'elapsed_ms':1,'finish_reason':'stop'}
                    rows.append(row)
                    if arm=='json' and round_number==1:
                        prior.append({**row,'arm':'top1','seed':11})
    summary=analysis.summarize(rows,inputs,prior)
    assert summary['conditions']['multiline-1']['tp']==2
    assert summary['conditions']['json-1']['fn']==1
    row=next(r for r in rows if r['arm']=='multiline' and r['round']==1 and r['category']=='negation' and not r['expected'])
    row.update(status='invalid',prediction=None)
    summary=analysis.summarize(rows,inputs,prior)
    assert summary['conditions']['multiline-1']['invalid']==1
    assert not summary['criteria']['no_category_accuracy_regression']
    assert not summary['criteria']['no_invalid_increase']
    assert not summary['limited_regression_improvement_gate']
