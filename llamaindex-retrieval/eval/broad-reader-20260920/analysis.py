from collections import defaultdict
from statistics import mean
import random

from protocol import metrics


def correct(row):return row['status']=='valid' and row['prediction']==row['expected']


def summarize(rows,inputs,prior):
    cases=[v['case'] for v in inputs];by_id={v['id']:v for v in cases}
    expected={(c['id'],a,n) for c in cases for a in ('json','multiline') for n in (1,2)}
    index={(r['case_id'],r['arm'],r['round']):r for r in rows}
    assert len(index)==len(rows) and set(index)==expected
    groups={}
    for dimension in ('cohort','category','family'):
        values=defaultdict(list)
        for c in cases:values[c[dimension]].append(c['id'])
        groups[dimension]={name:{f'{a}-{n}':metrics([index[i,a,n] for i in ids])
            for a in ('json','multiline') for n in (1,2)} for name,ids in values.items()}
    stability={}
    for arm in ('json','multiline'):
        pairs=[(index[c['id'],arm,1],index[c['id'],arm,2]) for c in cases]
        stability[arm]={'classification':mean((a['status'],a['prediction'])==(b['status'],b['prediction']) for a,b in pairs),
            'raw_tokens':mean(a['raw_text']==b['raw_text'] and a['token_ids']==b['token_ids'] for a,b in pairs)}
    changes=[];old_losses=[]
    for c in cases:
        for n in (1,2):
            a=index[c['id'],'json',n];b=index[c['id'],'multiline',n]
            delta=int(correct(b))-int(correct(a))
            if delta:changes.append({'case_id':c['id'],'cohort':c['cohort'],'category':c['category'],
                                     'round':n,'delta':delta,'json':a,'multiline':b})
            if c['cohort']!='new960' and delta<0:old_losses.append({'case_id':c['id'],'round':n})
    family_deltas=defaultdict(list)
    for c in cases:
        if c['cohort']=='new960':
            family_deltas[c['family']].append(int(correct(index[c['id'],'multiline',1]))-int(correct(index[c['id'],'json',1])))
    assert len(family_deltas)==40 and all(len(v)==24 for v in family_deltas.values())
    delta={k:mean(v) for k,v in family_deltas.items()};names=sorted(delta)
    rng=random.Random(20260920);boot=sorted(mean(delta[rng.choice(names)] for _ in names) for _ in range(10000))
    interval=[boot[249],boot[9749]]
    prior={r['case_id']:r for r in prior if r['arm']=='top1' and r['seed']==11}
    baselines=[index[c['id'],'json',1] for c in cases if c['cohort']=='historical160']
    prior_agreement=mean(r['raw_text']==prior[r['case_id'].split(':',1)[1]]['raw_text'] and
        r['token_ids']==prior[r['case_id'].split(':',1)[1]]['token_ids'] for r in baselines)
    check_groups=[v for dim in ('cohort','category') for v in groups[dim].values()]
    criteria={'no_historical_correct_case_lost':not old_losses,
        'no_cohort_category_fp_increase':all(v[f'multiline-{n}']['fp']<=v[f'json-{n}']['fp'] for v in check_groups for n in (1,2)),
        'no_cohort_category_invalid_increase':all(v[f'multiline-{n}']['invalid']<=v[f'json-{n}']['invalid'] for v in check_groups for n in (1,2)),
        'repeat_stability':all(v['classification']==1 for v in stability.values()),
        'new_gain_at_least_5_points':mean(delta.values())>=.05,'new_family_interval_above_zero':interval[0]>0,
        'historical160_baseline_exact':prior_agreement==1}
    return {'calls':len(rows),'dataset_rows':len(cases),'unique_case_ids':1184,
        'historical_duplicate_memberships':40,'groups':groups,'stability':stability,
        'changes':changes,'historical_correct_cases_lost':old_losses,
        'new_mean_accuracy_gain':mean(delta.values()),'new_family_95_interval':interval,
        'new_family_deltas':delta,'prior_baseline_agreement':prior_agreement,'criteria':criteria,
        'limited_regression_improvement_gate':all(criteria.values()),
        'length_exits':sum(r['finish_reason']=='length' for r in rows),'production_promotion':False,
        'independent_gold_review':False}
