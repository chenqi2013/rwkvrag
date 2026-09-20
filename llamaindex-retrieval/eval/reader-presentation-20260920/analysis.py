from collections import defaultdict
from statistics import mean
import random

from protocol import metrics


def correct(row):
    return row['status'] == 'valid' and row['prediction'] == row['expected']


def same_raw(a, b):
    return a['raw_text'] == b['raw_text'] and a['token_ids'] == b['token_ids']


def summarize(rows, inputs, prior):
    cases = [v['case'] for v in inputs]
    expected = {(c['id'], arm, round_number) for c in cases
                for arm in ('json', 'multiline') for round_number in (1, 2)}
    index = {(r['case_id'], r['arm'], r['round']): r for r in rows}
    assert len(index) == len(rows) and set(index) == expected
    group = lambda arm, number: [index[c['id'], arm, number] for c in cases]
    conditions = {f'{arm}-{number}': metrics(group(arm, number))
                  for arm in ('json', 'multiline') for number in (1, 2)}
    categories = {cat: {f'{arm}-{number}': metrics([r for r in group(arm, number)
                    if r['category'] == cat]) for arm in ('json', 'multiline') for number in (1, 2)}
                  for cat in sorted({c['category'] for c in cases})}
    stability = {arm: {'classification': mean((a['status'], a['prediction']) == (b['status'], b['prediction'])
        for a,b in zip(group(arm, 1), group(arm, 2))),
        'raw_and_tokens': mean(same_raw(a,b) for a,b in zip(group(arm, 1),group(arm, 2)))}
        for arm in ('json', 'multiline')}
    prior_index = {r['case_id']: r for r in prior if r['arm']=='top1' and r['seed']==11}
    assert set(prior_index) == {c['id'] for c in cases}
    baseline_agreement = mean(same_raw(r, prior_index[r['case_id']]) for r in group('json', 1))
    families = defaultdict(list)
    changes = []
    for a,b in zip(group('json', 1),group('multiline', 1)):
        delta = int(correct(b))-int(correct(a))
        families[a['family']].append(delta)
        if delta or (a['status'],a['prediction']) != (b['status'],b['prediction']):
            changes.append({'case_id':a['case_id'],'category':a['category'],'delta':delta,
                            'json':a,'multiline':b})
    assert len({len(values) for values in families.values()}) == 1
    family_deltas = {name:mean(values) for name,values in families.items()}
    names = sorted(family_deltas)
    rng = random.Random(20260920)
    boot = sorted(mean(family_deltas[rng.choice(names)] for _ in names) for _ in range(10000))
    interval = [boot[249],boot[9749]]
    gain = conditions['multiline-1']['accuracy_including_invalid']-conditions['json-1']['accuracy_including_invalid']
    criteria = {'gain_at_least_5_points': gain>=.05, 'family_interval_above_zero': interval[0]>0,
        'no_fp_increase':all(conditions[f'multiline-{n}']['fp']<=conditions[f'json-{n}']['fp'] for n in (1,2)),
        'no_invalid_increase':all(conditions[f'multiline-{n}']['invalid']<=conditions[f'json-{n}']['invalid'] for n in (1,2)),
        'no_category_accuracy_regression':all(v[f'multiline-{n}']['accuracy_including_invalid']>=v[f'json-{n}']['accuracy_including_invalid'] for v in categories.values() for n in (1,2)),
        'table_recall_improves':all(categories['table'][f'multiline-{n}']['recall']>categories['table'][f'json-{n}']['recall'] for n in (1,2)),
        'both_classification_stable':all(v['classification']==1 for v in stability.values()),
        'prior_json_baseline_exact':baseline_agreement==1}
    return {'calls':len(rows),'seen_cases':len(cases),'families':len(families),
        'conditions':conditions,'categories':categories,'stability':stability,
        'prior_json_raw_token_agreement':baseline_agreement,'mean_accuracy_gain':gain,
        'family_cluster_95_interval':interval,'family_deltas':family_deltas,'changes':changes,
        'criteria':criteria,'limited_regression_improvement_gate':all(criteria.values()),
        'efficiency':{arm:{'mean_ms':mean(r['elapsed_ms'] for r in rows if r['arm']==arm),
                          'length_exits':sum(r['finish_reason']=='length' for r in rows if r['arm']==arm)}
                      for arm in ('json','multiline')},
        'unseen_validation':False,'production_promotion':False}
