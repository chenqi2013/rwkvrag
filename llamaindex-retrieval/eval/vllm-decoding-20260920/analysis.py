from collections import defaultdict
import random
from statistics import mean

from protocol import metrics

SEEDS = (11, 29, 47, 71, 101)


def correct(row):
    return row['status'] == 'valid' and row['prediction'] == row['expected']


def summarize(rows, inputs):
    cases = [v['case'] for v in inputs]
    keys = {(c['id'], arm, seed) for c in cases for arm, seeds in
            [('top1', (11, 101)), ('fake', SEEDS)] for seed in seeds}
    indexed = {(r['case_id'], r['arm'], r['seed']): r for r in rows}
    assert len(indexed) == len(rows) and set(indexed) == keys
    group = lambda arm, seed: [indexed[c['id'], arm, seed] for c in cases]
    baseline = group('top1', 11)
    base_metrics = metrics(baseline)
    fake = {str(s): metrics(group('fake', s)) for s in SEEDS}
    delta = {c['id']: mean(int(correct(indexed[c['id'], 'fake', s])) for s in SEEDS)
                        - int(correct(indexed[c['id'], 'top1', 11])) for c in cases}
    families = defaultdict(list)
    for c in cases:
        families[c['family']].append(c['id'])
    assert len({len(v) for v in families.values()}) == 1
    family_delta = {f: mean(delta[i] for i in ids) for f, ids in families.items()}
    names = sorted(families)
    rng = random.Random(20260920)
    boot = sorted(mean(family_delta[rng.choice(names)] for _ in names) for _ in range(10000))
    interval = [boot[249], boot[9749]]
    repeat = group('top1', 101)
    repeat_agreement = mean((a['status'], a['prediction']) == (b['status'], b['prediction'])
                            for a, b in zip(baseline, repeat))
    raw_agreement = mean(a['raw_text'] == b['raw_text'] and a.get('token_ids') == b.get('token_ids')
                         for a, b in zip(baseline, repeat))
    criteria = {
        'mean_gain_at_least_5_points': mean(delta.values()) >= .05,
        'family_interval_above_zero': interval[0] > 0,
        'no_seed_increases_fp': all(v['fp'] <= base_metrics['fp'] for v in fake.values()),
        'no_seed_increases_invalid': all(v['invalid'] <= base_metrics['invalid'] for v in fake.values()),
        'no_seed_reduces_accuracy': all(v['accuracy_including_invalid'] >= base_metrics['accuracy_including_invalid'] for v in fake.values()),
        'top1_labels_identical': repeat_agreement == 1,
    }
    averages = {k: mean(v[k] for v in fake.values()) for k in
        ('accuracy_including_invalid', 'recall', 'fp', 'fn', 'invalid', 'pair_accuracy')}
    categories = {}
    for cat in sorted({c['category'] for c in cases}):
        ids = {c['id'] for c in cases if c['category'] == cat}
        categories[cat] = {'top1': metrics([r for r in baseline if r['case_id'] in ids]),
            'fake_by_seed': {str(s): metrics([r for r in group('fake', s) if r['case_id'] in ids]) for s in SEEDS}}
    efficiency = {}
    for arm in ('top1', 'fake'):
        selected = [r for r in rows if r['arm'] == arm]
        efficiency[arm] = {'mean_ms': mean(r['elapsed_ms'] for r in selected),
            'mean_completion_tokens': mean(r['completion_tokens'] for r in selected),
            'length_exits': sum(r['finish_reason'] == 'length' for r in selected)}
    return {'unique_seen_regression_cases': len(cases), 'families': len(families), 'calls': len(rows),
        'baseline': base_metrics, 'top1_repeat': metrics(repeat), 'fake_by_seed': fake,
        'fake_mean': averages, 'mean_accuracy_delta': mean(delta.values()),
        'family_cluster_95_percentile_interval': interval, 'family_deltas': family_delta,
        'top1_classification_agreement': repeat_agreement, 'top1_raw_and_token_agreement': raw_agreement,
        'criteria': criteria, 'limited_regression_improvement_gate': all(criteria.values()),
        'categories': categories, 'efficiency': efficiency,
        'unseen_validation': False, 'production_promotion': False}
