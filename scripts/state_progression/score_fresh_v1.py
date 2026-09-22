"""Summarize complete manual review of frozen 24-case paired raw replies."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


GRADES={'runtime_failure':-1,'wrong':0,'partial':1,'full':2}


def digest(raw):return hashlib.sha256(raw).hexdigest()


def main(args):
    cases=[json.loads(line) for line in args.cases.read_text().splitlines()]
    if len(cases)!=24 or any(x['suite']!='fresh_github' for x in cases):
        raise ValueError('Only the frozen 24 fresh GitHub cases may be scored here')
    run=json.loads((args.eval/'RUN.json').read_text())
    if run['inputs_sha256']!=digest(args.cases.read_bytes()):raise ValueError('Evaluation input changed')
    raw=[json.loads(p.read_text()) for p in (args.eval/'records').glob('*.json')]
    expected={(case['id'],round_no,arm) for case in cases for round_no in (1,2) for arm in ('zero','trained')}
    actual={(r['id'],r['round'],r['arm']):r for r in raw}
    if len(raw)!=len(expected) or set(actual)!=expected:raise ValueError('Incomplete raw paired outputs')
    reviews=[json.loads(line) for line in args.reviews.read_text().splitlines()]
    judged={}
    for row in reviews:
        if set(row)!={'id','round','arm','raw_sha256','grade','reason'}:
            raise ValueError('Review fields incomplete or extra')
        key=(row['id'],row['round'],row['arm'])
        if key not in expected or key in judged or row['grade'] not in GRADES:
            raise ValueError('Review membership, duplication or grade invalid')
        if not isinstance(row['reason'],str) or len(row['reason'].strip())<8:
            raise ValueError('Manual reason missing')
        if row['raw_sha256']!=digest(actual[key]['raw_text'].encode()):
            raise ValueError('Review refers to changed raw answer')
        judged[key]=row
    if set(judged)!=expected:raise ValueError('Not every zero/trained round was manually reviewed')
    counts=Counter();pairs=[];regressions=[];gains=[]
    for case in cases:
        identity=case['id'];category=case['category'];values={}
        for round_no in (1,2):
            pair={arm:judged[(identity,round_no,arm)] for arm in ('zero','trained')}
            values[round_no]=pair
            for arm,row in pair.items():counts[f'{category}/r{round_no}/{arm}/{row["grade"]}']+=1
            pairs.append({'id':identity,'category':category,'round':round_no,
                'zero':pair['zero']['grade'],'trained':pair['trained']['grade']})
        changed=[GRADES[values[n]['trained']['grade']]-GRADES[values[n]['zero']['grade']] for n in (1,2)]
        if any(delta<0 for delta in changed):regressions.append({'id':identity,'deltas':changed})
        if all(delta>=0 for delta in changed) and any(delta>0 for delta in changed) and \
                all(values[n]['trained']['grade'] not in {'wrong','runtime_failure'} for n in (1,2)):
            gains.append({'id':identity,'deltas':changed})
    summary={'cases':len(cases),'reviews':len(judged),'counts':dict(sorted(counts.items())),
        'new_regression_cases':regressions,'stable_gain_cases':gains,
        'all_rounds_and_arms_reviewed':True,'reviewer':'implementer_not_independent',
        'evaluation_kind':'fixed_evidence_not_live_retrieval',
        'raw_run_sha256':digest((args.eval/'RUN.json').read_bytes()),
        'review_sha256':digest(args.reviews.read_bytes()),
        'limits':'Manual judgment may still be wrong; pairwise grades do not prove production quality.'}
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'PAIRS.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in pairs))
    (args.out/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--eval',type=Path,required=True)
    p.add_argument('--reviews',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
