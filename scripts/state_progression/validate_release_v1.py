"""Pre-registered quantity and diversity gates for the first isolated State release."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def norm(value):return re.sub(r'\s+','',unicodedata.normalize('NFKC',value).casefold())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    summary=json.loads((args.release/'SUMMARY.json').read_text())
    excluded=json.loads(args.exclusions.read_text())
    rows={split:[json.loads(x) for x in (args.release/(split+'.jsonl')).read_text().splitlines()]
          for split in ('train','dev','holdout')}
    if any(sha(args.release/(split+'.jsonl'))!=summary['files'][split+'.jsonl'] for split in rows):
        raise ValueError('Release file changed after export')
    all_rows=[r for group in rows.values() for r in group]
    if len({r['id'] for r in all_rows})!=len(all_rows):raise ValueError('Duplicate release IDs')
    if {r['id'] for r in all_rows}&set(excluded):raise ValueError('Excluded candidate entered release')
    family_splits={}
    for split,group in rows.items():
        for r in group:
            if r['split']!=split:raise ValueError('Release split mismatch')
            for f in r['source_families']:
                if f in family_splits and family_splits[f]!=split:raise ValueError('Source family crossed release split')
                family_splits[f]=split
    train=rows['train'];family_counts=Counter(f for r in train for f in r['source_families'])
    questions={norm((r['raw_item'].get('question') or r['raw_item'].get('field',{}).get('question') or '')) for r in train}
    engineering_multisource=sum(r['job_id'].startswith('engineering-') and r['kind']=='writer' and
                                len({e['source_id'] for e in r['raw_item']['evidence']})>=3 for r in train)
    gates={'export_acceptance':summary['training_admitted'],
           'train_examples_at_least_2000':len(train)>=2000,
           'train_source_families_at_least_400':len(family_counts)>=400,
           'largest_source_family_at_most_6pct':max(family_counts.values(),default=0)/max(1,len(train))<=.06,
           'unique_normalized_questions_at_least_85pct':len(questions)/max(1,len(train))>=.85,
           'real_multi_project_writer_at_least_30':engineering_multisource>=30,
           'all_exclusions_absent':not ({r['id'] for r in all_rows}&set(excluded))}
    report={'gates':gates,'ready_for_training':all(gates.values()),
            'counts':{'train':len(train),'dev':len(rows['dev']),'holdout':len(rows['holdout']),
                      'train_source_families':len(family_counts),'unique_train_questions':len(questions),
                      'largest_source_family_examples':max(family_counts.values(),default=0),
                      'engineering_writer_three_or_more_sources':engineering_multisource},
            'release_summary_sha256':sha(args.release/'SUMMARY.json'),
            'exclusions_sha256':sha(args.exclusions),
            'limits':'Diversity and source count are admission checks, not semantic accuracy or live retrieval quality.'}
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'SUMMARY.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))
    if not report['ready_for_training']:raise SystemExit(2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True)
    p.add_argument('--exclusions',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
