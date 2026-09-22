"""Versioned conservative admission exclusions; never alter generated labels or audit receipts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    exclusions=json.loads(args.isolation.read_text());counts=Counter();job_ids=set()
    for path in sorted(args.engineering.glob('*/results/*.json')):
        record=json.loads(path.read_text())
        if record['job_id'] in job_ids:raise ValueError('Duplicate engineering job')
        job_ids.add(record['job_id'])
        for item in record.get('accepted',[]):
            if item['stage']!='progress':continue
            counts['all_engineering_progress_candidates']+=1
            if item['id'] in exclusions:counts['already_isolated']+=1
            else:
                exclusions[item['id']]='conservative_engineering_progress_quarantine_pending_independent_relevance_review'
                counts['additional_quarantine']+=1
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'EXCLUSIONS.json').write_text(json.dumps(exclusions,ensure_ascii=False,indent=2)+'\n')
    report={'isolation_sha256':sha(args.isolation),'engineering_results':len(job_ids),
            'engineering_progress':dict(counts),'total_excluded_items':len(exclusions),
            'basis':'Manual stratified review found accepted progress using newly added irrelevant project documents; quarantine entire engineering progress stratum for first State run.',
            'other_progression_candidates': 'Original V2 and atomic V3 remain subject to strict audit and export gates.'}
    (args.out/'SUMMARY.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--isolation',type=Path,required=True)
    p.add_argument('--engineering',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
