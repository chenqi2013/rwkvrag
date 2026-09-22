"""Check every exported progress row against its exact accepted delta verdict."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(raw):return hashlib.sha256(raw).hexdigest()
def row_digest(row):return digest(json.dumps(row,ensure_ascii=False,sort_keys=True).encode())


def main(args):
    release_summary=json.loads((args.release/'SUMMARY.json').read_text())
    rows=[];file_pins={}
    for split in ('train','dev','holdout'):
        path=args.release/(split+'.jsonl');raw=path.read_bytes()
        if digest(raw)!=release_summary['files'][path.name]:raise ValueError('Exported release file changed')
        file_pins[path.name]=digest(raw)
        rows.extend(json.loads(line) for line in raw.splitlines())
    progress={r['id']:r for r in rows if r['stage']=='progress'}
    if len(progress)!=sum(r['stage']=='progress' for r in rows):raise ValueError('Duplicate progress ID')
    decisions={};audit_pins={}
    for root in args.audit:
        inputs_path=root/'INPUTS.json';summary_path=root/'SUMMARY.json'
        inputs=json.loads(inputs_path.read_text());summary=json.loads(summary_path.read_text())
        target_ids=inputs['target_ids']
        if len(target_ids)!=len(set(target_ids)) or summary['planned']!=len(target_ids):
            raise ValueError('Audit input membership changed')
        if summary.get('reviewed',0)+summary.get('failed',0)!=len(target_ids):
            raise ValueError('Audit did not finish every target')
        source_release=Path(inputs['release'])
        if any(digest((source_release/name).read_bytes())!=sha for name,sha in inputs['release_pins'].items()):
            raise ValueError('Audited source release changed')
        result_paths=list((root/'results').glob('*.json'))
        if len(result_paths)!=len(target_ids):raise ValueError('Audit result count changed')
        for path in result_paths:
            result=json.loads(path.read_text());identity=result['id']
            if identity not in target_ids or identity in decisions:
                raise ValueError('Unexpected or duplicate audit identity')
            decisions[identity]=result
        audit_pins[str(root)]={'inputs':digest(inputs_path.read_bytes()),'summary':digest(summary_path.read_bytes())}
    issues=[];counts=Counter()
    for identity,row in progress.items():
        decision=decisions.get(identity)
        issue=None
        if decision is None:issue='unreviewed'
        elif decision['row_sha256']!=row_digest(row):issue='reviewed_different_row'
        elif decision['status']!='reviewed' or decision['accept'] is not True:issue='rejected_or_failed_entered_release'
        if issue:issues.append({'id':identity,'split':row['split'],'issue':issue});counts[issue]+=1
        else:counts['accepted_'+row['split']]+=1
    report={'release_files_sha256':file_pins,'audit_pins':audit_pins,
            'released_progress':len(progress),'matched_accepted':len(progress)-len(issues),
            'issues':issues,'counts':dict(counts),'safe_for_progress_training':not issues,
            'limits':'Exact review binding and cardinality are structural gates, not semantic quality proof.'}
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'SUMMARY.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='issues'},ensure_ascii=False))
    if issues:raise SystemExit(2)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--audit',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
