"""Join exact progress review rows and fail closed before releasing State data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(raw):return hashlib.sha256(raw).hexdigest()
def row_digest(row):return digest(json.dumps(row,ensure_ascii=False,sort_keys=True).encode())


def main(args):
    release_summary=json.loads((args.release/'SUMMARY.json').read_text())
    rows={}
    pins={}
    for split in ('train','dev','holdout'):
        path=args.release/(split+'.jsonl');raw=path.read_bytes()
        if digest(raw)!=release_summary['files'][path.name]:raise ValueError('Release file changed')
        pins[path.name]=digest(raw)
        for line in raw.splitlines():
            row=json.loads(line)
            if row['id'] in rows:raise ValueError('Duplicate release row')
            rows[row['id']]=row
    progress={key:row for key,row in rows.items() if row['stage']=='progress'}
    reviewed={};audit_pins={}
    for root in args.audit:
        inputs_path=root/'INPUTS.json';summary_path=root/'SUMMARY.json'
        inputs=json.loads(inputs_path.read_text());summary=json.loads(summary_path.read_text())
        audit_pins[str(root)]={'inputs':digest(inputs_path.read_bytes()),'summary':digest(summary_path.read_bytes())}
        expected=set(inputs['target_ids'])
        source_release=Path(inputs['release'])
        if any(digest((source_release/name).read_bytes())!=sha for name,sha in inputs['release_pins'].items()):
            raise ValueError('Audited diagnostic input changed')
        if (len(expected)!=len(inputs['target_ids']) or summary['planned']!=len(expected)
                or summary.get('reviewed',0)+summary.get('failed',0)!=len(expected)):
            raise ValueError('Audit target membership invalid')
        result_paths=list((root/'results').glob('*.json'))
        if len(result_paths)!=len(expected):raise ValueError('Incomplete audit result files')
        observed=set()
        for path in result_paths:
            result=json.loads(path.read_text());identity=result['id']
            if identity not in expected or identity in observed:raise ValueError('Unexpected/duplicate audit target')
            observed.add(identity)
            if identity not in progress:continue  # Earlier diagnostic row did not survive final export.
            if identity in reviewed:raise ValueError('Progress target reviewed twice')
            if result['row_sha256']!=row_digest(progress[identity]):
                raise ValueError('Reviewed row differs from final release: '+identity)
            if result['status']=='reviewed':
                verdict=result['verdict']
                if (set(verdict)!={'accept','new_relevant_fact','corrects_prior_error','answers_new_requirement','reason'}
                        or type(result['accept']) is not bool or result['accept'] is not verdict['accept']
                        or not isinstance(verdict['reason'],str) or not verdict['reason'].strip()):
                    raise ValueError('Review decision mismatch')
            elif result['status']=='failed':
                if result['accept'] is not False:raise ValueError('Failed review was admitted')
            else:raise ValueError('Unknown audit status')
            reviewed[identity]=result
        if observed!=expected:raise ValueError('Audit target missing')
    if set(reviewed)!=set(progress):
        missing=set(progress)-set(reviewed)
        raise ValueError(f'Progress review coverage incomplete: {len(missing)} missing')
    rejected={identity:'progress_delta_'+('review_failed' if row['status']=='failed' else 'no_real_advance')
              for identity,row in reviewed.items() if not row['accept']}
    base=json.loads(args.base_exclusions.read_text())
    if set(base)&set(rejected):raise ValueError('Delta review overlaps prior exclusions')
    merged=dict(base,**rejected)
    counts=Counter()
    for identity,row in reviewed.items():
        split=progress[identity]['split']
        counts[split+'/'+('accepted' if row['accept'] else row['status']+'_rejected')]+=1
    report={'reviewed_progress':len(reviewed),'delta_rejected':len(rejected),
            'combined_exclusions':len(merged),'counts':dict(counts),
            'release_files_sha256':pins,'audit_pins':audit_pins,
            'base_exclusions_sha256':digest(args.base_exclusions.read_bytes()),
            'same_teacher_not_independent':True,
            'limits':'Review admission is not an independent semantic accuracy estimate.'}
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'DELTA-EXCLUSIONS.json').write_text(json.dumps(rejected,ensure_ascii=False,indent=2)+'\n')
    (args.out/'EXCLUSIONS.json').write_text(json.dumps(merged,ensure_ascii=False,indent=2)+'\n')
    (args.out/'SUMMARY.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--audit',type=Path,nargs='+',required=True)
    p.add_argument('--base-exclusions',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
