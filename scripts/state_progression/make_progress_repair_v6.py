"""Freeze distinct synthetic themes for evidence-resolving follow-up labels."""
import argparse
import hashlib
import json
from pathlib import Path


THEMES={
    'software_ops':[
        'A fictitious API gateway changes version compatibility after a signed migration note; compare client behavior by version.',
        'A task queue initially lacks retry outcomes; later worker traces distinguish attempted, skipped, and completed jobs.',
        'A database migration has old and new schemas, a rollback condition, and a later verification report.',
        'An access-control policy gains a narrow role exception after a security review; separate allowed actions from claims of deployment.',
        'A cache invalidation incident receives new timing traces and a configuration revision; identify what remains unmeasured.'
    ],
    'manufacturing':[
        'A fictional battery pack has thermal tests at two loads and a later cold-start result that changes suitability.',
        'A robot arm calibration report is followed by a second-axis measurement and a corrected tolerance sheet.',
        'A pump endurance comparison begins with flow rates and later adds seal-failure observations and service costs.',
        'A sensor batch quality notice gains serial-range inspection results; distinguish affected and unaffected lots.',
        'A packaging line defect report gains a repair verification run and a remaining production hold condition.'
    ],
    'transport':[
        'A bus route comparison needs transfer time, accessibility, and a later revised weekend timetable.',
        'A rail connection question begins with planned times and later receives actual service and platform-change records.',
        'A ferry capacity choice initially lacks weather restrictions; a later operating notice supplies route-specific limits.',
        'An airport baggage process has conflicting target times until a dated operational correction is released.',
        'A bike-share expansion proposal receives usage counts and a later maintenance-capacity audit.'
    ],
    'environment':[
        'A watershed report has two monitoring stations and a later upstream measurement that resolves a missing comparison.',
        'An air-quality sensor network gains a calibration bulletin that narrows which readings remain comparable.',
        'Two waste-processing options are compared on throughput, residues, and a later permit condition.',
        'A shoreline restoration project moves from planned planting to measured survival with site-specific uncertainty.',
        'A wildlife monitoring survey gains a second-season count and a correction for missed observation days.'
    ],
    'education':[
        'A course prerequisite rule is revised for one cohort after an official faculty decision.',
        'An examination timetable gains room-capacity evidence and a later conflict-resolution notice.',
        'A laboratory safety procedure has a missing ventilation test until an inspection report arrives.',
        'Two campus shuttle plans receive a later accessibility survey and cost adjustment.',
        'A library-hour proposal gains observed evening visits and a staffing limit before a final choice.'
    ],
    'public_admin':[
        'A permit application passes one review but still lacks final authorization until a dated decision record.',
        'A municipal budget amendment changes only selected line items; compare before and after without rewriting unchanged totals.',
        'An accessibility audit receives a later repair certificate for some, but not all, buildings.',
        'A zoning consultation has conflicting stakeholder positions and a later vote with a precise scope.',
        'A record-digitization project gains a checksum report and an unresolved missing-volume list.'
    ],
    'research':[
        'Two telescope observations differ in exposure and filter; a later calibration makes only one comparison valid.',
        'An enzyme assay initially lacks control measurements and later receives them with units and uncertainty.',
        'A geological core comparison gains depth-adjusted dating and a corrected sample attribution.',
        'Three solar-panel prototypes gain a second temperature test that changes a conditional recommendation.',
        'A linguistic survey receives a later sample breakdown that limits a previously broad conclusion.'
    ],
    'procurement':[
        'Three fictitious bids meet different mandatory criteria; later addendum changes one criterion but not the deadline.',
        'A warranty comparison begins with duration and later adds exclusions for specified components.',
        'Two maintenance contracts gain verified annual labor costs and a remaining spare-parts uncertainty.',
        'A phased rollout plan gains completed milestone evidence while later phases remain merely scheduled.',
        'An inventory substitution proposal receives compatibility testing and a supplier recall for one batch.'
    ],
    'arts_media':[
        'A film festival schedule changes one screening after a venue closure; distinguish announced from completed events.',
        'A museum exhibition catalog gains provenance documentation for some objects, not the entire collection.',
        'A podcast rights agreement gains a geographic exception and a later expiration notice.',
        'An archive catalog receives a newly found edition and a corrected publication year.',
        'Two translations are compared by source edition, release date, and a later errata notice.'
    ],
    'sports':[
        'A tournament tie-break rule is applied only after final match statistics arrive.',
        'An equipment rule changes for a specified division and date; compare old and new eligibility.',
        'A venue accessibility plan gains completed inspection results but one entrance remains pending.',
        'An athlete eligibility question begins with a provisional roster and later official clearance.',
        'A ranking table is updated after an appeal; preserve unaffected teams and the exact correction.'
    ],
    'agriculture_food':[
        'Two irrigation trials gain a second soil-moisture reading and a water-use limit.',
        'A crop cultivar comparison gains a disease-resistance result under a distinct growing condition.',
        'A cold-chain shipment receives logger records for two legs and a later exception report.',
        'A nutrition label is corrected for one serving size while ingredient statements remain unchanged.',
        'A livestock sensor pilot receives missing-animal counts and a later battery-life verification.'
    ],
    'emergency_ops':[
        'A flood-warning plan gains upstream gauge readings and a later verified evacuation trigger.',
        'A quake building inspection distinguishes preliminary visual checks from signed structural clearance.',
        'A shelter allocation report gains actual occupancy and a separate accessibility requirement.',
        'An emergency-radio outage receives repeater logs and a repair test with a remaining coverage gap.',
        'A repair sequence after a storm gains completed work orders for some locations and planned work for others.'
    ],
}


def main(args):
    jobs=[]
    for domain,briefs in THEMES.items():
        if len(briefs)!=5:raise ValueError('Every domain must contain five distinct briefs')
        for index,brief in enumerate(briefs,1):
            identity='repair-v6-'+hashlib.sha256((domain+'\n'+brief).encode()).hexdigest()[:20]
            jobs.append({'id':identity,'split':'train','material_mode':'synthetic',
                'source_families':[f'progress-repair-v6/{domain}/{index:02d}'],
                'sources':[{'id':'THEME','title':f'{domain} scenario {index}',
                    'text':'Create new fictional organizations and objects. Theme: '+brief}]})
    if len(jobs)!=60 or len({x['id'] for x in jobs})!=60:
        raise ValueError('Expected 60 distinct theme jobs')
    args.out.mkdir(parents=True,exist_ok=False)
    path=args.out/'JOBS.jsonl'
    path.write_text(''.join(json.dumps(job,ensure_ascii=False)+'\n' for job in jobs))
    summary={'jobs':60,'source_families':60,'split':'train',
        'manually_authored_theme_domains':list(THEMES),'historical_question_inputs':0,
        'historical_answer_inputs':0,'material_mode':'new_fictional_documents',
        'jobs_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.out/'SOURCE-SELECTION.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
