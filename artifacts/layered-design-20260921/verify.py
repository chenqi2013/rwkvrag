"""Read-only checks for design input bindings; no model calls or semantic scoring."""
import argparse,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
base=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((base/'MANIFEST.json').read_text());focus=json.loads((base/'FOCUSED-REPLAY.json').read_text());pins=json.loads((base/'INTEGRITY.json').read_text())
assert all(sha(base/n)==v for n,v in pins.items())
assert len(manifest['cases'])==160 and {r['index'] for r in manifest['cases']}==set(range(160))
assert len(focus['cases'])==20 and len({r['index'] for r in focus['cases']})==20
members={r['index']:r for r in manifest['cases']};traces=0
for row in manifest['cases']:
 cases=[]
 for arm,binding in row['arms'].items():
  path=a.source_root/binding['path'];assert sha(path)==binding['sha256'];raw=json.loads(path.read_text());assert raw['index']==row['index'] and raw['case_id']==row['case_id'] and raw['status']==binding['status'];cases.append(raw['case'])
 assert cases[0]==cases[1]
for row in focus['cases']:
 assert members[row['index']]['case_id']==row['case_id']
 for arm,binding in row['arms'].items():
  path=a.source_root/binding['path'];assert sha(path)==binding['sha256']==members[row['index']]['arms'][arm]['sha256'];raw=json.loads(path.read_text());result=raw.get('result') or {}
  assert row['case']==raw['case'] and binding['original_answer']==raw.get('answer')
  assert all(result.get(k)==v for k,v in binding['upstream_snapshot'].items())
  assert {t['path'] for t in binding['traces']}=={str(t.relative_to(a.source_root)) for t in (path.parent/'traces').glob('*.json')}
  for trace in binding['traces']:
   tracepath=a.source_root/trace['path'];assert sha(tracepath)==trace['sha256'];original=json.loads(tracepath.read_text())
   assert all(original.get(k)==trace[k] for k in ('stage','prompt_sha256','raw_text_sha256'));traces+=1
listed={int(i):cid for i,cid in re.findall(r'^\| (\d+) \| ([^ |]+) \|',a.plan.read_text(),re.M)}
assert listed=={r['index']:r['case_id'] for r in focus['cases']},(listed,{r['index']:r['case_id'] for r in focus['cases']})
print(json.dumps(dict(status='BINDINGS_VERIFIED_NOT_SEMANTIC_TEST',full_pairs=160,raw_results=320,focus_pairs=20,focused_trace_files=traces,model_calls=0,training_updates=0,source_root=str(a.source_root),verifier_sha256=sha(Path(__file__)),input_integrity_sha256=sha(base/'INTEGRITY.json')),indent=2))
