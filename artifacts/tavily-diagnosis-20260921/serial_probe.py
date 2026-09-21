import json,time,sys,re,hashlib,os
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,'/home/chase/GitHub/RWKV-SearchReader/src')
from rwkv_search_reader.config import get_settings
import requests
out=Path(__file__).parent
# Supply a private JSON array of credentials; never print or archive that file.
keys=list(dict.fromkeys(json.loads(Path(sys.argv[1]).read_text())))
rows=[]
for i,key in enumerate(keys):
 if i:time.sleep(6)
 row={'ordinal':i+1,'key_sha256_prefix':hashlib.sha256(key.encode()).hexdigest()[:12],'started_at':datetime.now(timezone.utc).isoformat()}
 tick=time.monotonic()
 try:
  with requests.Session() as session:
   r=session.post('https://api.tavily.com/search',headers={'Authorization':'Bearer '+key},json={'query':'GitHub documentation','search_depth':'basic','max_results':1,'include_answer':False,'include_raw_content':False},timeout=(5,12))
  row.update(status=r.status_code,headers={k:v for k,v in r.headers.items() if k.lower() in ['retry-after','x-request-id','request-id','cf-ray','server','date']})
  if r.status_code>=400:row['error_body']=re.sub(r'tvly-[A-Za-z0-9_-]+','[REDACTED]',r.text[:2000])
  else:row['result_count']=len(r.json().get('results',[]))
 except requests.RequestException as e:row['transport_error']=type(e).__name__
 row['elapsed_s']=round(time.monotonic()-tick,3);rows.append(row)
 (out/'serial-search.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
 print(json.dumps(row,ensure_ascii=False),flush=True)
