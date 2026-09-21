import json,hashlib,time,base64
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent

def main():
    for name,h in json.loads((HERE/'PINS.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    probe=json.loads((HERE/'PROBE.json').read_text());out=ROOT/'data/quality-runs/funnel-structured-output-20260921/probe1';out.mkdir(parents=True,exist_ok=False)
    raw=json.dumps(probe['request'],ensure_ascii=False).encode();tick=time.monotonic()
    response=httpx.post('http://127.0.0.1:18426/v1/completions',content=raw,headers={'Content-Type':'application/json'},timeout=180)
    record={'request':probe['request'],'http_status':response.status_code,'elapsed_s':time.monotonic()-tick,'response_body_base64':base64.b64encode(response.content).decode(),'response_sha256':hashlib.sha256(response.content).hexdigest(),'response':response.json()}
    with (out/'RESULT.json').open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2)
    print(response.status_code,flush=True)
    if response.status_code==200:
        choice=response.json()['choices'][0];assert choice['prompt_token_ids']==probe['baseline_input_token_ids'];print(choice['text'],flush=True)
if __name__=='__main__':main()
