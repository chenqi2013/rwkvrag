import asyncio,base64,hashlib,json,time
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
async def main():
    for name,h in json.loads((HERE/'PINS.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    rows=json.loads((HERE/'INPUTS.json').read_text());out=ROOT/'data/quality-runs/funnel-empty-example-probe-20260921/run1';out.mkdir(parents=True,exist_ok=False)
    async with httpx.AsyncClient(timeout=180) as client:
        for n,row in enumerate(rows):
            payload={**row['parameters'],'prompt':row['candidate_prompt']};tick=time.monotonic()
            response=await client.post('http://127.0.0.1:18426/v1/completions',json=payload)
            data=response.json();record={'id':row['id'],'request':payload,'http_status':response.status_code,'response':data,'response_body_base64':base64.b64encode(response.content).decode(),'response_sha256':hashlib.sha256(response.content).hexdigest(),'elapsed_s':time.monotonic()-tick,'baseline_raw_text':row['baseline_raw_text']}
            with (out/f'{n:03d}.json').open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2)
            print(n,response.status_code,data.get('choices',[{}])[0].get('text'),flush=True)
if __name__=='__main__':asyncio.run(main())
