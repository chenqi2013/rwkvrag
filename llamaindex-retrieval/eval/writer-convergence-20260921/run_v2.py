"""Execute frozen Writer-only ablation; reuse audited model launch/verification."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'data/experiments/writer-convergence-20260921/run2'


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    for name, expected in json.loads((HERE/'PINS-v2.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    loader = importlib.util.spec_from_file_location('paired_launcher', HERE.parent/'model-size-paired-20260921/run_v2.py')
    launch = importlib.util.module_from_spec(loader); loader.loader.exec_module(launch)
    launch.OUT = OUT/'preflight'; launch.OUT.mkdir()
    started = time.monotonic(); proc = None; records = []; error = None
    try:
        launch.verify()  # Frozen engine/model/input checks; original records untouched.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(launch.MODELS['7.2b']))
        cases = json.loads((HERE/'INPUTS.json').read_text())
        tokens = {(c['ordinal'], arm): tokenizer.encode(c[arm], add_special_tokens=False)
                  for c in cases for arm in ['baseline','decision']}
        assert all(len(t)+2048 <= 16384 for t in tokens.values())
        save(OUT/'BINDING.json', {'pins':json.loads((HERE/'PINS-v2.json').read_text()),
             'engine':str(launch.ENGINE), 'parameters':launch.PARAMS, 'model':str(launch.MODELS['7.2b']),
             'state':'zero, no imports', 'started_at':datetime.now(timezone.utc).isoformat(),
             'calls_planned':752,'training_updates':0})
        with httpx.Client(base_url='http://127.0.0.1:18426', timeout=180) as client:
            for part,(arm,rnd) in enumerate([('baseline',1),('decision',1),('decision',2),('baseline',2)],1):
                proc=launch.start('7.2b',part); deadline=time.monotonic()+600
                while True:
                    if proc.poll() is not None: raise RuntimeError('engine exited')
                    try:
                        if client.get('/health',timeout=2).status_code==200:break
                    except httpx.HTTPError:pass
                    if time.monotonic()>deadline:raise TimeoutError('startup')
                    time.sleep(2)
                directory=OUT/f'{arm}-round{rnd}';directory.mkdir()
                for c in cases:
                    if time.monotonic()-started>14400:raise TimeoutError('4h budget')
                    request={'model':'paired-eval','prompt':c[arm],**launch.PARAMS}
                    wire=json.dumps(request,ensure_ascii=False,separators=(',',':')).encode()
                    row={'ordinal':c['ordinal'],'id':c['id'],'arm':arm,'round':rnd,'request':request,
                         'request_sha256':launch.sha(wire),'started_at':datetime.now(timezone.utc).isoformat()}
                    save(directory/f"{c['ordinal']:04d}.request.json",row)
                    tick=time.monotonic()
                    try:
                        response=client.post('/v1/completions',content=wire,headers={'content-type':'application/json'})
                        row.update(http_status=response.status_code,response_body_base64=base64.b64encode(response.content).decode(),response_sha256=launch.sha(response.content))
                        response.raise_for_status();body=response.json();choice=body['choices'][0]
                        assert choice['prompt_token_ids']==tokens[c['ordinal'],arm]
                        row.update(status='recorded',raw_text=choice['text'],raw_text_sha256=launch.sha(choice['text'].encode()),
                                   output_token_ids=choice['token_ids'],input_token_ids=choice['prompt_token_ids'],
                                   finish_reason=choice['finish_reason'],usage=body['usage'],repetition=launch.repeat_flags(choice['text']))
                    except Exception as exc:row.update(status='failed',error=repr(exc))
                    row['elapsed_ms']=(time.monotonic()-tick)*1000
                    save(directory/f"{c['ordinal']:04d}.json",row)
                    records.append({k:row.get(k) for k in ['arm','round','ordinal','status','finish_reason','elapsed_ms','raw_text_sha256']})
                    if row['status']=='failed':raise RuntimeError(row['error'])
                    if len(records)%20==0:print(json.dumps({'recorded':len(records),'arm':arm,'round':rnd}),flush=True)
                launch.stop(proc);proc=None;time.sleep(5)
    except BaseException as exc:error=repr(exc);raise
    finally:
        launch.stop(proc)
        save(OUT/'SUMMARY.json',{'recorded':len(records),'planned':752,'error':error,'elapsed_s':time.monotonic()-started,'records':records})


if __name__=='__main__':main()
