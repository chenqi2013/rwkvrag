"""Sequential paired model experiment; raw requests and responses immutable."""
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import httpx

HERE=Path(__file__).resolve().parent
ENGINE=Path('/home/chase/rwkvrag/data/services/vllm-decode-20260920/engine')
BASE=Path('/home/chase/rwkvrag/data/services/model-size-paired-20260921')
OUT=Path('/home/chase/rwkvrag/data/experiments/model-size-paired-20260921/run2')
GPU='GPU-a9570da2-547a-c2b3-0cab-7bbdc1a8a8b0'
MODELS={'2.9b':BASE/'model-2.9b','7.2b':Path('/mnt/nas-model/g1j/hf/rwkv7-g1j-7.2b-20260831-ctx16384')}
PARAMS={'max_tokens':2048,'temperature':1.0,'top_p':1.0,'top_k':1,
        'presence_penalty':0.0,'frequency_penalty':0.0,'stream':False,
        'stop':['✿','\nUser:','\n### User'],'stop_token_ids':[0],
        'ignore_eos':False,'add_special_tokens':False,'seed':11,
        'return_token_ids':True,'skip_special_tokens':False,'penalty_decay':0.996}

def sha(b):return hashlib.sha256(b).hexdigest()
def file_sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2)

def repeat_flags(text):
    s=re.sub(r'\s+','',text)
    gram=next((s[i:i+40] for i in range(max(0,len(s)-39)) if s.count(s[i:i+40])>=3),None)
    parts=[re.sub(r'\s+','',p) for p in re.split(r'[。！？\n]',text)]
    sentence=next((p for p in parts if len(p)>=12 and parts.count(p)>=3),None)
    return {'repeated_40chars':gram,'repeated_sentence':sentence,
            'automatic_candidate_only':bool(gram or sentence)}

def verify():
    assert socket.gethostname()=='rwkv-82'
    assert Path('/etc/machine-id').read_text().strip()=='bcd164d5ad3a4ab3b0790412e32e69f3'
    pins=json.loads((HERE/'PINS-v2.json').read_text())
    assert all(file_sha(HERE/p)==h for p,h in pins.items())
    manifest=ENGINE.parent/'ENGINE-SOURCE.json'
    assert file_sha(manifest)=='b8be05cd8dddcdb9eac88bca6af0ff2486982edd9407328fb6b064fd0c416168'
    binding=json.loads(manifest.read_text())
    for p,h in binding['files'].items():assert file_sha(ENGINE/p)==h,p
    conv=json.loads((MODELS['2.9b']/'CONVERSION.json').read_text())
    assert conv['mapping_script_sha256']==file_sha(HERE/'convert_29.py')
    for p,h in conv['files'].items():assert file_sha(MODELS['2.9b']/p)==h,p
    for p,h in binding['model_metadata'].items():assert file_sha(MODELS['7.2b']/p)==h,p
    shards={}
    for h,p in re.findall(r'([a-f0-9]{64})  (model-\S+\.safetensors)',(MODELS['7.2b']/'PROVENANCE.md').read_text()):
        assert file_sha(MODELS['7.2b']/p)==h,p
        shards[p]=h
    assert len(shards)==6
    assert file_sha(MODELS['2.9b']/'tokenizer.json')==file_sha(MODELS['7.2b']/'tokenizer.json')
    from transformers import AutoTokenizer
    tokenizers={k:AutoTokenizer.from_pretrained(str(p)) for k,p in MODELS.items()}
    rows=json.loads((HERE/'INPUTS.json').read_text());assert len(rows)==188
    for r in rows:
        a=tokenizers['2.9b'].encode(r['prompt'],add_special_tokens=False)
        b=tokenizers['7.2b'].encode(r['prompt'],add_special_tokens=False)
        assert a==b and len(a)+2048<=16384,r['id']
        r['prompt_token_ids']=a
    save(OUT/'BINDING.json',{'pins':pins,'conversion':conv,'model72_shards':shards,
        'engine_manifest_sha256':file_sha(manifest),'parameters':PARAMS,'calls_planned':752,
        'max_input_tokens':max(len(r['prompt_token_ids']) for r in rows),
        'started_at':datetime.now(timezone.utc).isoformat(),'training_updates':0})
    save(OUT/'TOKENIZED-INPUTS.json',rows)
    return rows

def start(arm,part):
    gpu=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip()
    index,uuid,used,total=[x.strip() for x in gpu.split(',')]
    assert index=='3' and uuid==GPU and int(total)-int(used)>32000
    with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',18426))!=0
    cmd=[str(ENGINE/'.venv/bin/python'),'-m','vllm.entrypoints.openai.api_server',
         '--model',str(MODELS[arm]),'--served-model-name','paired-eval',
         '--host','127.0.0.1','--port','18426','--dtype','float16',
         '--mamba-ssm-cache-dtype','float32','--max-model-len','16384',
         '--max-num-seqs','4','--max-num-batched-tokens','2048',
         '--gpu-memory-utilization','0.30','--enable-chunked-prefill',
         '--enable-prefix-caching','--async-scheduling','--generation-config','vllm']
    env={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'VLLM_RWKV_STATE_CACHE_MAX_BYTES':'268435456',
         'PYTHONPATH':str(ENGINE)}
    save(OUT/f'SERVICE-{part}.json',{'arm':arm,'command':cmd,'gpu':gpu,'state':'zero, no imports'})
    log=(OUT/f'service-{part}.log').open('xb')
    proc=subprocess.Popen(cmd,cwd=ENGINE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close()
    return proc

def stop(proc):
    if proc is None:return
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=15)

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    proc=None;records=[];started=time.monotonic();error=None
    try:
        rows=verify()
        with httpx.Client(base_url='http://127.0.0.1:18426',timeout=180) as client:
            for part,(arm,rounds) in enumerate([('2.9b',[1]),('7.2b',[1]),('7.2b',[2]),('2.9b',[2])],1):
                proc=start(arm,part)
                deadline=time.monotonic()+600
                while True:
                    if proc.poll() is not None:raise RuntimeError('engine exited during startup')
                    try:
                        r=client.get('/health',timeout=2)
                        if r.status_code==200:break
                    except httpx.HTTPError:pass
                    if time.monotonic()>deadline:raise TimeoutError('engine startup')
                    time.sleep(2)
                save(OUT/f'MODELS-{part}.json',client.get('/v1/models').json())
                for rnd in rounds:
                    directory=OUT/f'{arm}-round{rnd}';directory.mkdir()
                    for row in rows:
                        if time.monotonic()-started>14400:raise TimeoutError('experiment deadline')
                        request={'model':'paired-eval','prompt':row['prompt'],**PARAMS}
                        wire=json.dumps(request,ensure_ascii=False,separators=(',',':')).encode()
                        record={'arm':arm,'round':rnd,'ordinal':row['ordinal'],'id':row['id'],
                                'suite':row['suite'],'category':row['category'],
                                'prompt_sha256':row['prompt_sha256'],'request':request,
                                'request_sha256':sha(wire),'started_at':datetime.now(timezone.utc).isoformat()}
                        save(directory/f"{row['ordinal']:04d}.request.json",record)
                        t=time.monotonic()
                        try:
                            response=client.post('/v1/completions',content=wire,headers={'content-type':'application/json'})
                            record.update(http_status=response.status_code,response_body_base64=base64.b64encode(response.content).decode(),response_sha256=sha(response.content))
                            response.raise_for_status();body=response.json();choice=body['choices'][0]
                            if choice.get('prompt_token_ids')!=row['prompt_token_ids']:
                                raise ValueError('input token mismatch')
                            raw=choice['text']
                            record.update(status='recorded',raw_text=raw,raw_text_sha256=sha(raw.encode()),
                                          finish_reason=choice['finish_reason'],usage=body['usage'],
                                          output_token_ids=choice.get('token_ids'),input_tokens_equal=True,
                                          repetition=repeat_flags(raw))
                        except Exception as exc:
                            record.update(status='failed',error=repr(exc))
                        record['elapsed_ms']=(time.monotonic()-t)*1000
                        save(directory/f"{row['ordinal']:04d}.json",record)
                        records.append({k:record.get(k) for k in ['arm','round','ordinal','id','status','finish_reason','raw_text_sha256','elapsed_ms','repetition']})
                        if record['status']=='failed':raise RuntimeError(record['error'])
                        if len(records)%20==0:print(json.dumps({'recorded':len(records),'planned':752,'arm':arm,'round':rnd,'elapsed_s':time.monotonic()-started}),flush=True)
                stop(proc);proc=None
                time.sleep(5)
    except BaseException as exc:
        error=repr(exc);raise
    finally:
        stop(proc)
        save(OUT/'SUMMARY.json',{'recorded':len(records),'planned':752,'error':error,
             'status':'EXECUTION_COMPLETE_NOT_QUALITY_GATE' if len(records)==752 and not error else 'FAILED',
             'elapsed_s':time.monotonic()-started,'records':records})
        print(json.dumps({'finished':len(records),'error':error}),flush=True)

if __name__=='__main__':main()
