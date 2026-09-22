"""Role-separated initial-State training on the frozen native fp32io16 equations."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time

HERE=Path(__file__).resolve().parent
GPU='GPU-a9570da2-547a-c2b3-0cab-7bbdc1a8a8b0'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()

def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2)

def check_lr(value):
    if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or not 1e-6<=value<=1e-4:
        raise ValueError('Learning rate must be finite and within 1e-6..1e-4')

def load_rows(path,max_tokens):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Empty/duplicate training examples')
    for r in rows:
        ids,labels,n=r['input_ids'],r['labels'],r['prompt_tokens']
        if (r['split']!='train' or r['state_role'] not in {'resolver','writer'} or not 0<n<len(ids)<=max_tokens
                or ids[-1]!=0 or labels != [-100]*n+ids[n:] or any(type(x) is not int or not 0<=x<65536 for x in ids)):
            raise ValueError('Invalid role, train split, target mask, EOS or token budget')
    return rows

def load_model(config):
    if (socket.gethostname()!='rwkv-82' or Path('/etc/machine-id').read_text().strip()!='bcd164d5ad3a4ab3b0790412e32e69f3'
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU):
        raise ValueError('Only authorized host physical GPU3 may be used')
    sys.path.insert(0,str(HERE/'runtime'))
    sys.path.insert(0,config['engine_root'])
    os.environ['VLLM_RWKV7_WKV_MODE']='fp32io16'
    for path,digest in config['external_pins'].items():
        if sha(path)!=digest:raise ValueError('External runtime identity changed: '+path)
    for path,digest in config['source_pins'].items():
        if sha(HERE/path)!=digest:raise ValueError('Training source changed: '+path)
    import torch
    if torch.__version__!='2.11.0+cu128' or torch.cuda.device_count()!=1:
        raise ValueError('Torch ABI or visible device count changed')
    if str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')!=GPU[4:]:
        raise ValueError('Physical GPU UUID mismatch')
    torch.cuda.set_per_process_memory_fraction(config['max_reserved_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_fp16_accumulation=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    from rwkv_lh.statetune_native_recurrence import load_extension
    load_extension(config['extension']['path'],config['extension']['sha256'])
    from rwkv_lh.statetune_native_model import build
    from rwkv_lh import statetune_core as core
    model=build({'model_artifact':str(HERE/'model_config'),'context_tokens':config['max_tokens'],'base':config['base']})
    if model.native.wkv_state_dtype!=torch.float32:raise ValueError('Native recurrent State must be FP32')
    params=core.state_parameters(model,layers=model.layout.layers)
    if any(p.dtype!=torch.float32 or not p.requires_grad for p in params.values()):raise ValueError('Trainable State dtype mismatch')
    base={n:p for n,p in model.named_parameters() if n not in params}
    if any(p.requires_grad for p in base.values()):raise ValueError('Base must be frozen')
    return torch,model,params,base

def numerical_preflight(torch,model,params):
    from rwkv_lh.statetune_core import max_logit_difference
    results=[]
    initial={n:p.detach().clone() for n,p in params.items()}
    for nonzero in (False,True):
        with torch.no_grad():
            for index,p in enumerate(params.values()):
                if nonzero:
                    p.copy_((torch.arange(p.numel(),device=p.device,dtype=torch.float32).reshape(p.shape)%17-8)*0.0001)
                else:p.zero_()
        for length in (16,64):
            tokens=((torch.arange(length,device='cuda',dtype=torch.long)*37+101)%65536).unsqueeze(0)
            with torch.no_grad():
                state=model.native.zero_state(1)
                state[1].copy_(torch.stack([p.transpose(-1,-2) for p in params.values()]).unsqueeze(1))
                reference=model.native.forward_all_logits(tokens,state)
                actual=model(tokens)
                difference=max_logit_difference(actual,reference)
                match=bool(torch.equal(actual.argmax(-1),reference.argmax(-1)))
                results.append({'nonzero':nonzero,'tokens':length,'max_logit_difference':difference,'all_argmax_match':match})
                if difference>0.001 or not match:raise ValueError('Training/serving prefix alignment failed: '+str(results[-1]))
                del reference,actual,state
    with torch.no_grad():
        for n,p in params.items():p.copy_(initial[n])
    tokens=((torch.arange(64,device='cuda',dtype=torch.long)*37+101)%65536).unsqueeze(0)
    model.zero_grad(set_to_none=True)
    loss=torch.nn.functional.cross_entropy(model(tokens[:,:-1])[0,-8:].float(),tokens[0,-8:])
    loss.backward()
    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params.values()):
        raise ValueError('Missing/nonfinite recurrent State gradients')
    if not all(torch.equal(p,initial[n]) for n,p in params.items()):raise ValueError('Preflight mutated State')
    model.zero_grad(set_to_none=True)
    results.append({'backward_loss':float(loss.detach()),'optimizer_updates':0})
    return results

def main(args):
    config=json.loads(args.config.read_text())
    check_lr(config['learning_rate'])
    args.out.mkdir(parents=True,exist_ok=False)
    began=time.monotonic();updates=0
    write(args.out/'CONFIG.json',config)
    try:
        torch,model,params,base=load_model(config)
        versions={n:p._version for n,p in base.items()}
        write(args.out/'NUMERICAL-PREFLIGHT.json',numerical_preflight(torch,model,params))
        print(json.dumps({'phase':'numerical_preflight_passed','seconds':time.monotonic()-began}),flush=True)
        if args.mode=='preflight':
            write(args.out/'COMPLETED.json',{'mode':'preflight','optimizer_updates':0,'elapsed_s':time.monotonic()-began})
            return
        if not args.data or not args.data_sha256 or sha(args.data)!=args.data_sha256:raise ValueError('Training data pin mismatch')
        rows=load_rows(args.data,config['max_tokens'])
        if len(rows)<config['minimum_train_examples']:raise ValueError('Training set too small')
        if {r['state_role'] for r in rows}!={'resolver','writer'}:raise ValueError('Both role datasets required')
        write(args.out/'DATA.json',{'sha256':args.data_sha256,'counts':dict(Counter(r['state_role'] for r in rows))})
        # Full longest sequence backward must succeed before the first optimizer step.
        longest=max(rows,key=lambda r:len(r['input_ids']))
        ids=torch.tensor([longest['input_ids'][:-1]],device='cuda',dtype=torch.long)
        labels=torch.tensor(longest['labels'][1:],device='cuda',dtype=torch.long)
        logits=model(ids)[0];mask=labels!=-100
        loss=torch.nn.functional.cross_entropy(logits[mask].float(),labels[mask]);loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values()):
            raise ValueError('Longest complete sample backward failed')
        write(args.out/'LONGEST-PREFLIGHT.json',{'id':longest['id'],'tokens':len(longest['input_ids']),
            'loss':float(loss.detach()),'optimizer_updates':0,'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        del ids,labels,logits,mask,loss
        model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        checkpoints=[]
        for role in ('resolver','writer'):
            with torch.no_grad():
                for p in params.values():p.zero_()
            optimizer=torch.optim.AdamW(list(params.values()),lr=config['learning_rate'],weight_decay=0.,eps=1e-8)
            selected=[r for r in rows if r['state_role']==role]
            role_dir=args.out/role;role_dir.mkdir()
            with (role_dir/'updates.jsonl').open('x') as log:
                for epoch in range(config['epochs']):
                    order=list(selected);random.Random(config['seed']+epoch).shuffle(order)
                    for offset in range(0,len(order),config['accumulation']):
                        if time.monotonic()-began>config['max_seconds']:raise TimeoutError('Registered training wall limit reached')
                        group=order[offset:offset+config['accumulation']]
                        optimizer.zero_grad(set_to_none=True);losses=[]
                        for row in group:
                            ids=torch.tensor([row['input_ids'][:-1]],device='cuda',dtype=torch.long)
                            labels=torch.tensor(row['labels'][1:],device='cuda',dtype=torch.long)
                            logits=model(ids)[0];mask=labels!=-100
                            loss=torch.nn.functional.cross_entropy(logits[mask].float(),labels[mask])
                            if not torch.isfinite(loss):raise ValueError('Nonfinite target loss')
                            (loss/len(group)).backward();losses.append(float(loss.detach()))
                            del ids,labels,logits,mask,loss
                        if any(p.grad is not None or p.requires_grad for p in base.values()):raise ValueError('Base gradient appeared')
                        norm=float(torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True))
                        if norm==0:raise ValueError('Zero State gradient')
                        check_lr(optimizer.param_groups[0]['lr']);optimizer.step();updates+=1
                        if not all(torch.isfinite(p).all() for p in params.values()):raise ValueError('Nonfinite State update')
                        if torch.cuda.max_memory_reserved()>config['max_reserved_bytes']:raise MemoryError('Training memory ceiling exceeded')
                        record={'role':role,'epoch':epoch+1,'update':updates,'ids':[r['id'] for r in group],
                            'losses':losses,'lr':optimizer.param_groups[0]['lr'],'gradient_norm':norm,'elapsed_s':time.monotonic()-began}
                        log.write(json.dumps(record)+'\n');log.flush()
                        if updates%10==0:print(json.dumps(record),flush=True)
                    state={n:p.detach().cpu().float().contiguous().clone() for n,p in params.items()}
                    checkpoint=role_dir/f'epoch-{epoch+1}.pth';torch.save(state,checkpoint)
                    checkpoints.append({'role':role,'epoch':epoch+1,'path':str(checkpoint),'sha256':sha(checkpoint),'dtype':'float32','layout':'H,V,K'})
            del optimizer
        if any(p._version!=versions[n] for n,p in base.items()):raise ValueError('Frozen base tensor modified')
        write(args.out/'COMPLETED.json',{'optimizer_updates':updates,'checkpoints':checkpoints,'examples':len(rows),
            'epochs':config['epochs'],'elapsed_s':time.monotonic()-began,'base_unchanged':True,
            'quality_verified':False,'production_promoted':False,'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
    except BaseException as exc:
        write(args.out/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc),'optimizer_updates':updates,'elapsed_s':time.monotonic()-began})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['preflight','train'],required=True);p.add_argument('--data',type=Path);p.add_argument('--data-sha256')
    main(p.parse_args())
