"""Raw paired greedy decoding from zero and role-specific FP32 State, two rounds."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import time

spec=importlib.util.spec_from_file_location('native_training',Path(__file__).with_name('train_v2.py'))
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)

def main(args):
    if t.sha(args.cases)!=args.cases_sha256:raise ValueError('Evaluation inputs changed')
    if t.sha(args.vocab)!=args.vocab_sha256:raise ValueError('Evaluation vocabulary changed')
    completed=json.loads((args.training/'COMPLETED.json').read_text())
    states={}
    for item in completed['checkpoints']:
        if item['epoch']==2:states[item['role']]=item
    if set(states)!={'resolver','writer'}:raise ValueError('Final role states incomplete')
    rows=[json.loads(line) for line in args.cases.read_text().splitlines()]
    config=json.loads(args.config.read_text())
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'records').mkdir()
    t.write(args.out/'RUN.json',{'inputs_sha256':args.cases_sha256,'training_receipt_sha256':t.sha(args.training/'COMPLETED.json'),
        'states':states,'vocab_sha256':args.vocab_sha256,'config_sha256':t.sha(args.config),
        'evaluator_sha256':t.sha(Path(__file__)),'decode':'greedy_argmax','rounds':2,
        'raw_answers_unchanged':True,'live_retrieval':False})
    torch,model,params,base=t.load_model(config)
    checkpoints={}
    for role,item in states.items():
        p=Path(item['path'])
        if t.sha(p)!=item['sha256']:raise ValueError('Final State checkpoint changed')
        state=torch.load(p,map_location='cpu',weights_only=True)
        if set(state)!=set(params):raise ValueError('State keys differ')
        for name,value in state.items():
            if value.dtype!=torch.float32 or value.shape!=params[name].shape or not torch.isfinite(value).all():raise ValueError('Invalid FP32 State')
        checkpoints[role]=torch.stack([state[n].transpose(-1,-2) for n in params]).unsqueeze(1).to('cuda')
    # Decode bytes with the exact vocabulary; never repair generated text or citations.
    import ast
    vocab={}
    for line in args.vocab.read_text().splitlines():
        identity,rest=line.split(' ',1);literal,size=rest.rsplit(' ',1);value=ast.literal_eval(literal)
        vocab[int(identity)]=value.encode() if isinstance(value,str) else value
    if set(vocab)!=set(range(1,65530)) or any(not isinstance(value,bytes) or not value for value in vocab.values()):
        raise ValueError('Incomplete or invalid exact vocabulary')
    totals=Counter();began=time.monotonic()
    try:
        with torch.inference_mode():
            for round_number in (1,2):
                for ordinal,row in enumerate(rows):
                    for arm in (('zero','trained') if round_number==1 else ('trained','zero')):
                        result={'id':row['id'],'suite':row['suite'],'category':row['category'],'state_role':row['state_role'],
                            'round':round_number,'arm':arm,'prompt_sha256':row.get('prompt_sha256'),'raw_text':'','generated_ids':[]}
                        if 'unsupported_reason' in row:
                            result.update(status='unsupported_input',reason=row['unsupported_reason'])
                        else:
                            start=time.monotonic();state=model.native.zero_state(1)
                            if arm=='trained':state[1].copy_(checkpoints[row['state_role']])
                            prompt=torch.tensor([row['input_ids']],device='cuda',dtype=torch.long)
                            logits=model.native.forward_last_at(prompt,state,torch.tensor([prompt.shape[1]-1],device='cuda',dtype=torch.long))
                            generated=[];termination='length'
                            for step in range(row['max_output_tokens']):
                                if not torch.isfinite(logits).all():raise ValueError('Nonfinite decode logits')
                                token=int(logits.reshape(-1,65536)[-1].argmax());generated.append(token)
                                if token==0:termination='stop';break
                                if token not in vocab:termination='unmapped_token';break
                                if step+1<row['max_output_tokens']:
                                    logits=model.native.forward_last_at(torch.tensor([[token]],device='cuda',dtype=torch.long),state,
                                        torch.tensor([0],device='cuda',dtype=torch.long))
                            raw=b''.join(vocab[x] for x in generated if x in vocab)
                            try:text=raw.decode('utf-8');utf8=True
                            except UnicodeDecodeError:text=raw.decode('utf-8',errors='replace');utf8=False
                            result.update(status=termination,raw_text=text,raw_hex=raw.hex(),
                                utf8_valid=utf8,generated_ids=generated,elapsed_s=time.monotonic()-start,
                                input_tokens=len(row['input_ids']),state_sha256=states[row['state_role']]['sha256'] if arm=='trained' else None)
                            del state,logits,prompt
                        t.write(args.out/'records'/f'{ordinal:05d}-{arm}-r{round_number}.json',result)
                        totals[arm+'/'+result['status']]+=1;totals['records']+=1
                        if totals['records']%10==0:print(json.dumps(dict(totals,elapsed_s=time.monotonic()-began)),flush=True)
    except BaseException as exc:
        t.write(args.out/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc),
            'completed_records':totals['records'],'planned_records':len(rows)*4})
        raise
    finally:
        t.write(args.out/'SUMMARY.json',dict(totals,planned_records=len(rows)*4,elapsed_s=time.monotonic()-began,
               semantic_review='pending',live_retrieval=False,production_promoted=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--training',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True);p.add_argument('--cases-sha256',required=True)
    p.add_argument('--vocab',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--vocab-sha256',required=True)
    main(p.parse_args())
