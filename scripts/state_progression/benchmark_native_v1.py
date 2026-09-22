"""Zero-update throughput and longest-sequence memory check on the pinned native trainer."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

spec=importlib.util.spec_from_file_location('train_v2',Path(__file__).with_name('train_v2.py'))
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)


def main(args):
    if t.sha(args.data)!=args.data_sha256:raise ValueError('Benchmark sample file changed')
    rows=[json.loads(line) for line in args.data.read_text().splitlines()]
    if len(rows)!=3 or len({r['id'] for r in rows})!=3:raise ValueError('Expected three distinct pinned samples')
    config=json.loads(args.config.read_text())
    args.out.mkdir(parents=True,exist_ok=False)
    t.write(args.out/'RUN.json',{'config_sha256':t.sha(args.config),'data_sha256':args.data_sha256,
        'script_sha256':t.sha(Path(__file__)),'trainer_sha256':t.sha(Path(__file__).with_name('train_v2.py')),
        'optimizer_updates':0})
    began=time.monotonic();torch,model,params,base=t.load_model(config)
    versions={name:p._version for name,p in base.items()}
    originals={name:p.detach().clone() for name,p in params.items()}
    timings=[]
    for row in rows:
        t.load_rows(args.data,config['max_tokens'])
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
        ids=torch.tensor([row['input_ids'][:-1]],device='cuda',dtype=torch.long)
        labels=torch.tensor(row['labels'][1:],device='cuda',dtype=torch.long)
        logits=model(ids)[0];mask=labels!=-100
        loss=torch.nn.functional.cross_entropy(logits[mask].float(),labels[mask]);loss.backward()
        torch.cuda.synchronize();elapsed=time.monotonic()-start
        if not bool(torch.isfinite(loss)) or not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params.values()):
            raise ValueError('Nonfinite zero-update benchmark')
        if any(p.grad is not None for p in base.values()):raise ValueError('Frozen base received gradient')
        timings.append({'id':row['id'],'role':row['state_role'],'tokens':len(row['input_ids']),
                        'seconds_forward_backward':elapsed,'loss':float(loss.detach()),
                        'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        print(json.dumps(timings[-1]),flush=True)
        del ids,labels,logits,mask,loss
        model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    if any(p._version!=versions[n] for n,p in base.items()) or any(not torch.equal(p,originals[n]) for n,p in params.items()):
        raise ValueError('Benchmark changed checkpoint or State')
    t.write(args.out/'COMPLETED.json',{'samples':timings,'optimizer_updates':0,
        'elapsed_s':time.monotonic()-began,'quality_verified':False,'production_promoted':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--data-sha256',required=True)
    p.add_argument('--out',type=Path,required=True);main(p.parse_args())
