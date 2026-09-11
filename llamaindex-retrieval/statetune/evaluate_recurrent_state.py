"""Pinned official RWKV7 recurrent inference; independent of the training FLA kernel."""
import argparse
import importlib
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from preflight_state import bound, host_guard, sha, write, CHECKPOINT
from train_state import check_config
from state_tokens import Vocabulary
from evaluate_state import generate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--config-sha256', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    if sha(args.config) != args.config_sha256:
        raise ValueError('configuration changed')
    config = json.loads(args.config.read_text())
    for path, digest in config['bindings'].items():
        if sha(bound(path)) != digest:
            raise ValueError('binding changed: '+path)
    pilot = json.loads(bound(config['pilot']).read_text())
    check_config(pilot)
    spec = config['evaluation']
    if (spec['split'] not in ['dev', 'heldout'] or spec['max_output_tokens'] != 32
            or len(spec['states']) != 2 or config['wall_seconds'] != 1800):
        raise ValueError('recurrent evaluation limit')
    gpu = host_guard()
    remaining = (datetime.fromisoformat(config['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()
    if remaining <= 60:
        raise ValueError('work window exhausted')
    output = bound(args.output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    def expired(*unused):
        raise TimeoutError('recurrent evaluation deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(min(1800, int(remaining)-15))
    forwards = 0
    try:
        runtime = bound(config['official_runtime'])
        manifest = json.loads((runtime/'RUNTIME-SOURCE.json').read_text())
        if manifest['installed_version'] != '0.8.30' or manifest['compile_cuda_extension']:
            raise ValueError('official runtime version changed')
        for name, pin in manifest['files'].items():
            if sha(runtime/'rwkv'/name) != pin['sha256']:
                raise ValueError('official runtime source changed')
        if sha(bound(pilot['checkpoint'])) != CHECKPOINT:
            raise ValueError('base checkpoint changed')
        if sha(bound(spec['inputs'])) != spec['inputs_sha256']:
            raise ValueError('inputs changed')
        rows = [json.loads(s) for s in bound(spec['inputs']).read_text().splitlines()]
        if len(rows) != 11 or len({r['id'] for r in rows}) != 11 or any(r['split'] != spec['split'] for r in rows):
            raise ValueError('evaluation coverage changed')
        vocab = Vocabulary(runtime/'rwkv/rwkv_vocab_v20230424.txt')
        import hashlib
        prepared = []
        for row in rows:
            ids = vocab.encode(row['prompt'])
            if (len(ids) != row['input_token_count'] or len(ids)+31 > 4096
                    or not row['prompt'].endswith('Assistant: <think></think>\n')
                    or hashlib.sha256(row['prompt'].encode()).hexdigest() != row['prompt_sha256']):
                raise ValueError('prompt/token mismatch')
            prepared.append((row, ids))
        os.environ.update({'RWKV_V7_ON': '1', 'RWKV_CUDA_ON': '0', 'RWKV_JIT_ON': '1',
            'TORCH_FORCE_WEIGHTS_ONLY_LOAD': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            'PYTHONDONTWRITEBYTECODE': '1'})
        if any(n == 'rwkv' or n.startswith('rwkv.') for n in sys.modules):
            raise ValueError('unverified RWKV module already loaded')
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
        assert torch.cuda.device_count() == 1
        sys.path.insert(0, str(runtime))
        module = importlib.import_module('rwkv.model')
        if Path(module.__file__).resolve() != runtime/'rwkv/model.py':
            raise ValueError('unexpected model module')
        # The official module enables these during import; override afterwards.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        numerical = {'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                     'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                     'cudnn_benchmark': torch.backends.cudnn.benchmark}
        assert not any(numerical.values())
        write(output/'STARTED.json', {'config': config, 'config_sha256': args.config_sha256,
              'gpu': gpu, 'backend': 'official rwkv0.8.30 pureTorch recurrent',
              'numerical_flags': numerical,
              'initial_state_axes': 'H,V,K', 'state_transposed': False, 'gold_opened': False})
        model = module.RWKV(model=str(bound(pilot['checkpoint']))[:-4], strategy='cuda bf16')
        if (model.n_layer, model.n_embd, model.n_head, model.head_size) != (32, 2560, 40, 64):
            raise ValueError('unexpected model dimensions')
        base = {n: t.detach().cpu().clone() for n, t in model.z.items()}
        versions = {n: t._version for n, t in model.z.items()}
        torch.cuda.reset_peak_memory_stats()
        records = []
        for snapshot in spec['states']:
            if sha(bound(snapshot['path'])) != snapshot['sha256']:
                raise ValueError('state checkpoint changed')
            canonical = torch.load(bound(snapshot['path']), map_location='cpu', weights_only=True)
            keys = {f'blocks.{i}.att.time_state' for i in range(32)}
            if set(canonical) != keys or not all(t.dtype == torch.float32 and t.shape == (40,64,64)
                                                 and torch.isfinite(t).all() for t in canonical.values()):
                raise ValueError('invalid canonical state')
            for row, tokens in prepared:
                # New 96-slot state per question. The official x070 branch uses H,V,K directly.
                working = []
                for i in range(32):
                    working.extend([torch.zeros(2560, dtype=torch.bfloat16, device='cuda:0'),
                        canonical[f'blocks.{i}.att.time_state'].to('cuda:0').clone(),
                        torch.zeros(2560, dtype=torch.bfloat16, device='cuda:0')])
                last_length = None
                def decide(prefix):
                    nonlocal forwards, working, last_length
                    forwards += 1
                    if forwards > 704:
                        raise ValueError('forward call cap')
                    if last_length is None:
                        logits, working = model.forward(prefix, working)
                    else:
                        if len(prefix) != last_length+1:
                            raise ValueError('nonsequential generation')
                        logits, working = model.forward([prefix[-1]], working)
                    last_length = len(prefix)
                    if not torch.isfinite(logits).all():
                        raise ValueError('nonfinite logits')
                    return int(logits.argmax())
                started = time.monotonic()
                raw = generate(tokens, vocab.by_id, decide)
                if not all(torch.isfinite(t).all() for t in working):
                    raise ValueError('nonfinite recurrent state')
                result = {'id': row['id'], 'state': snapshot['name'],
                          'state_sha256': snapshot['sha256'], 'prompt_sha256': row['prompt_sha256'],
                          'output': raw, 'seconds': time.monotonic()-started}
                name = f'record-{len(records):04d}.json'
                write(output/name, result)
                records.append({'path': name, 'sha256': sha(output/name)})
                print(json.dumps({'state': snapshot['name'], 'id': row['id'],
                                  'seconds': result['seconds'], 'forwards': forwards}), flush=True)
        equal = {n: t._version == versions[n] and not t.requires_grad
                 and torch.equal(t.cpu(), base[n]) for n,t in model.z.items()}
        if not all(equal.values()) or sha(bound(pilot['checkpoint'])) != CHECKPOINT:
            raise ValueError('base runtime/checkpoint mutated')
        write(output/'BASE-INTEGRITY.json', equal)
        write(output/'COMPLETED.json', {'status': 'GENERATION_COMPLETE', 'records': records,
              'config_sha256': args.config_sha256, 'split': spec['split'], 'gold_opened': False,
              'optimizer_updates': 0, 'forward_calls': forwards, 'seconds': time.monotonic()-began,
              'runtime_tensors_unchanged': len(equal), 'checkpoint_sha256': CHECKPOINT,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'production_promoted': False})
    except BaseException as error:
        write(output/'FAILED.json', {'error': repr(error), 'forward_calls': forwards,
                                    'seconds': time.monotonic()-began})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
