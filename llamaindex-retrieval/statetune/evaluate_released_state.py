"""Raw official recurrent evaluation of a pinned released dataset, with no gold reads.

The executed v5 evaluator remains frozen. This entry consumes the released split
SHA and explicit row count instead of the v5-only eleven-case contract.
"""
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
from state_tokens import Vocabulary
from evaluate_state import generate



REQUIRED_CODE = ["statetune/evaluate_released_state.py", "statetune/preflight_state.py",
                 "statetune/evaluate_state.py", "statetune/train_state.py",
                 "statetune/pilot_runtime.py", "src/llamaindex_retrieval/state_tokens.py",
                 "src/llamaindex_retrieval/state_training.py"]
ZERO_SHA = "f4503080abf4210de4240c9e58a7375001043e7fc3d1f3a8d5aceca6c3254fd3"
OFFICIAL_MANIFEST_SHA = "8aa4c2cd14c793febc365420f3ab26090c1b9526e03a41570453d26d7fbd1474"


def validate(config):
    """Verify released inputs and runtime on CPU, without reading gold content."""
    if (config.get('schema') != 'rwkv_released_evaluation_v1'
            or type(config.get('wall_seconds')) is not int
            or not 60 <= config['wall_seconds'] <= 3600
            or datetime.fromisoformat(config['deadline_utc']).tzinfo is None):
        raise ValueError('evaluation execution contract')
    required = {'llamaindex-retrieval/'+name for name in REQUIRED_CODE}
    required.add(config['official_runtime']+'/RUNTIME-SOURCE.json')
    if not required <= set(config['bindings']):
        raise ValueError('missing runtime bindings')
    for path, digest in config['bindings'].items():
        if sha(bound(path)) != digest:
            raise ValueError('bound runtime changed')
    release_path = bound(config['release'])
    if sha(release_path) != config['release_sha256']:
        raise ValueError('release receipt changed')
    release = json.loads(release_path.read_text())
    if (release.get('schema') != 'rwkv_reader_release_v1'
            or release.get('training_data_admitted') is not True
            or release.get('prompt_protocol') != 'rwkv_g1j_no_think_v1'
            or release.get('input_layout') != 'task_last'
            or release.get('max_sequence_tokens') != 4096
            or release.get('max_generation_tokens') != 32):
        raise ValueError('unadmitted evaluation release')
    spec = config['evaluation']
    if (spec['split'] not in ['dev', 'heldout'] or spec['max_output_tokens'] != 32
            or type(spec['count']) is not int or not 2 <= spec['count'] <= 256
            or not 1 <= len(spec['states']) <= 7
            or (spec['split'] == 'heldout' and len(spec['states']) != 2)):
        raise ValueError('evaluation coverage/budget contract')
    if (spec['inputs_sha256'] != release['dataset_files'][spec['split']+'.inputs.jsonl']
            or spec['gold_sha256'] != release['dataset_files'][spec['split']+'.gold.jsonl']
            or sha(bound(spec['inputs'])) != spec['inputs_sha256']):
        raise ValueError('released split binding mismatch')
    names = [s['name'] for s in spec['states']]
    if (len(names) != len(set(names)) or names[0] != 'zero'
            or spec['states'][0]['sha256'] != ZERO_SHA):
        raise ValueError('unique state names and fixed zero control required')
    for snapshot in spec['states']:
        if sha(bound(snapshot['path'])) != snapshot['sha256']:
            raise ValueError('state checkpoint changed')
    runtime = bound(config['official_runtime'])
    if sha(runtime/'RUNTIME-SOURCE.json') != OFFICIAL_MANIFEST_SHA:
        raise ValueError('complete frozen official manifest required')
    manifest = json.loads((runtime/'RUNTIME-SOURCE.json').read_text())
    if manifest['installed_version'] != '0.8.30' or manifest['compile_cuda_extension']:
        raise ValueError('official runtime version changed')
    for name, pin in manifest['files'].items():
        path = (runtime/'rwkv'/name).resolve()
        if not path.is_relative_to(runtime/'rwkv') or sha(path) != pin['sha256']:
            raise ValueError('official source changed')
    vocab_path = runtime/'rwkv/rwkv_vocab_v20230424.txt'
    if sha(vocab_path) != release['vocabulary_sha256']:
        raise ValueError('released vocabulary changed')
    vocab = Vocabulary(vocab_path)
    rows = [json.loads(s) for s in bound(spec['inputs']).read_text().splitlines()]
    if (len(rows) != spec['count'] or len({r['id'] for r in rows}) != spec['count']
            or any(r['split'] != spec['split'] or r.get('input_layout') != 'task_last'
                   or r.get('prompt_protocol') != 'rwkv_g1j_no_think_v1' for r in rows)):
        raise ValueError('released input identity/layout mismatch')
    import hashlib
    for row in rows:
        ids = vocab.encode(row['prompt'])
        if (len(ids) != row['input_token_count'] or len(ids)+31 > 4096
                or not row['prompt'].endswith('Assistant: <think></think>\n')
                or hashlib.sha256(row['prompt'].encode()).hexdigest() != row['prompt_sha256']):
            raise ValueError('released prompt/token mismatch')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--config-sha256', required=True)
    ap.add_argument('--output')
    ap.add_argument('--validate-only', action='store_true')
    args = ap.parse_args()
    if sha(args.config) != args.config_sha256:
        raise ValueError('configuration changed')
    config = json.loads(args.config.read_text())
    validate(config)
    if args.validate_only:
        print(json.dumps({'status': 'CPU_VALIDATED', 'count': config['evaluation']['count'],
                          'gold_opened': False, 'model_calls': 0, 'optimizer_updates': 0}))
        return
    if not args.output:
        ap.error('--output required for GPU evaluation')
    pilot = {'checkpoint': config['checkpoint']}
    spec = config['evaluation']
    if (datetime.fromisoformat(config['deadline_utc'])-datetime.now(timezone.utc)).total_seconds() <= 60:
        raise ValueError('work window exhausted')
    gpu = host_guard()
    remaining = (datetime.fromisoformat(config['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()
    if remaining <= 60:
        raise ValueError('work window exhausted')
    output = bound(args.output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    def expired(*unused):
        raise TimeoutError('recurrent evaluation deadline')
    def interrupted(*unused):
        raise InterruptedError('released evaluation received SIGTERM')
    previous_alarm = signal.signal(signal.SIGALRM, expired)
    previous_term = signal.signal(signal.SIGTERM, interrupted)
    signal.alarm(min(config['wall_seconds'], int(remaining)-15))
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
        if len(rows) != spec['count'] or len({r['id'] for r in rows}) != spec['count'] or any(r['split'] != spec['split'] for r in rows):
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
                    if forwards > len(spec['states'])*spec['count']*32:
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
        signal.signal(signal.SIGALRM, previous_alarm)
        signal.signal(signal.SIGTERM, previous_term)


if __name__ == '__main__':
    main()
