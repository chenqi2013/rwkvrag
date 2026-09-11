"""Bounded state-only training or raw evaluation, pinned to the v4 runtime/data."""
import argparse
import json
import signal
import time
from pathlib import Path

from preflight_state import bound, host_guard, sha, write
from train_state import check_config
from pilot_runtime import load_model, check_base, save_state
from state_training import accumulation_groups, read_training_tokens
from state_tokens import Vocabulary
from evaluate_state import generate


def train(torch, model, state, config, pilot, output):
    settings = config['training']
    epochs = settings['epochs']
    if not (1 <= epochs <= 12 and settings['learning_rate'] in [1e-4, 1e-5, 1e-6, 1e-7]
            and settings['accumulation'] == 2 and settings['seed'] == 20260909):
        raise ValueError('unsupported bounded training configuration')
    rows = read_training_tokens(bound(pilot['train_tokens']), pilot['train_sha256'], 21)
    parameters = [p for _, p in state]
    optimizer = torch.optim.AdamW(parameters, lr=settings['learning_rate'],
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    checkpoints = [save_state(torch, state, output, 0)]
    updates = []
    model.train()
    for epoch in range(1, epochs+1):
        groups = accumulation_groups(rows, seed=settings['seed']+epoch-1, size=2)
        for group in groups:
            optimizer.zero_grad(set_to_none=True)
            values = []
            for row in group:
                ids = torch.tensor([row['input_ids'][:-1]], device='cuda:0')
                labels = torch.tensor([row['labels'][1:]], device='cuda:0')
                logits = model(ids)
                mask = labels != -100
                loss = torch.nn.functional.cross_entropy(logits[mask].float(), labels[mask])
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite training loss')
                (loss/len(group)).backward()
                values.append(float(loss.detach()))
                del ids, labels, logits, mask, loss
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError('missing/nonfinite state gradient')
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
            if norm == 0:
                raise ValueError('zero gradient')
            step = len(updates)+1
            record = {'step': step, 'epoch': epoch, 'ids': [r['id'] for r in group],
                      'loss_divisor': len(group), 'sample_losses': values,
                      'gradient_l2_before_clip': norm}
            write(output/f'step-{step:03d}-INTENT.json', record)
            optimizer.step()
            torch.cuda.synchronize()
            if not all(torch.isfinite(p).all() for p in parameters):
                raise ValueError('nonfinite state')
            write(output/f'step-{step:03d}-DONE.json', record)
            updates.append(record)
        checkpoints.append(save_state(torch, state, output, len(updates)))
        epoch_losses = [loss for item in updates if item['epoch'] == epoch for loss in item['sample_losses']]
        print(json.dumps({'epoch': epoch, 'updates': len(updates),
                          'online_mean_train_loss': sum(epoch_losses)/21}), flush=True)
    optimizer.zero_grad(set_to_none=True)
    if len(updates) != epochs*11 or sum(len(r['ids']) for r in updates) != epochs*21:
        raise ValueError('epoch coverage mismatch')
    return {'status': 'TRAIN_COMPLETE', 'optimizer_updates': len(updates),
            'backward_calls': epochs*21, 'checkpoints': checkpoints,
            'dev_or_heldout_content_read': False}


def evaluate(torch, model, state, config, pilot, output):
    spec = config['evaluation']
    if spec['split'] not in ['train', 'dev', 'heldout'] or spec['max_output_tokens'] != 32:
        raise ValueError('unsupported evaluation protocol')
    if sha(bound(spec['inputs'])) != spec['inputs_sha256']:
        raise ValueError('evaluation inputs changed')
    rows = [json.loads(s) for s in bound(spec['inputs']).read_text().splitlines()]
    expected_count = 21 if spec['split'] == 'train' else 11
    if (len(rows) != expected_count or len({r['id'] for r in rows}) != expected_count
            or any(r['split'] != spec['split'] for r in rows)):
        raise ValueError('wrong evaluation coverage')
    vocab = Vocabulary(bound(pilot['peft_source'])/'rwkv_vocab_v20230424.txt')
    prepared = []
    import hashlib
    for row in rows:
        if hashlib.sha256(row['prompt'].encode()).hexdigest() != row['prompt_sha256']:
            raise ValueError('prompt changed')
        tokens = vocab.encode(row['prompt'])
        if len(tokens) != row['input_token_count'] or len(tokens)+31 > 4096:
            raise ValueError('prompt token budget')
        if not row['prompt'].endswith('Assistant: <think></think>\n'):
            raise ValueError('canonical boundary missing')
        prepared.append((row, tokens))
    states = spec['states']
    if not 1 <= len(states) <= 13 or len({s['name'] for s in states}) != len(states):
        raise ValueError('state evaluation cap/identity')
    model.eval().requires_grad_(False)
    records = []
    forwards = 0
    with torch.no_grad():
        for snapshot in states:
            if sha(bound(snapshot['path'])) != snapshot['sha256']:
                raise ValueError('checkpoint changed')
            canonical = torch.load(bound(snapshot['path']), map_location='cpu', weights_only=True)
            if set(canonical) != {n for n, _ in state}:
                raise ValueError('state keys changed')
            for n, p in state:
                value = canonical[n]
                if value.shape != p.shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                    raise ValueError('invalid state tensor')
                p.copy_(value)
            for row, tokens in prepared:
                def decide(prefix):
                    nonlocal forwards
                    forwards += 1
                    if forwards > len(states)*expected_count*32:
                        raise ValueError('generation forward cap')
                    logits = model(torch.tensor([prefix], device='cuda:0'))[0, -1].float()
                    if not torch.isfinite(logits).all():
                        raise ValueError('nonfinite logits')
                    return int(logits.argmax())
                result = {'id': row['id'], 'state': snapshot['name'],
                          'state_sha256': snapshot['sha256'], 'prompt_sha256': row['prompt_sha256'],
                          'output': generate(tokens, vocab.by_id, decide)}
                name = f'record-{len(records):04d}.json'
                write(output/name, result)
                records.append({'path': name, 'sha256': sha(output/name)})
            if not all(torch.equal(p.cpu(), canonical[n]) for n, p in state):
                raise ValueError('initial state changed during evaluation')
            print(json.dumps({'state': snapshot['name'], 'records': len(records),
                              'forwards': forwards}), flush=True)
    return {'status': 'GENERATION_COMPLETE', 'records': records, 'forward_calls': forwards,
            'optimizer_updates': 0, 'gold_opened': False, 'split': spec['split']}


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
            raise ValueError('source/receipt binding changed: '+path)
    pilot = json.loads(bound(config['pilot']).read_text())
    check_config(pilot)
    if config['mode'] not in ['train', 'evaluate'] or not 1 <= config['wall_seconds'] <= 3600:
        raise ValueError('execution limit')
    from datetime import datetime, timezone
    remaining = (datetime.fromisoformat(config['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()
    if remaining <= 60:
        raise ValueError('work window exhausted')
    gpu = host_guard()
    remaining = (datetime.fromisoformat(config['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()
    if remaining <= 60:
        raise ValueError('work window exhausted during GPU check')
    wall = min(config['wall_seconds'], int(remaining)-15)
    output = bound(args.output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    def expired(*unused):
        raise TimeoutError('bounded experiment deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(wall)
    try:
        write(output/'STARTED.json', {'config': config, 'config_sha256': args.config_sha256,
                                     'gpu': gpu, 'effective_wall_seconds': wall})
        torch, model, state, base, weights, versions = load_model(pilot, output)
        if config['mode'] == 'train':
            result = train(torch, model, state, config, pilot, output)
        else:
            result = evaluate(torch, model, state, config, pilot, output)
        equal = check_base(torch, base, weights, versions)
        write(output/'BASE-INTEGRITY.json', equal)
        write(output/'COMPLETED.json', {**result, 'config_sha256': args.config_sha256,
              'base_tensors_unchanged': len(equal), 'seconds': time.monotonic()-began,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'production_promoted': False})
    except BaseException as error:
        write(output/'FAILED.json', {'error': repr(error), 'seconds': time.monotonic()-began,
              'confirmed_updates': len(list(output.glob('step-*-DONE.json'))),
              'attempted_updates': len(list(output.glob('step-*-INTENT.json')))})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
