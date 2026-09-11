"""Fixed train-only class-weight ablation; shares the frozen v4 model runtime."""
import argparse
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from preflight_state import bound, host_guard, sha, write
from train_state import check_config
from pilot_runtime import load_model, check_base, save_state
from state_training import accumulation_groups, read_training_tokens
from state_tokens import Vocabulary


def example_weights(rows, vocab, negative_weight):
    """Global mean-one weights, preserving the actual last-group denominator."""
    raw = {}
    for row in rows:
        target = b''.join(vocab.by_id[t] for t in row['input_ids'][row['prompt_tokens']:-1])
        raw[row['id']] = negative_weight if target == b'NONE' else 1.0
    if len(raw) != len(rows) or not rows:
        raise ValueError('empty/duplicate training identity')
    normalizer = len(rows)/sum(raw.values())
    return {name: value*normalizer for name, value in raw.items()}


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
    settings = config['training']
    if (settings['epochs'] != 12 or settings['learning_rate'] != 1e-4
            or settings['negative_weight'] not in [2, 4] or settings['seed'] != 20260909
            or settings['accumulation'] != 2 or config['wall_seconds'] != 2400):
        raise ValueError('weighted trial contract changed')
    gpu = host_guard()
    remaining = (datetime.fromisoformat(config['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()
    if remaining <= 60:
        raise ValueError('work window exhausted')
    wall = min(2400, int(remaining)-15)
    output = bound(args.output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    def expired(*unused):
        raise TimeoutError('weighted experiment deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(wall)
    updates = []
    try:
        rows = read_training_tokens(bound(pilot['train_tokens']), pilot['train_sha256'], 21)
        vocab = Vocabulary(bound(pilot['peft_source'])/'rwkv_vocab_v20230424.txt')
        weights_by_id = example_weights(rows, vocab, settings['negative_weight'])
        write(output/'STARTED.json', {'config': config, 'config_sha256': args.config_sha256,
              'gpu': gpu, 'effective_wall_seconds': wall, 'weights_by_id': weights_by_id,
              'dev_or_heldout_content_read': False})
        torch, model, state, base, weights, versions = load_model(pilot, output)
        parameters = [p for _, p in state]
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        checkpoints = [save_state(torch, state, output, 0)]
        model.train()
        for epoch in range(1, 13):
            for group in accumulation_groups(rows, seed=20260909+epoch-1, size=2):
                optimizer.zero_grad(set_to_none=True)
                values = []
                for row in group:
                    ids = torch.tensor([row['input_ids'][:-1]], device='cuda:0')
                    labels = torch.tensor([row['labels'][1:]], device='cuda:0')
                    logits = model(ids)
                    mask = labels != -100
                    loss = torch.nn.functional.cross_entropy(logits[mask].float(), labels[mask])
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite loss')
                    (loss*weights_by_id[row['id']]/len(group)).backward()
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
                          'sample_weights': [weights_by_id[r['id']] for r in group],
                          'gradient_l2_before_clip': norm}
                write(output/f'step-{step:03d}-INTENT.json', record)
                optimizer.step()
                torch.cuda.synchronize()
                if not all(torch.isfinite(p).all() for p in parameters):
                    raise ValueError('nonfinite state')
                write(output/f'step-{step:03d}-DONE.json', record)
                updates.append(record)
            checkpoints.append(save_state(torch, state, output, len(updates)))
            losses = [loss for item in updates if item['epoch'] == epoch for loss in item['sample_losses']]
            print(json.dumps({'epoch': epoch, 'updates': len(updates),
                              'online_mean_unweighted_loss': sum(losses)/21}), flush=True)
        optimizer.zero_grad(set_to_none=True)
        if len(updates) != 132 or sum(len(r['ids']) for r in updates) != 252:
            raise ValueError('training coverage changed')
        equal = check_base(torch, base, weights, versions)
        write(output/'BASE-INTEGRITY.json', equal)
        write(output/'COMPLETED.json', {'status': 'TRAIN_COMPLETE', 'optimizer_updates': len(updates),
              'backward_calls': 252, 'checkpoints': checkpoints, 'config_sha256': args.config_sha256,
              'base_tensors_unchanged': len(equal), 'seconds': time.monotonic()-began,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'negative_weight': settings['negative_weight'], 'weights_by_id': weights_by_id,
              'dev_or_heldout_content_read': False, 'production_promoted': False})
    except BaseException as error:
        write(output/'FAILED.json', {'error': repr(error), 'seconds': time.monotonic()-began,
              'confirmed_updates': len(list(output.glob('step-*-DONE.json'))),
              'attempted_updates': len(list(output.glob('step-*-INTENT.json')))})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
