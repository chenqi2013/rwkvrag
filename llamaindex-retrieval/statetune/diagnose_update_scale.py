"""Train-only, reversible state perturbations along the actual first Adam step."""
import argparse
import json
import signal
import time
from pathlib import Path

from preflight_state import bound, host_guard, sha, write
from train_state import check_config
from pilot_runtime import load_model, check_base
from state_training import read_training_tokens, longest_training_row
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
            raise ValueError('binding changed: ' + path)
    pilot = json.loads(bound(config['pilot']).read_text())
    check_config(pilot)
    if config['scales'] != [0.001, 0.0001, 0.00001, 0.000001, 0.0000001]:
        raise ValueError('fixed perturbation grid changed')
    if config['limits'] != {'wall_seconds': 900, 'backwards': 1, 'forwards': 200,
                            'optimizer_steps': 1, 'generation_tokens': 32}:
        raise ValueError('diagnostic limits changed')
    gpu = host_guard()
    output = bound(args.output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    def expired(*unused):
        raise TimeoutError('900-second diagnostic limit')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(900)
    forwards = backwards = updates = 0
    try:
        rows = read_training_tokens(bound(pilot['train_tokens']), pilot['train_sha256'], 21)
        row = longest_training_row(rows)
        if row['id'] != config['sample_id']:
            raise ValueError('longest training sample changed')
        write(output/'STARTED.json', {'config': config, 'config_sha256': args.config_sha256,
                                     'gpu': gpu, 'train_only': True})
        torch, model, state, base, weights, versions = load_model(pilot, output)
        parameters = [p for _, p in state]
        def loss_for(item):
            nonlocal forwards
            forwards += 1
            if forwards > 200:
                raise ValueError('forward cap')
            ids = torch.tensor([item['input_ids'][:-1]], device='cuda:0')
            labels = torch.tensor([item['labels'][1:]], device='cuda:0')
            logits = model(ids)
            mask = labels != -100
            loss = torch.nn.functional.cross_entropy(logits[mask].float(), labels[mask])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite loss')
            return loss
        model.train()
        loss = loss_for(row)
        before = float(loss.detach())
        backwards += 1
        loss.backward()
        del loss
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError('invalid gradient')
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
        gradients = [p.grad.detach().clone() for p in parameters]
        optimizer = torch.optim.AdamW(parameters, lr=0.001, betas=(0.9, 0.999),
                                     eps=1e-8, weight_decay=0)
        write(output/'step-01-INTENT.json', {'sample_id': row['id'], 'loss': before,
                                           'gradient_l2_before_clip': norm})
        optimizer.step()
        updates += 1
        torch.cuda.synchronize()
        directions = [p.detach().clone() / 0.001 for p in parameters]
        if not all(torch.isfinite(d).all() for d in directions):
            raise ValueError('nonfinite Adam direction')
        predicted_derivative = float(sum((g.double()*d.double()).sum()
                                         for g, d in zip(gradients, directions))) * norm
        write(output/'step-01-DONE.json', {'diagnostic_only': True,
                                          'predicted_derivative': predicted_derivative})
        optimizer.zero_grad(set_to_none=True)
        del optimizer, gradients
        model.eval()
        def set_scale(scale):
            with torch.no_grad():
                for p, direction in zip(parameters, directions):
                    p.copy_(direction * scale)
        results = []
        with torch.no_grad():
            set_scale(0)
            zero = float(loss_for(row))
            for scale in config['scales']:
                pair = {'scale': scale}
                for sign, name in [(1, 'descent_loss'), (-1, 'opposite_loss')]:
                    set_scale(sign * scale)
                    pair[name] = float(loss_for(row))
                pair['central_derivative'] = (pair['descent_loss'] - pair['opposite_loss'])/(2*scale)
                results.append(pair)
                write(output/f'scale-{scale:g}.json', pair)
                print(json.dumps(pair), flush=True)
            # One candidate chosen strictly on this train sample, then checked on all train.
            best = min(results, key=lambda r: (r['descent_loss'], r['scale']))
            scale = best['scale'] if best['descent_loss'] < zero * 0.99 else None
            vocab = Vocabulary(bound(pilot['peft_source'])/'rwkv_vocab_v20230424.txt')
            evaluations = []
            for candidate in ([0, scale] if scale is not None else [0]):
                set_scale(candidate)
                losses = [{'id': item['id'], 'loss': float(loss_for(item))} for item in rows]
                def decide(prefix):
                    nonlocal forwards
                    forwards += 1
                    if forwards > 200:
                        raise ValueError('forward cap')
                    logits = model(torch.tensor([prefix], device='cuda:0'))[0, -1].float()
                    if not torch.isfinite(logits).all():
                        raise ValueError('nonfinite generation logits')
                    return int(logits.argmax())
                raw = generate(row['input_ids'][:row['prompt_tokens']], vocab.by_id, decide)
                evaluations.append({'scale': candidate, 'train_losses': losses,
                                    'mean_train_loss': sum(x['loss'] for x in losses)/len(losses),
                                    'sample_output': raw})
            set_scale(0)
        assert all(torch.count_nonzero(p) == 0 for p in parameters)
        equal = check_base(torch, base, weights, versions)
        write(output/'BASE-INTEGRITY.json', equal)
        write(output/'COMPLETED.json', {'status': 'DIAGNOSTIC_COMPLETE',
              'config_sha256': args.config_sha256, 'zero_train_mode_loss': before,
              'zero_eval_mode_loss': zero, 'gradient_l2_before_clip': norm,
              'predicted_derivative': predicted_derivative, 'grid': results,
              'selected_scale_on_train': scale, 'evaluations': evaluations,
              'forwards': forwards, 'backwards': backwards, 'optimizer_updates': updates,
              'state_restored_zero': True, 'base_tensors_unchanged': len(equal),
              'dev_or_heldout_content_read': False, 'seconds': time.monotonic()-began})
    except BaseException as error:
        write(output/'FAILED.json', {'error': repr(error), 'forwards': forwards,
              'backwards': backwards, 'optimizer_updates': updates,
              'seconds': time.monotonic()-began})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
