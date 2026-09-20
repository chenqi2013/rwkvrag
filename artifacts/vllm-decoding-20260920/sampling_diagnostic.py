"""Separate API parameter smoke check, not a semantic quality evaluation.

Fixed before execution: one open-ended prompt, eight requests, no retries.
Top1 seeds 11/29; Fake Think seeds 11/29/47; full-vocabulary sampling
at temperature 1 and top_p 1 with seeds 11/29/47. Only check whether the
same endpoint can produce different continuations; do not rank quality.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import urllib.request

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = Path('data/services/vllm-decode-20260920/model-metadata')
    tokenizer = AutoTokenizer.from_pretrained(metadata)
    prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': '请写一句富有想象力的短句，不超过十五个字。'}],
        tokenize=False, add_generation_prompt=True, rwkv_generation_prompt='fake_think')
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    configs = [('top1', 1, 1.0, s) for s in (11, 29)]
    configs += [('fake', 32, 0.28, s) for s in (11, 29, 47)]
    configs += [('full', 0, 1.0, s) for s in (11, 29, 47)]
    plan = {'started_at': datetime.now(timezone.utc).isoformat(), 'prompt': prompt,
            'prompt_token_ids': ids, 'conditions': configs,
            'script_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
            'purpose': 'parameter smoke check only, not a quality comparison'}
    (args.output/'PLAN.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    for i, (arm, top_k, top_p, seed) in enumerate(configs):
        payload = {'model': 'rwkvrag-g1j72-vllm-eval', 'prompt': ids,
                   'temperature': 1.0, 'top_k': top_k, 'top_p': top_p, 'seed': seed,
                   'presence_penalty': 0, 'frequency_penalty': 0, 'penalty_decay': .996,
                   'max_tokens': 40, 'stop_token_ids': [0],
                   'stop': ['✿', '\nUser:', '\n### User'],
                   'skip_special_tokens': False, 'return_token_ids': True}
        request = urllib.request.Request('http://127.0.0.1:18426/v1/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read().decode()
        record = {'arm': arm, 'payload': payload, 'raw_response': raw}
        (args.output/f'{i:02d}.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
        result = json.loads(raw)['choices'][0]
        print(json.dumps({'arm': arm, 'seed': seed, 'text': result['text'],
                          'finish_reason': result['finish_reason']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
