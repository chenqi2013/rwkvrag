import argparse
import asyncio
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import perf_counter

import httpx
from protocol import parameters, parse
from analysis import summarize

HERE = Path(__file__).resolve().parent
MODEL = 'rwkvrag-g1j72-vllm-eval'
URL = 'http://127.0.0.1:18426'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    inputs = json.loads((HERE/'inputs.json').read_text())
    schedule = json.loads((HERE/'schedule.json').read_text())
    by_id = {v['case']['id']: v for v in inputs}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'calls').mkdir()
    files = [p for p in HERE.iterdir() if p.is_file()]
    files.extend([HERE.parent/'reader-label-20260920/protocol.py', HERE.parent/'evidence-support-20260920/protocol.py'])
    hashes = {str(p): sha256(p.read_bytes()).hexdigest() for p in files}
    save(args.output/'BINDINGS.json', {'files': hashes,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'started_at': datetime.now(timezone.utc).isoformat(), 'scheduled_calls': len(schedule),
        'seen_cases': len(inputs), 'source': 'local optimized vllm-rwkv; exact deployment binding separately attached'})
    rows = []
    async with httpx.AsyncClient(timeout=120) as client:
        async def health():
            (await client.get(URL+'/health')).raise_for_status()
            response = await client.get(URL+'/v1/models')
            response.raise_for_status()
            data = response.json()
            assert any(m['id'] == MODEL for m in data['data'])
            return data
        save(args.output/'HEALTH-BEFORE.json', await health())
        try:
            for i, item in enumerate(schedule):
                v = by_id[item['case_id']]
                case = v['case']
                payload = {'model': MODEL, 'prompt': v['prompt_token_ids'], **parameters(item)}
                row = {**item, 'expected': case['expected'], 'family': case['family'],
                       'category': case['category'], 'pair': case['pair'], 'prediction': None,
                       'status': 'invalid', 'error': None}
                trace = {'payload': payload, 'prompt': v['prompt'], 'prompt_sha256': v['prompt_sha256']}
                began = perf_counter()
                try:
                    response = await client.post(URL+'/v1/completions', json=payload)
                    trace.update(http_status=response.status_code, raw_response=response.text)
                    response.raise_for_status()
                    data = response.json()
                    choice = data['choices'][0]
                    row.update(raw_text=choice['text'], finish_reason=choice['finish_reason'],
                               token_ids=choice.get('token_ids'), completion_tokens=data['usage']['completion_tokens'])
                    assert data['usage']['prompt_tokens'] == len(v['prompt_token_ids'])
                    assert choice['prompt_token_ids'] == v['prompt_token_ids']
                    assert isinstance(row['token_ids'], list) and len(row['token_ids']) == row['completion_tokens']
                    if choice['finish_reason'] != 'stop':
                        raise ValueError('not_naturally_stopped')
                    row['prediction'] = parse(choice['text'])
                    row['status'] = 'valid'
                except ValueError as error:
                    row['error'] = {'type': type(error).__name__, 'detail': str(error)}
                finally:
                    row['elapsed_ms'] = round((perf_counter()-began)*1000, 2)
                    save(args.output/'calls'/f'{i:04d}.json', {'result': row, 'trace': trace})
                # HTTP/connection failures abort rather than become semantic decisions.
                rows.append(row)
                if (i+1) % 40 == 0:
                    await health()
                    print(json.dumps({'completed_calls': i+1, 'total': len(schedule)}), flush=True)
        finally:
            save(args.output/'ROWS.json', rows)
            save(args.output/'EXECUTION.json', {'complete': len(rows) == len(schedule), 'calls': len(rows)})
    assert all(sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    summary = summarize(rows, inputs)
    save(args.output/'SUMMARY.json', summary)
    print(json.dumps({'complete': True, 'limited_regression_improvement_gate': summary['limited_regression_improvement_gate']}), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
