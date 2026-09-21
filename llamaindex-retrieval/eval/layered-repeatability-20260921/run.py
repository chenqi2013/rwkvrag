"""R0 exact-input repeatability. Standalone, immutable records, no retries."""
import argparse
import base64
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import time

import httpx

HERE = Path(__file__).resolve().parent
STATE = Path('/home/chase/rwkvrag/data/experiments/g1j72-assessment-state-2k-20260920/pilot-fp16-v1/preflight-run1/state-zero.pth')
STATE_SHA = '2451b35fbca3c563739b375046e1219427614da6b0bc6909690c17f114ecbb66'
MANIFEST = Path('/home/chase/rwkvrag/data/services/vllm-decode-20260920/ENGINE-SOURCE.json')
MANIFEST_SHA = 'b8be05cd8dddcdb9eac88bca6af0ff2486982edd9407328fb6b064fd0c416168'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    with path.open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def initial(data):
    return bool(data['workers']) and all(
        w['finish_reason'] == 'initial' and w['processed_token_count'] == 0
        and w['pending_tail_token_count'] == 0 for w in data['workers'])


def request_for(row, ref):
    payload = copy.deepcopy(row['payload'])
    if row['stage'] == 'assess':
        if payload.get('vllm_xargs') != {'rwkv_state_read_ref': 'FRESH_ZERO_STATE_REF'}:
            raise ValueError('unexpected historical State binding')
        payload['vllm_xargs']['rwkv_state_read_ref'] = ref
    elif 'vllm_xargs' in payload:
        raise ValueError('State on a non-assessment stage')
    return payload


def response_record(row, response):
    response.raise_for_status()
    body = response.json()
    if body.get('prompt_token_ids') != row['prompt_token_ids']:
        raise ValueError('actual input token mismatch; stop contaminated run')
    choices = body['choices']
    if len(choices) != 1:
        raise ValueError('expected exactly one choice')
    choice = choices[0]
    message = choice['message']
    if (not isinstance(message.get('content'), str) or message.get('reasoning')
            or message.get('reasoning_content') or message.get('tool_calls')):
        raise ValueError('response text transformed')
    text = message['content']
    return {'raw_text': text, 'raw_text_sha256': digest(text.encode()),
            'finish_reason': choice['finish_reason'], 'output_token_ids': choice.get('token_ids'),
            'usage': body['usage'], 'input_tokens_equal': True,
            'historical_raw_equal': text == row['historical_raw_text'],
            'historical_finish_equal': choice['finish_reason'] == row['historical_finish_reason']}


def main(output):
    assert socket.gethostname() == 'rwkv-82'
    assert Path('/etc/machine-id').read_text().strip() == 'bcd164d5ad3a4ab3b0790412e32e69f3'
    output.mkdir(parents=True, exist_ok=False)
    pins = json.loads((HERE / 'PINS.json').read_text())
    assert all(digest((HERE / p).read_bytes()) == h for p, h in pins.items())
    assert digest(STATE.read_bytes()) == STATE_SHA
    assert digest(MANIFEST.read_bytes()) == MANIFEST_SHA
    rows = json.loads((HERE / 'INPUTS.json').read_text())
    assert len(rows) == 442
    save(output / 'BINDING.json', {'pins': pins, 'state_sha256': STATE_SHA,
         'engine_manifest_sha256': MANIFEST_SHA, 'planned': 1326, 'repetitions': 3,
         'order': 'three full passes; index, historical started_at, call_id',
         'concurrency': 1, 'per_call_seconds': 180, 'total_seconds': 10800,
         'retry_count': 0, 'started_at': datetime.now(timezone.utc).isoformat()})
    ref, recorded, status, error = None, [], 'FAILED', None
    began = time.monotonic()
    with httpx.Client(base_url='http://127.0.0.1:18426', timeout=180) as client:
        try:
            health = client.get('/health')
            health.raise_for_status()
            with STATE.open('rb') as f:
                response = client.post('/v1/rwkv/state/upload', files={'file': (STATE.name, f, 'application/octet-stream')})
            save(output / 'UPLOAD.json', {'status': response.status_code, 'body': response.text})
            response.raise_for_status()
            state = response.json()
            ref = state['rwkv_state_ref']
            assert initial(state)
            for repetition in range(3):
                directory = output / f'round-{repetition + 1}'
                directory.mkdir()
                for n, row in enumerate(rows):
                    if time.monotonic() - began >= 10800:
                        raise TimeoutError('global deadline reached')
                    wire = json.dumps(request_for(row, ref), ensure_ascii=False, separators=(',', ':')).encode()
                    record = {'round': repetition + 1, 'ordinal': n, 'call_id': row['call_id'],
                              'index': row['index'], 'stage': row['stage'], 'status': 'failed',
                              'request_body_base64': base64.b64encode(wire).decode(),
                              'request_sha256': digest(wire), 'source_sha256': row['source_sha256'],
                              'prompt_sha256': row['prompt_sha256'],
                              'started_at': datetime.now(timezone.utc).isoformat()}
                    start = time.monotonic()
                    try:
                        response = client.post('/v1/chat/completions', content=wire,
                                               headers={'content-type': 'application/json'},
                                               timeout=min(180, 10800 - (time.monotonic() - began)))
                        record.update(http_status=response.status_code,
                                      response_body_base64=base64.b64encode(response.content).decode(),
                                      response_sha256=digest(response.content))
                        record.update(response_record(row, response), status='recorded')
                    except Exception as exc:
                        record['error'] = repr(exc)
                        raise
                    finally:
                        record['elapsed_ms'] = (time.monotonic() - start) * 1000
                        save(directory / f'{n:04d}.json', record)
                        recorded.append({k: record.get(k) for k in ('round', 'ordinal', 'index', 'stage', 'status', 'raw_text_sha256', 'finish_reason', 'historical_raw_equal')})
                    if len(recorded) % 25 == 0:
                        print(json.dumps({'recorded': len(recorded), 'planned': 1326, 'elapsed_s': time.monotonic()-began}), flush=True)
            response = client.get('/v1/rwkv/state/' + ref)
            save(output / 'STATE-AFTER.json', {'status': response.status_code, 'body': response.text})
            response.raise_for_status()
            assert initial(response.json())
            status = 'EXECUTION_COMPLETE_NOT_QUALITY_GATE'
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            try:
                if ref is not None:
                    response = client.delete('/v1/rwkv/state/' + ref)
                    save(output / 'DROP.json', {'status': response.status_code, 'body': response.text})
                    response.raise_for_status()
            finally:
                unchanged = all(digest((HERE / p).read_bytes()) == h for p, h in pins.items())
                save(output / 'SUMMARY.json', {'status': status, 'error': error, 'recorded': len(recorded),
                     'planned': 1326, 'rows': recorded, 'inputs_unchanged': unchanged,
                     'elapsed_s': time.monotonic() - began, 'semantic_accuracy_verified': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
