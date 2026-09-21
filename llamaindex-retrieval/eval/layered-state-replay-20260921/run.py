"""R1 paired old-State replay from identical frozen upstream inputs."""
import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time

import httpx

from replay_transport import MANIFEST, MANIFEST_SHA, STATE, STATE_SHA, digest, initial, request_for, response_record, save

HERE = Path(__file__).resolve().parent
TRAINED = STATE.parents[1] / 'train-run1/dose-2000/state-epoch-2.pth'
TRAINED_SHA = 'd7531abc17996bbac51a119914f7efbe0728e96736fc6eedefa42a9408d6d1d5'


def schedule(rows):
    for round_number, arms in [(1, ('zero', '2000')), (2, ('2000', 'zero'))]:
        for ordinal, row in enumerate(rows):
            for arm in arms:
                yield round_number, ordinal, arm, row


def main(output, r0):
    assert socket.gethostname() == 'rwkv-82'
    assert Path('/etc/machine-id').read_text().strip() == 'bcd164d5ad3a4ab3b0790412e32e69f3'
    assert digest(MANIFEST.read_bytes()) == MANIFEST_SHA
    summary = json.loads((r0 / 'SUMMARY.json').read_text())
    assert summary['status'] == 'EXECUTION_COMPLETE_NOT_QUALITY_GATE'
    assert summary['recorded'] == 1326 and summary['inputs_unchanged']
    pins = json.loads((HERE / 'PINS.json').read_text())
    assert all(digest((HERE / p).read_bytes()) == h for p, h in pins.items())
    states = {'zero': (STATE, STATE_SHA), '2000': (TRAINED, TRAINED_SHA)}
    assert all(digest(path.read_bytes()) == h for path, h in states.values())
    rows = json.loads((HERE / 'INPUTS.json').read_text())
    gold = json.loads((HERE / 'ASSESS-EXPECTATIONS.json').read_text())['rows']
    assert len(rows) == len(gold) == 141
    assert {r['call_id'] for r in rows} == {r['call_id'] for r in gold}
    by_id = {r['call_id']: r for r in gold}
    assert all(r['prompt_sha256'] == by_id[r['call_id']]['prompt_sha256'] for r in rows)
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'BINDING.json', {'pins': pins, 'planned': 564, 'concurrency': 1,
         'per_call_seconds': 180, 'global_seconds': 7200, 'retry_count': 0,
         'states': {a: h for a, (_, h) in states.items()}, 'r0_summary_sha256': digest((r0/'SUMMARY.json').read_bytes()),
         'started_at': datetime.now(timezone.utc).isoformat()})
    refs, recorded, status, error = {}, [], 'FAILED', None
    start = time.monotonic()
    with httpx.Client(base_url='http://127.0.0.1:18426', timeout=180) as client:
        try:
            for arm, (path, _) in states.items():
                with path.open('rb') as f:
                    response = client.post('/v1/rwkv/state/upload', files={'file': (path.name, f, 'application/octet-stream')})
                save(output / f'UPLOAD-{arm}.json', {'status': response.status_code, 'body': response.text})
                response.raise_for_status()
                data = response.json()
                refs[arm] = data['rwkv_state_ref']
                assert initial(data)
            for round_number, ordinal, arm, row in schedule(rows):
                if time.monotonic() - start >= 7200:
                    raise TimeoutError('R1 deadline')
                wire = json.dumps(request_for(row, refs[arm]), ensure_ascii=False, separators=(',', ':')).encode()
                record = {'round': round_number, 'ordinal': ordinal, 'arm': arm, 'call_id': row['call_id'],
                          'index': row['index'], 'stage': 'assess', 'status': 'failed',
                          'request_body_base64': base64.b64encode(wire).decode(), 'request_sha256': digest(wire),
                          'prompt_sha256': row['prompt_sha256'], 'source_sha256': row['source_sha256'],
                          'state_sha256': states[arm][1], 'started_at': datetime.now(timezone.utc).isoformat()}
                began = time.monotonic()
                try:
                    response = client.post('/v1/chat/completions', content=wire,
                                           headers={'content-type': 'application/json'},
                                           timeout=min(180, 7200-(time.monotonic()-start)))
                    record.update(http_status=response.status_code, response_body_base64=base64.b64encode(response.content).decode(),
                                  response_sha256=digest(response.content))
                    record.update(response_record(row, response), status='recorded')
                except BaseException as exc:
                    record['error'] = repr(exc)
                    raise
                finally:
                    record['elapsed_ms'] = (time.monotonic()-began)*1000
                    directory = output / f'round-{round_number}' / arm
                    directory.mkdir(parents=True, exist_ok=True)
                    save(directory / f'{ordinal:04d}.json', record)
                    recorded.append({k: record.get(k) for k in ('round', 'ordinal', 'arm', 'index', 'status', 'raw_text_sha256', 'finish_reason')})
                if len(recorded) % 20 == 0:
                    print(json.dumps({'recorded': len(recorded), 'planned': 564, 'elapsed_s': time.monotonic()-start}), flush=True)
            for arm, ref in refs.items():
                response = client.get('/v1/rwkv/state/' + ref)
                save(output / f'AFTER-{arm}.json', {'status': response.status_code, 'body': response.text})
                response.raise_for_status()
                assert initial(response.json())
            status = 'EXECUTION_COMPLETE_REQUIRES_SEMANTIC_REVIEW'
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            cleanup_errors = []
            for arm, ref in refs.items():
                try:
                    response = client.delete('/v1/rwkv/state/' + ref)
                    save(output / f'DROP-{arm}.json', {'status': response.status_code, 'body': response.text})
                    response.raise_for_status()
                except Exception as exc:
                    cleanup_errors.append(repr(exc))
            save(output / 'SUMMARY.json', {'status': status, 'error': error, 'planned': 564, 'recorded': len(recorded),
                 'rows': recorded, 'cleanup_errors': cleanup_errors, 'elapsed_s': time.monotonic()-start,
                 'inputs_unchanged': all(digest((HERE / p).read_bytes()) == h for p, h in pins.items()),
                 'semantic_accuracy_verified': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--r0', type=Path, required=True)
    args = parser.parse_args()
    main(args.output, args.r0)
