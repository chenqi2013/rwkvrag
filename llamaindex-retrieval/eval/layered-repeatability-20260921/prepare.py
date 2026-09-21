"""Freeze exact zero-arm historical inputs for R0; no generated/gold answers."""
import hashlib
import json
from pathlib import Path

from llamaindex_retrieval.offline_replay import extract_call

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    source = ROOT / 'data/quality-runs/g1j72-state-layered-2k-20260920/run4'
    index = ROOT / 'artifacts/layered-offline-20260921/run1/FOCUSED-INPUTS.json'
    rows = []
    for binding in json.loads(index.read_text()):
        if binding['arm'] != 'zero':
            continue
        raw = (source / binding['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == binding['trace_sha256']
        trace = json.loads(raw)
        extracted = extract_call(trace)
        request = extracted['payload']
        if trace['stage'] == 'assess':
            assert request['vllm_xargs'] == {'rwkv_state_read_ref': trace['state_binding']['read_ref']}
            request['vllm_xargs']['rwkv_state_read_ref'] = 'FRESH_ZERO_STATE_REF'
        else:
            assert 'vllm_xargs' not in request and trace['state_binding']['read_ref'] is None
        rows.append({
            'index': binding['index'], 'call_id': trace['call_id'], 'stage': trace['stage'],
            'started_at': trace['started_at'], 'source_path': binding['path'],
            'source_sha256': binding['trace_sha256'], 'historical_wire_sha256': extracted['wire_sha256'],
            'payload': request, 'prompt': trace['prompt'], 'prompt_sha256': trace['prompt_sha256'],
            'prompt_token_ids': trace['prompt_token_ids'], 'evidence_ids': trace['evidence_ids'],
            'historical_raw_text': trace['raw_text'], 'historical_raw_sha256': trace['raw_text_sha256'],
            'historical_finish_reason': trace['finish_reason'],
        })
    rows.sort(key=lambda r: (r['index'], r['started_at'], r['call_id']))
    assert len(rows) == 442 and len({r['call_id'] for r in rows}) == 442
    with (HERE / 'INPUTS.json').open('x') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
