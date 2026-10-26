"""Immutable, local diagnostic captures; never automatically create training labels."""
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path


def observations(record):
    result = record.get('result') or {}
    issues = []
    for index, call in enumerate(record.get('model_calls', [])):
        if call.get('status') != 'completed':
            issues.append({'category': 'model_execution', 'stage': call.get('stage'),
                'status': call.get('status'), 'call_index': index,
                'raw_sha256': call.get('raw_sha256')})
        for message in call.get('messages', []):
            try:
                frame = json.loads(message['content'].split('\n', 1)[1])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if frame.get('representation') in {
                    'derived_readings_with_source_lineage', 'derived_summaries_with_source_lineage'}:
                issues.append({'category': 'originals_not_forwarded', 'stage': call.get('stage'),
                               'call_index': index})
            if call.get('stage') == 'plan' and 'as_of' not in frame:
                issues.append({'category': 'query_clock_not_supplied', 'call_index': index,
                               'applicability': 'requires_review'})
    for error in result.get('errors', []):
        issues.append({'category': 'pipeline_incomplete', 'detail': error})
    if record.get('status') != 'completed':
        issues.append({'category': 'job_incomplete', 'status': record.get('status')})
    return issues


def private_write(path, data):
    with path.open('xb') as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(data)


def capture_record(record_path, folder):
    """Content-address a complete raw record. Repeat capture is idempotent."""
    record_path, folder = Path(record_path), Path(folder)
    raw = record_path.read_bytes()
    record = json.loads(raw)
    checksum = sha256(raw).hexdigest()
    identity = str(record['id'])
    if len(identity) != 32 or any(c not in '0123456789abcdef' for c in identity):
        raise ValueError('invalid diagnostic job identity')
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = folder / (identity + '-' + checksum[:16])
    try:
        target.mkdir(mode=0o700)
    except FileExistsError:
        manifest = json.loads((target / 'manifest.json').read_text())
        stored = (target / 'record.json.gz').read_bytes()
        if (manifest['record_sha256'] != checksum or sha256(stored).hexdigest() != manifest['archive_sha256']
                or gzip.decompress(stored) != raw):
            raise ValueError('diagnostic capture integrity mismatch')
        return target
    packed = gzip.compress(raw, mtime=0)
    private_write(target / 'record.json.gz', packed)
    manifest = {'schema': 'trial-diagnostic-capture-v1', 'job_id': identity,
        'record_path': str(record_path.resolve()), 'record_sha256': checksum,
        'archive_sha256': sha256(packed).hexdigest(), 'record_bytes': len(raw),
        'protocol': result_protocol(record), 'observations': observations(record),
        'semantic_review': 'pending', 'training_eligible': False,
        'use': 'diagnostic/evaluation only; synthesize future training from disjoint new sources'}
    private_write(target / 'manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2).encode())
    return target


def result_protocol(record):
    return ((record.get('result') or {}).get('answer') or {}).get('generation', {}).get('protocol')
