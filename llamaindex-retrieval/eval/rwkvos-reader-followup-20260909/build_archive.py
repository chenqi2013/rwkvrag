#!/usr/bin/env python3
"""Prepare/build frozen followup records; reuse the published safe tar verifier.

No model calls. Prepare is forbidden before the final complete declaration.
Credential values are scanned in memory, never written. Existing archives stay
external dependencies. PEFT source is research tooling, not production RAG code.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys


OLD = Path(__file__).resolve().parents[1] / 'rwkvos-29b-20260909'
WRAPPER = OLD / 'build_archive.py'
WRAPPER_SHA = '4285c54128f9d432794ddbee0270bf7cbec03e4d11283f2831ea59cec2bb7f6a'
if hashlib.sha256(WRAPPER.read_bytes()).hexdigest() != WRAPPER_SHA:
    raise RuntimeError('published credential/archive wrapper changed')
spec = importlib.util.spec_from_file_location('followup_published_archive', WRAPPER)
published = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = published
spec.loader.exec_module(published)
builder = published.builder
CredentialScan = published.CredentialScan
original_secret_scan = published.published_secret_scan
BASE_SHA = published.BASE_SHA256
OLD_MANIFEST_SHA = 'ef9eed2cd9abc82542a60138d30f4fa8ec6580592a1a2a1490f050f688a6b754'
SOURCE_MANIFEST_SHA = '15d64abc9a9ce960c89b1ae4bfbcc0c8730782e4f3ec3f370fb669716fdc3b9f'

ROOTS = {
    'backend-batch-audit-v1': 'backend-audit',
    'batch-parity-v1': 'batch-parity',
    'batch-original-wire-v1': 'original-wire',
    'reader-fixtures-v1': 'reader-fixtures',
    'reader-task-design-v1': 'reader-design',
    'reader-per-task-v1': 'reader-per-task',
    'reader-focused-task-v1': 'reader-focused',
    'reader-presentation-inputs-v1': 'reader-presentation',
    'reader-presentation-v1': 'reader-presentation',
    'reader-open-thinking-v1': 'reader-open-thinking',
    'reader-examples-v1': 'reader-examples',
    'external-zero-state-v1': 'external-zero-state',
    'local-statetune-readiness-v1': 'local-readiness',
    'local-reader-replay-v1': 'local-reader',
    'local-state-smoke-v1': 'local-state-smoke',
    'local-runtime-alignment-v1': 'local-runtime-alignment',
    'peft-source-v1': 'research-peft-source',
}
REQUIRED_EXPERIMENTS = {
    'batch-parity-v1', 'batch-original-wire-v1', 'reader-per-task-v1',
    'reader-focused-task-v1', 'reader-presentation-v1',
    'reader-open-thinking-v1', 'reader-examples-v1',
    'external-zero-state-v1', 'local-reader-replay-v1',
    'local-state-smoke-v1', 'local-runtime-alignment-v1',
}
# Only exact path + full-file SHA + JSON pointer to reviewed permission prose.
# Add entries only after inspecting the actual false positive. Empty by default.
SCAN_EXCEPTIONS = {
    'local-statetune-readiness-v1/DOWNLOAD-PLAN.json': {
        'sha256': 'b8272a55d80ec0c90c849f5d8ba3bf0382f3753080982ec647b272058e691fb1',
        'json_pointer': '/authorization',
    },
}
BINARY_EVIDENCE = {
    'local-state-smoke-v1/gpu-01/untrained-zero-time-state.pth': {
        'bytes': 10495555,
        'sha256': 'd9e760c5cbf6101ba86266c944aa101990f43ee186d49b416578cc83dc932aff',
        'role': 'All-zero 32-layer diagnostic state; untrained and zero optimizer updates. Not uploaded during the smoke phase; a later external-zero-state-v1 upload attempt ended without a response, so the remote outcome is unknown. Not the model checkpoint.',
    },
    'local-runtime-alignment-v1/asymmetric-initial-state.pth': {
        'bytes': 10495555,
        'sha256': '5dd2f9ad0b17ee2aebf1b3f19bbf55e0bde731f6ccc9758e3f0c09da72735944',
        'role': 'Fixed-seed asymmetric input state for numerical comparison; untrained, no optimizer updates, not uploaded.',
    },
    'local-runtime-alignment-v1/peft-fla-asymmetric-logits.pth': {
        'bytes': 10487337,
        'sha256': '139d7d69b522acb5f745e862986c557fd517f42935c2a1507ded44cada5c6dbe',
        'role': 'Full 40 x 65536 diagnostic logits from PEFT/FLA asymmetric-state forward; not model weights.',
    },
    'local-runtime-alignment-v1/peft-fla-zero-logits.pth': {
        'bytes': 10487337,
        'sha256': '4cc089dff4463d0077b78db490abdca049287dbb39e0d5dc72ade71fef25b5a5',
        'role': 'Full 40 x 65536 diagnostic logits from PEFT/FLA zero-state forward; not model weights.',
    },
    'local-runtime-alignment-v1/rwkv-torch-asymmetric-logits.pth': {
        'bytes': 10487337,
        'sha256': 'dba708feac009fae91fcc2951c6f97e15ed2ca0127ebe8c7c4f5901f9ef02276',
        'role': 'Full 40 x 65536 diagnostic logits from pure Torch asymmetric-state forward; not model weights.',
    },
    'local-runtime-alignment-v1/rwkv-torch-zero-logits.pth': {
        'bytes': 10487337,
        'sha256': '300e38685ca60a9bc0ef97ba790b93650b3c829189bf7f89d9fe691db801407c',
        'role': 'Full 40 x 65536 diagnostic logits from pure Torch zero-state forward; not model weights.',
    },
}
EXACT_SOURCE_EXCLUSIONS = {
    'peft-source-v1/json2binidx_tool/20B_tokenizer.json': {
        'bytes': 2467981,
        'sha256': '56ac4821e129d2c520fdaba60abd920fa852ada51b45c0dd52bbb6bd8c985ade',
        'reason': 'Unused 20B tokenizer vocabulary has numeric password/cookie token keys; preserve strict credential-key scanning and restore this exact public Git blob separately if needed.',
        'repository': 'https://github.com/Joluck/RWKV-PEFT',
        'commit': '5704c39f8ab1d2ac63936ab392aadb6ba526e1a5',
        'git_path': 'json2binidx_tool/20B_tokenizer.json',
        'local_git_blob_bytes_verified': True,
        'restore': 'Obtain the named file at the pinned commit and verify its exact bytes/SHA. It was not used by this experiment; no public copy is duplicated here.',
    },
    'peft-source-v1/scripts/validate_and_start_round71_v4_vllm.sh': {
        'bytes': 5694,
        'sha256': '141a0857b1b5670a6c311f80eefbf7a911d292f6b33a37450497ef605ec5f1f4',
        'reason': 'Unexecuted historical 13.3B launcher contains bearer-style text. Excluded without relaxing authentication scanning; original private snapshot/pins unchanged.',
        'tracked_in_pinned_commit': False,
        'public_restore_url': None,
        'restore': 'Only from the owner original local peft-source-v1 snapshot, checking exact bytes/SHA. No public restoration URL is claimed; the file is not needed for this run.',
    },
}
SIGNED_URL = re.compile(
    rb'[?&](?:x-amz-signature|x-goog-signature|signature|policy|key-pair-id|sig)=', re.I
)


def scan_public(data, name, depth=0):
    if SIGNED_URL.search(data):
        raise ValueError('signed URL rejected: ' + name + '; no values printed')
    relative = name.removeprefix('records/')
    rule = SCAN_EXCEPTIONS.get(relative)
    if rule is not None:
        if set(rule) != {'sha256', 'json_pointer'}:
            raise ValueError('scan exception has unsupported fields')
        if hashlib.sha256(data).hexdigest() != rule['sha256']:
            raise ValueError('reviewed scan exception bytes changed: ' + relative)
        pointer = rule['json_pointer']
        if not isinstance(pointer, str) or not pointer.startswith('/'):
            raise ValueError('scan exception must use an exact JSON pointer')
        obj = json.loads(data)
        parent = obj
        keys = [part.replace('~1', '/').replace('~0', '~') for part in pointer[1:].split('/')]
        for key in keys[:-1]:
            parent = parent[int(key)] if isinstance(parent, list) else parent[key]
        key = keys[-1]
        if key != 'authorization' or not isinstance(parent, dict) or not isinstance(parent[key], str):
            raise ValueError('only reviewed authorization prose may receive a scan-key exception')
        if 'reviewed_experiment_permission_text' in parent:
            raise ValueError('scan-only replacement key already exists')
        parent['reviewed_experiment_permission_text'] = parent.pop(key)
        # Values, including the permission sentence, remain scanned. Raw file is untouched.
        data = json.dumps(obj, ensure_ascii=False).encode()
    original_secret_scan(data, name, depth)


builder.secret_scan = scan_public


def exclusion(relative):
    if relative.as_posix() in BINARY_EVIDENCE:
        return None
    if relative.as_posix() in EXACT_SOURCE_EXCLUSIONS:
        return EXACT_SOURCE_EXCLUSIONS[relative.as_posix()]['reason']
    if any(part in {'checkpoints', 'weights', 'cache', '.cache', '.pytest_cache', '.ruff_cache'}
           or part.startswith('.credentials') for part in relative.parts):
        return 'weights/cache/private credential directory excluded without reading contents; pinned metadata and download receipts retained'
    if relative.suffix.lower() in {'.pth', '.pt', '.ckpt', '.safetensors', '.bin', '.part', '.partial'}:
        return 'model or binary download payload excluded; public URL/revision/size/SHA receipts retained'
    return builder.exclusion(relative)


def check_release(records, freeze, expected_sha):
    relative = freeze.resolve().relative_to(records).as_posix()
    path = builder.checked_source(records, relative)
    if builder.file_sha(path) != expected_sha:
        raise ValueError('final freeze SHA differs')
    data = builder.read_json(path)
    if data.get('status') != 'complete':
        raise ValueError('final freeze must explicitly declare status=complete')
    bindings = data.get('bindings')
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError('final freeze requires complete relative-path SHA bindings')
    for name, digest in bindings.items():
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('invalid binding SHA: ' + name)
        source = builder.checked_source(records, name)
        if builder.file_sha(source) != digest:
            raise ValueError('frozen file changed: ' + name)
    experiments = data.get('completed_experiments')
    if (not isinstance(experiments, list) or len(experiments) != len(set(experiments))
            or not REQUIRED_EXPERIMENTS <= set(experiments)):
        raise ValueError('final freeze must include all eleven ended execution groups, including preserved failures and local diagnostics')
    for name in experiments:
        if len(builder.safe_name(name).parts) != 1:
            raise ValueError('experiment must be a top-level record directory')
        receipt = name + '/COMPLETED.json'
        if receipt not in bindings:
            raise ValueError('completed experiment lacks bound terminal receipt: ' + name)
    roots = dict(ROOTS)
    additions = data.get('archive_roots', {})
    if not isinstance(additions, dict):
        raise ValueError('archive_roots must map top-level directory to group')
    for name, group in additions.items():
        if len(builder.safe_name(name).parts) != 1 or not re.fullmatch(r'[a-z0-9][a-z0-9-]*', group):
            raise ValueError('unsafe archive group declaration')
        if name in roots and roots[name] != group:
            raise ValueError('cannot reclassify a fixed record directory')
        roots[name] = group
    if not set(experiments) <= set(roots):
        raise ValueError('completed experiment has no declared archive group')
    return relative, bindings, roots


def dependencies():
    manifest = OLD / 'archives/RECORDS-MANIFEST.json'
    if builder.file_sha(manifest) != OLD_MANIFEST_SHA:
        raise ValueError('published dependency manifest changed')
    old = builder.read_json(manifest)
    source = next(row for row in old['members']
                  if row['path'] == 'records/source-publication-v1/SOURCE-MANIFEST.json')
    if source['sha256'] != SOURCE_MANIFEST_SHA:
        raise ValueError('published production source manifest differs')
    return [{
        'kind': 'previously_published_external_api_records',
        'manifest': '../rwkvos-29b-20260909/archives/RECORDS-MANIFEST.json',
        'manifest_sha256': OLD_MANIFEST_SHA,
        'archives': old['archives'],
        'production_source_member': source,
        'production_source_distinct_from_research_peft': True,
        'restore_instructions': 'Use the old package safe verifier/extract. Restore records to the original experiment root or map paths explicitly; do not execute PEFT snapshot as production RAG code.',
        'historical_dependencies': 'Historical raw calls, API contracts and exact source snapshots referenced by followup fixtures are in this previous package; archives are not duplicated.',
        'external_api_weights_or_deployed_revision_verified': False,
    }, dict(published.INDEX_DEPENDENCY, kind='previously_published_5000_document_index',
            note='Full 5000-record corpus/index export dependency; this followup Reader diagnosis used fixed actual source fixtures and does not create a new index.')]


def prepare(records, target, freeze, freeze_sha, credential_source):
    freeze_relative, bindings, roots = check_release(records, freeze, freeze_sha)
    scanner = CredentialScan(credential_source)
    members, excluded, seen = [], [], set()

    def add(path, group):
        relative_path = path.relative_to(records)
        relative = relative_path.as_posix()
        if relative in seen:
            raise ValueError('duplicate selection: ' + relative)
        seen.add(relative)
        reason = exclusion(relative_path)
        if reason:
            exact = EXACT_SOURCE_EXCLUSIONS.get(relative)
            if exact:
                source = builder.checked_source(records, relative)
                if source.stat().st_size != exact['bytes'] or builder.file_sha(source) != exact['sha256']:
                    raise ValueError('exact excluded source bytes changed: ' + relative)
            excluded.append({'origin': 'records', 'path': relative, 'reason': reason})
            return
        source = builder.checked_source(records, relative)
        digest = builder.file_sha(source)
        binary = BINARY_EVIDENCE.get(relative)
        if binary and (digest != binary['sha256'] or source.stat().st_size != binary['bytes']):
            raise ValueError('diagnostic binary differs from exact allowlist: ' + relative)
        if relative != freeze_relative and bindings.get(relative) != digest:
            raise ValueError('selected file absent from final complete bindings: ' + relative)
        builder.scan_file(source, relative)
        scanner.scan_file(source, relative)
        members.append({'origin': 'records', 'source_path': relative, 'path': 'records/' + relative,
                        'group': group, 'bytes': source.stat().st_size, 'sha256': digest})

    for name, group in roots.items():
        root = records / name
        if not root.is_dir() or root.is_symlink():
            raise ValueError('completed directory missing or unsafe: ' + name)
        for folder, dirs, files in os.walk(root, followlinks=False):
            for entry in sorted(dirs):
                child = Path(folder) / entry
                relative = child.relative_to(records)
                if child.is_symlink():
                    raise ValueError('symbolic directory rejected: ' + relative.as_posix())
                reason = exclusion(relative)
                if reason:
                    dirs.remove(entry)
                    excluded.append({'origin': 'records', 'path': relative.as_posix(), 'reason': reason})
            for name in sorted(files):
                add(Path(folder) / name, group)
    for path in sorted(records.iterdir()):
        reason = exclusion(path.relative_to(records))
        if reason:
            excluded.append({'origin': 'records', 'path': path.name, 'reason': reason})
        elif path.is_file() or path.is_symlink():
            add(path, 'support')
        elif path.name not in roots:
            raise ValueError('unclassified directory requires final archive_roots binding: ' + path.name)
    if freeze_relative not in {row['source_path'] for row in members}:
        raise ValueError('final freeze was not selected')
    if not any(row['source_path'].startswith('reader-examples-v1/')
               and 'INTERPRETATION-NOTE' in row['source_path'] for row in members):
        raise ValueError('examples interpretation note must be retained')
    members.sort(key=lambda row: row['path'])
    builder.save_new(target, {
        'schema': 'bm250820-public-archive-selection-v1',
        'scope': 'Completed 2.9B followup records. Byte integrity only, not model quality certification.',
        'selection_script_sha256': builder.file_sha(Path(__file__)),
        'published_builder_sha256': BASE_SHA, 'published_credential_wrapper_sha256': WRAPPER_SHA,
        'final_freeze': {'path': 'records/' + freeze_relative, 'sha256': freeze_sha},
        'members': members, 'exclusions': sorted(excluded, key=lambda row: row['path']),
        'external_dependencies': dependencies(),
        'diagnostic_binary_evidence': BINARY_EVIDENCE,
        'exact_excluded_research_sources': EXACT_SOURCE_EXCLUSIONS,
        'research_snapshot_completeness': 'The public selection intentionally omits the two exact files listed above; the original full snapshot and manifest are unchanged. Executed PEFT/FLA modules are retained. This is not a complete PEFT working-tree copy.',
        'research_source_warning': 'peft-source-v1 is an independently pinned local state-tuning research snapshot, not the production RAG source or proof of API deployment identity.',
        'secret_scan': {'result': 'PASS', 'selected_files': len(members),
            'checks': ['pinned public pattern/JSON credential key/recursive base64 scan',
                       'signed URL rejection', 'actual CF credential values scanned in memory including encoded bodies'],
            'credential_values_or_hashes_stored': False, 'reviewed_false_positives': SCAN_EXCEPTIONS,
            'exception_scope': 'exact path + complete SHA + authorization prose JSON pointer only; all values remain scanned; raw bytes unchanged'},
    })
    print(json.dumps({'status': 'PREPARED', 'selection': str(target), 'sha256': builder.file_sha(target),
                      'members': len(members), 'bytes': sum(row['bytes'] for row in members)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'build', 'verify'))
    for name in ('records', 'selection', 'output', 'manifest', 'extract', 'final-freeze', 'credential-source'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--final-freeze-sha256')
    args = parser.parse_args()
    if args.command == 'verify':
        if not args.manifest:
            parser.error('--manifest is required')
        builder.verify(args.manifest, args.extract)
        return
    if not args.records or not args.selection:
        parser.error('--records and --selection are required')
    records = args.records.resolve()
    if args.command == 'prepare':
        if not args.final_freeze or not args.final_freeze_sha256 or not args.credential_source:
            parser.error('prepare requires final-freeze, exact SHA and credential-source')
        prepare(records, args.selection, args.final_freeze, args.final_freeze_sha256, args.credential_source)
    else:
        if not args.output:
            parser.error('--output is required')
        selection = builder.read_json(args.selection)
        if (selection.get('selection_script_sha256') != builder.file_sha(Path(__file__))
                or selection.get('published_builder_sha256') != BASE_SHA
                or selection.get('published_credential_wrapper_sha256') != WRAPPER_SHA):
            raise ValueError('selection does not pin the unchanged archive builders')
        final = selection['final_freeze']
        relative = final['path'].removeprefix('records/')
        check_release(records, records / relative, final['sha256'])
        builder.build(records, records, args.selection, args.output)


if __name__ == '__main__':
    main()
