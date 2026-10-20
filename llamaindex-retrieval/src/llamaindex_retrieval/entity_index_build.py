"""Build an immutable model-authored index from question-independent JSONL inputs."""
import argparse
import asyncio
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic

from . import entity_index


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def read_inputs(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = set()
    for row in rows:
        if (not isinstance(row.get('id'), str) or not row['id'] or row['id'] in ids
                or not isinstance(row.get('title'), str) or not isinstance(row.get('opening'), str)
                or not isinstance(row.get('headings', []), list)
                or any(not isinstance(h, str) for h in row.get('headings', []))):
            raise ValueError('index inputs require unique IDs, titles, openings and string headings')
        ids.add(row['id'])
    if not rows:
        raise ValueError('empty entity index input')
    return rows


async def build(model, inputs, out, *, concurrency, profile_bindings=None):
    """No retries, edits or substituted terms. Failed files remain in the index."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out/'records').mkdir()
    (out/'PLAN.md').write_text('Model-authored entities and aliases from supplied document metadata. '
        'One generation per file; no question/gold input, lexical scoring, retry or raw repair. '
        'Exact structural-normalized equality lookup in original input order. '
        'Completion and valid format do not establish retrieval quality.\n')
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError('invalid index concurrency')
    write(out/'INPUTS.json', inputs)
    write(out/'BINDINGS.json', {'factory_sha256': sha(entity_index.__file__),
        'builder_sha256': sha(__file__), 'inputs_sha256': sha(out/'INPUTS.json'),
        'concurrency': concurrency, 'normalization': entity_index.normalization_manifest(),
        'profile': profile_bindings})
    start = monotonic()
    sem = asyncio.Semaphore(concurrency)
    halted = asyncio.Event()
    completed = []

    async def one(index, row):
        async with sem:
            if halted.is_set():
                raise RuntimeError('model authentication/quota/rate failure; stop without retry')
            record = await entity_index.build_record(model, row['id'], row['title'], row['opening'],
                                                    row.get('headings', []))
            if 'source_sha256' in row:
                record['source_sha256'] = row['source_sha256']
            # Ordinals keep caller-controlled IDs out of filesystem paths.
            path = out/'records'/f'{index:06d}.json'
            write(path, record)
            completed.append(path.name)
            codes = [h.get('http_status') for h in (record.get('trace') or {}).get('http', [])]
            if any(code in {401, 402, 403, 429} for code in codes):
                halted.set()
                raise RuntimeError('model authentication/quota/rate failure; stop without retry')
            return record

    tasks = [asyncio.create_task(one(i, row)) for i, row in enumerate(inputs)]
    try:
        records = await asyncio.gather(*tasks)
    except BaseException as error:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        write(out/'INTERRUPTED.json', {'complete': False, 'error_type': type(error).__name__,
                                     'completed_records': sorted(completed)})
        raise
    with (out/'ENTITIES.jsonl').open('x') as stream:
        for i, record in enumerate(records):
            row = {key: record[key] for key in ('id', 'title', 'status', 'raw_text', 'parsed')}
            row['record_sha256'] = sha(out/'records'/f'{i:06d}.json')
            if 'source_sha256' in record:
                row['source_sha256'] = record['source_sha256']
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
    index = entity_index.EntityIndex(records)
    write(out/'LOOKUP.json', {'normalization': entity_index.normalization_manifest(), 'exact': index.exact})
    summary = {'complete': True, 'files': len(records), 'elapsed_seconds': monotonic()-start,
               'status_counts': dict(Counter(r['status'] for r in records)),
               'valid_files': sum(r['status'] == 'completed' and entity_index.valid_terms(r.get('parsed'), 12)
                                  for r in records), 'exact_keys': len(index.exact),
               'entities_sha256': sha(out/'ENTITIES.jsonl'), 'semantic_quality': 'not_measured',
               'api_cost': None}
    write(out/'SCORED.json', summary)
    return summary


async def execute(config_path, inputs_path, out):
    from .interactive_app import ObservedModel, Trial
    config = json.loads(Path(config_path).read_text())
    inputs = read_inputs(inputs_path)
    # Trial construction reads configuration only. Building never acquires sources.
    config['records_dir'] = str(Path(out).parent/(Path(out).name+'-unused-records'))
    trial = Trial(config)
    try:
        await trial.native.verify_model_identity()
        if trial.settings.native_require_fp32_runtime:
            await trial.native.verify_fp32_runtime()
        summary = await build(ObservedModel(trial.native, {}), inputs, out,
            concurrency=trial.settings.native_max_concurrency,
            profile_bindings={'config_sha256': sha(config_path),
              'original_inputs_sha256': sha(inputs_path), 'model': trial.settings.native_model,
              'base_url': trial.settings.native_base_url,
              'require_fp32_runtime': trial.settings.native_require_fp32_runtime})
        return summary
    finally:
        await trial.native.aclose()
        await trial.admin.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(execute(args.config, args.inputs, args.out))


if __name__ == '__main__':
    main()
