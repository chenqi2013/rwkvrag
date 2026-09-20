"""Select complete upstream pages for all old retrieval questions, plus the 5k background."""
from hashlib import sha256
import json
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from opencc import OpenCC

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'data/corpora/finewiki-restored-20260920'
REVISION = '8bd13e72e6a002407649b3e898535f42ceb1aeb9'


def digest(value):
    return sha256(value.encode()).hexdigest()


def canonical(row):
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def identity(row):
    return row['id'] + '@' + str(row['version']) + '@' + digest(canonical(row))[:16]


def main():
    cases_path = REPO / 'llamaindex-retrieval/eval/broad-qa-20260920/cases.json'
    cases = [c for c in json.loads(cases_path.read_text()) if c['endpoint'] == '/v1/ask']
    wanted = set()
    for case in cases:
        c = case['original_case']
        if c.get('expected_title'):
            wanted.add(c['expected_title'])
        wanted.update(c.get('expected_titles', []))
        wanted.update(c.get('expected_titles_all', []))
        for fact in c.get('oracle', {}).get('required_facts', []):
            for group in fact['evidence_sets']:
                wanted.update(e['title'] for e in group)
    converters = [OpenCC('t2s'), OpenCC('s2t')]
    lookup = wanted | {converter.convert(title) for converter in converters for title in wanted}
    target_values = pa.array(sorted(lookup))
    selected = {}
    provenance = {}
    waiting = {f'000_{i:05d}.parquet' for i in range(5)}
    while waiting:
        available = [ROOT / 'shards' / name for name in sorted(waiting) if (ROOT / 'shards' / name).exists()]
        if not available:
            time.sleep(10)
            continue
        for path in available:
            offset = 0
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=1024):
                flags = pc.is_in(batch.column(batch.schema.get_field_index('title')), value_set=target_values)
                positions = [i for i, flag in enumerate(flags.to_pylist()) if flag]
                for local_offset, row in zip(positions, batch.filter(flags).to_pylist()):
                    key = identity(row)
                    if key in selected:
                        assert row == selected[key]
                        provenance[key].setdefault('identical_duplicate_rows', []).append(
                            {'source_path': 'data/zhwiki/' + path.name, 'source_row': offset + local_offset})
                        continue
                    selected[key] = row
                    provenance[key] = {'source_path': 'data/zhwiki/' + path.name,
                        'source_row': offset + local_offset, 'dataset_revision': REVISION}
                offset += batch.num_rows
            waiting.remove(path.name)
            print(json.dumps({'scanned': path.name, 'rows': offset, 'target_pages_found': len(selected)}), flush=True)
    background_path = REPO / 'data/corpora/finewiki-zh-5000/original-rows.jsonl'
    background = {identity(r): r for r in (json.loads(line) for line in background_path.read_text().splitlines())}
    assert len(background) == 5000
    for key in background.keys() & selected.keys():
        assert background[key] == selected[key]
    union = {**background, **selected}
    destination = ROOT / 'selected-v2'
    destination.mkdir(exist_ok=False)
    (destination / 'texts').mkdir()
    manifests = []
    with (destination / 'original-rows.jsonl').open('x') as output:
        for key, row in sorted(union.items()):
            raw = canonical(row)
            output.write(raw + '\n')
            text_path = 'texts/' + str(row['page_id']) + '-' + str(row['version']) + '-' + digest(raw)[:16] + '.md'
            (destination / text_path).write_bytes(row['text'].encode('utf-8'))
            legacy_id = digest('finewiki-zh\0' + row['id'] + '\0' + row['title'])[:24]
            document_id = digest('finewiki-restored\0' + key)[:24]
            manifests.append({'id': row['id'], 'snapshot_id': key, 'document_id': document_id,
                'legacy_document_id': legacy_id, 'title': row['title'],
                'page_id': row['page_id'], 'version': row['version'], 'url': row['url'],
                'revision_url': f'https://zh.wikipedia.org/w/index.php?oldid={row["version"]}',
                'text_path': text_path, 'text_sha256': digest(row['text']),
                'text_characters': len(row['text']), 'original_row_sha256': digest(raw),
                'background': key in background, 'target': key in selected,
                'upstream': provenance.get(key), 'license': 'CC-BY-SA-4.0',
                'attribution': 'Wikipedia contributors; HuggingFaceFW/finewiki'})
    (destination / 'manifest.json').write_text(json.dumps(manifests, ensure_ascii=False, indent=2) + '\n')
    by_title = {}
    for row in manifests:
        by_title.setdefault(row['title'], []).append(row)
    alignment = []
    for case in cases:
        old = case['original_case']
        titles = [old['expected_title']] if old.get('expected_title') else old.get('expected_titles', [])
        titles = sorted(set(titles) | set(old.get('expected_titles_all', [])))
        if not titles:
            titles = sorted({e['title'] for f in old.get('oracle', {}).get('required_facts', [])
                for group in f['evidence_sets'] for e in group})
        variants = set(titles) | {converter.convert(t) for converter in converters for t in titles}
        found = [r for t in sorted(variants) for r in by_title.get(t, [])]
        reference = old.get('reference')
        alignment.append({'case_id': case['id'], 'expected_titles': titles,
            'found_titles': sorted({r['title'] for r in found}),
            'actual_document_ids': [r['document_id'] for r in found],
            'original_document_id': old.get('expected_document_id'),
            'original_document_id_matches_legacy_identity': old.get('expected_document_id') in [r['legacy_document_id'] for r in found]
                if old.get('expected_document_id') else None,
            'reference_exact_substring': any(reference in union[r['snapshot_id']]['text'] for r in found) if reference else None,
            'full_article_restored': bool(found)})
    summary = {'revision': REVISION, 'original_cases_sha256': sha256(cases_path.read_bytes()).hexdigest(),
        'background_original_rows_sha256': sha256(background_path.read_bytes()).hexdigest(),
        'pages': len(union), 'background_pages': len(background), 'target_pages': len(selected),
        'unique_article_ids': len({r['id'] for r in union.values()}),
        'version_policy': 'Keep every matching upstream version/row; collapse only byte-equivalent canonical rows and record all positions. Never choose a version because its answer matches gold.',
        'requested_titles': len(wanted), 'missing_titles': sorted(wanted - set(by_title)),
        'cases': alignment,
        'boundary': 'New selected full-article corpus with 5000 fixed distractors; not a byte-identical restoration of the deleted original complete index. Missing articles and nonliteral reference comparisons remain visible.'}
    (destination / 'COVERAGE.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'cases'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
