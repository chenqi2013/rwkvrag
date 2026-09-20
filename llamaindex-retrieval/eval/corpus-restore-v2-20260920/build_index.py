"""Build an isolated complete-article test index and history database."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from llama_index.core import Document
from opensearchpy import OpenSearch, helpers
from pymongo import MongoClient

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.ingest import ingest_documents
from llamaindex_retrieval.lexical_index import LexicalIndex

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / 'data/corpora/finewiki-restored-20260920'
DEPLOYMENT = ROOT / 'data/services/restored-regression-20260920-v2'
INDEX = 'rwkvrag-restored-regression-20260920-v2'
DATABASE = 'rwkvrag_restored_regression_20260920_v2'
KB = 'wiki-restored-20260920'


def main():
    DEPLOYMENT.mkdir(exist_ok=True)
    original_path = ROOT / 'data/services/local-app/settings.json'
    original = json.loads(original_path.read_text())
    config = {**original, 'opensearch_index': INDEX, 'mongo_database': DATABASE,
        'upload_dir': str(CORPUS)}
    settings_path = DEPLOYMENT / 'settings.json'
    if settings_path.exists():
        assert json.loads(settings_path.read_text()) == config
    else:
        settings_path.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    settings_path.chmod(0o600)
    settings = Settings(**config)
    search = OpenSearch(settings.opensearch_url, timeout=300)
    mongo = MongoClient(settings.mongo_url)
    assert not search.indices.exists(index=INDEX) and not search.indices.exists(index=INDEX + '-active')
    assert DATABASE not in mongo.list_database_names()
    manifests = []
    for directory in (CORPUS / 'selected-v2', CORPUS / 'wikipedia-supplement'):
        for row in json.loads((directory / 'manifest.json').read_text()):
            path = directory / row['text_path']
            raw = path.read_bytes()
            assert sha256(raw).hexdigest() == row['text_sha256']
            manifests.append((row, path, raw.decode('utf-8')))
    assert len({row['document_id'] for row, _, _ in manifests}) == len(manifests)
    canonical_files = {}
    for row, path, text in manifests:
        canonical_files.setdefault(row['text_sha256'], row['document_id'])
    files = []
    now = datetime.now(timezone.utc)
    for row, path, text in manifests:
        if canonical_files[row['text_sha256']] != row['document_id']:
            continue
        files.append({'id': row['document_id'], 'knowledge_base_id': KB,
            'filename': row['title'] + '.md', 'path': str(path), 'content_type': 'text/markdown',
            'extension': '.md', 'size': path.stat().st_size, 'sha256': row['text_sha256'],
            'source': 'finewiki-zh' if 'legacy_document_id' in row else 'wikipedia-snapshot',
            'status': 'ready', 'node_count': 0, 'error': None, 'last_job_id': None,
            'created_at': now, 'updated_at': now, 'origin': row})
    mongo[DATABASE].knowledge_bases.insert_one({'id': KB, 'name': '完整原文回归语料 20260920',
        'description': 'Frozen complete upstream pages and retained versions; isolated regression data.',
        'created_at': now, 'updated_at': now})
    mongo[DATABASE].files.insert_many(files)

    def documents():
        for row, path, text in manifests:
            source = 'finewiki-zh' if 'legacy_document_id' in row else 'wikipedia-snapshot'
            meta = {'document_id': row['document_id'], 'file_id': canonical_files[row['text_sha256']],
                'knowledge_base_id': KB, 'source': source, 'title': row['title'],
                'uri': row.get('revision_url', row['url']), 'kind': 'finewiki',
                'external_id': row['id'], 'snapshot_id': row['snapshot_id'],
                'legacy_document_id': row.get('legacy_document_id'),
                'page_id': row['page_id'], 'version': row['version'],
                'source_text_sha256': row['text_sha256'], 'source_text_encoding': 'utf-8',
                'file': str(path), 'in_language': 'zh', 'wikiname': 'zhwiki'}
            yield Document(id_=row['document_id'], text=text, metadata=meta)

    lexical = LexicalIndex(settings)
    stats = ingest_documents(settings, documents(), 32, False, lexical_index=lexical,
        progress_callback=lambda d, n: print(json.dumps({'documents': d, 'nodes': n}), flush=True)
            if d % 512 == 0 else None)
    search.indices.refresh(index=lexical.index_name)
    digest = sha256()
    nodes = 0
    counts = {}
    # Every stored span is checked against its complete source snapshot.
    originals = {row['document_id']: (row, text) for row, _, text in manifests}
    for hit in helpers.scan(search, index=lexical.index_name, query={'query': {'match_all': {}}},
            size=500, scroll='5m'):
        node = hit['_source']
        row, text = originals[node['document_id']]
        span = node['metadata']['source_span']
        assert node['text'] == text[span['start']:span['end']]
        assert sha256(node['text'].encode()).hexdigest() == span['sha256']
        assert node['metadata']['source_text_sha256'] == row['text_sha256']
        counts[node['document_id']] = counts.get(node['document_id'], 0) + 1
        digest.update(json.dumps(node, ensure_ascii=False, sort_keys=True).encode())
        nodes += 1
    assert nodes == stats['nodes'] and len(counts) == len(manifests)
    file_counts = {}
    for key, count in counts.items():
        fid = canonical_files[originals[key][0]['text_sha256']]
        file_counts[fid] = file_counts.get(fid, 0) + count
    for key, count in file_counts.items():
        mongo[DATABASE].files.update_one({'id': key}, {'$set': {'node_count': count}})
    report = {'index': INDEX, 'alias': lexical.index_name, 'knowledge_base_id': KB,
        'database': DATABASE, 'documents': len(manifests), 'physical_files': len(files), 'nodes': nodes,
        'all_source_spans_verified': True, 'scan_order_content_sha256': digest.hexdigest(),
        'settings_sha256': sha256(settings_path.read_bytes()).hexdigest(),
        'production_settings_sha256': sha256(original_path.read_bytes()).hexdigest(),
        'changed_configuration_fields': sorted(k for k in config if config[k] != original.get(k)),
        'corpus_manifest_sha256': {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
            for p in (CORPUS / 'selected-v2/manifest.json', CORPUS / 'wikipedia-supplement/manifest.json')},
        'no_model_or_prompt_changes': True, 'production_index_unchanged_by_build': True}
    (DEPLOYMENT / 'BUILD.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    search.close()
    mongo.close()


if __name__ == '__main__':
    main()
