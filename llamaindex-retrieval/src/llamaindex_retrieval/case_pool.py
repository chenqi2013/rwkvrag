"""Immutable case-keyed source pools with shared content-addressed documents."""
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile

from .schemas import SourceItem


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()


def _write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(dir=path.parent) as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
        try:
            os.link(f.name, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise ValueError('case pool conflicts with frozen observation') from None


class CasePool:
    def __init__(self, store):
        self.store = Path(store)

    def _case_path(self, case_id):
        if not isinstance(case_id, str) or re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', case_id) is None:
            raise ValueError('case_pool requires a safe explicit case_id')
        return self.store/'cases'/(case_id+'.json')

    def _blob(self, value):
        raw = encode(value)
        key = sha256(raw).hexdigest()
        _write(self.store/'blobs'/(key+'.json'), raw)
        return key

    def read_blob(self, key):
        if not isinstance(key, str) or re.fullmatch(r'[0-9a-f]{64}', key) is None:
            raise ValueError('invalid case pool blob ID')
        raw = (self.store/'blobs'/(key+'.json')).read_bytes()
        if sha256(raw).hexdigest() != key:
            raise ValueError('corrupt case pool blob')
        return json.loads(raw)

    def freeze(self, case_id, *, catalog, documents, web_payload, metadata=None):
        path = self._case_path(case_id)
        ids = [c['id'] for c in catalog]
        if len(ids) != len(set(ids)) or set(ids) != set(documents):
            raise ValueError('full catalog must bind every document exactly once')
        references = {}
        for fid, source in documents.items():
            item = SourceItem.model_validate(source)
            if item.document_id != fid or item.source != 'knowledge_base':
                raise ValueError('frozen KB file binding mismatch')
            references[fid] = self._blob(item.model_dump(mode='json'))
        value = {'case_id': case_id, 'catalog': catalog, 'documents': references,
                 'web_payload': self._blob(web_payload), 'metadata': metadata or {}}
        envelope = {'schema': 'case-pool-v1', 'sha256': sha256(encode(value)).hexdigest(), 'value': value}
        _write(path, encode(envelope))
        return self.load(case_id)

    def load(self, case_id):
        envelope = json.loads(self._case_path(case_id).read_bytes())
        value = envelope['value']
        if (envelope['schema'] != 'case-pool-v1' or value['case_id'] != case_id
                or sha256(encode(value)).hexdigest() != envelope['sha256']
                or {c['id'] for c in value['catalog']} != set(value['documents'])
                or len(value['catalog']) != len(value['documents'])):
            raise ValueError('corrupt case pool manifest')
        return FrozenCase(self, value, envelope['sha256'])


class FrozenCase:
    def __init__(self, store, value, digest):
        self.store, self.value, self.sha256 = store, value, digest
        self.catalog, self.metadata = value['catalog'], value['metadata']

    def document(self, file_id):
        item = SourceItem.model_validate(self.store.read_blob(self.value['documents'][file_id]))
        if item.document_id != file_id or item.source != 'knowledge_base':
            raise ValueError('corrupt frozen document binding')
        return item

    def web(self):
        return self.store.read_blob(self.value['web_payload'])
