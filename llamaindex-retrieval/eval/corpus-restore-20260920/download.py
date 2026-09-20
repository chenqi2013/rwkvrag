"""Restore immutable upstream FineWiki shards; never invent corpus from gold answers."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import time
import urllib.request

REVISION = '8bd13e72e6a002407649b3e898535f42ceb1aeb9'
ROOT = Path('data/corpora/finewiki-restored-20260920')
TREE = f'https://huggingface.co/api/datasets/HuggingFaceFW/finewiki/tree/{REVISION}/data/zhwiki?recursive=false&expand=false'


def download(item):
    target = ROOT / 'shards' / Path(item['path']).name
    expected = item['lfs']['oid']
    if target.exists():
        assert sha256(target.read_bytes()).hexdigest() == expected
        return
    temporary = target.with_suffix('.parquet.part')
    url = f'https://huggingface.co/datasets/HuggingFaceFW/finewiki/resolve/{REVISION}/{item["path"]}?download=true'
    for attempt in range(1, 5):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            request = urllib.request.Request(url + f'&attempt={attempt}&time={int(time.time())}',
                headers={'Range': f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset:
                    assert response.status == 206
                    assert response.headers['Content-Range'].startswith(f'bytes {offset}-')
                with temporary.open('ab' if offset else 'wb') as output:
                    while data := response.read(4 * 1024 * 1024):
                        output.write(data)
            assert temporary.stat().st_size == item['size']
            digest = sha256()
            with temporary.open('rb') as source:
                while data := source.read(4 * 1024 * 1024):
                    digest.update(data)
            assert digest.hexdigest() == expected
            temporary.rename(target)
            print(json.dumps({'downloaded': item['path'], 'bytes': item['size'], 'sha256': expected}), flush=True)
            return
        except Exception as error:
            print(json.dumps({'path': item['path'], 'attempt': attempt, 'error': str(error)}), flush=True)
            if attempt == 4:
                raise
            time.sleep(2)


def main():
    (ROOT / 'shards').mkdir(parents=True, exist_ok=True)
    inventory = ROOT / 'UPSTREAM-TREE.json'
    if inventory.exists():
        raw = inventory.read_bytes()
    else:
        for attempt in range(5):
            try:
                with urllib.request.urlopen(TREE, timeout=30) as response:
                    raw = response.read()
                break
            except Exception as error:
                print(json.dumps({'catalog_attempt': attempt + 1, 'error': str(error)}), flush=True)
                if attempt == 4:
                    raise
                time.sleep(2)
        inventory.write_bytes(raw)
    items = [item for item in json.loads(raw) if item['path'].endswith('.parquet')]
    assert len(items) == 5
    print(json.dumps({'revision': REVISION, 'shards': len(items), 'bytes': sum(i['size'] for i in items)}), flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, items))


if __name__ == '__main__':
    main()
