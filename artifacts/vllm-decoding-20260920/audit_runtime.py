"""Read-only deployment audit; run with the isolated remote environment."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess

ROOT = Path('/home/chase/rwkvrag/data/services/vllm-decode-20260920')
ENGINE = ROOT / 'engine'
UNIT = 'rwkvrag-vllm-decode-20260920-attempt3.service'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    groups = [('engine', ROOT/'ENGINE-SOURCE.json', ENGINE),
              ('dependencies', ROOT/'DEPENDENCIES.json', ENGINE/'.venv/lib/python3.12/site-packages')]
    record = {'checked_at': datetime.now(timezone.utc).isoformat(), 'groups': {}}
    for name, manifest, base in groups:
        files = json.loads(manifest.read_text())['files']
        mismatches = [p for p, expected in files.items() if digest(base/p) != expected]
        record['groups'][name] = {'files': len(files), 'mismatches': mismatches,
                                  'manifest_sha256': digest(manifest)}
        assert not mismatches, mismatches
    record['service'] = subprocess.check_output(
        ['systemctl', '--user', 'show', UNIT, '-p', 'ActiveState', '-p', 'MainPID',
         '-p', 'ActiveEnterTimestamp', '-p', 'RuntimeMaxUSec'], text=True)
    pid = int(subprocess.check_output(['systemctl', '--user', 'show', UNIT,
                                      '-p', 'MainPID', '--value'], text=True))
    assert pid > 0 and 'ActiveState=active' in record['service']
    pids = [pid]
    for parent in pids:
        children = Path(f'/proc/{parent}/task/{parent}/children')
        if children.exists():
            pids.extend(int(v) for v in children.read_text().split())
    record['processes'] = {}
    libraries = set()
    for current in pids:
        maps = Path(f'/proc/{current}/maps')
        if not maps.exists():
            continue
        matches = sorted({line.split()[-1] for line in maps.read_text().splitlines()
                          if 'flashrwkv2' in line and '.so' in line})
        record['processes'][str(current)] = {'flashrwkv2_libraries': matches}
        libraries.update(matches)
    assert libraries, 'No loaded FlashRWKV2 shared library found in service processes'
    record['compiled_libraries'] = {p: digest(Path(p)) for p in sorted(libraries)}
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
