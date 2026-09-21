"""Guarded isolated launch of the user's exact engine snapshot on authorized GPU3."""
from datetime import datetime, timezone
from hashlib import file_digest
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

BASE = Path('/home/chase/rwkvrag/data/services/vllm-decode-20260920')
ROOT = BASE.parent / "layered-repeatability-20260921"
ENGINE = BASE / 'engine'
MODEL = Path('/mnt/nas-model/g1j/hf/rwkv7-g1j-7.2b-20260831-ctx16384')
GPU = 'GPU-a9570da2-547a-c2b3-0cab-7bbdc1a8a8b0'


def sha(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def main():
    assert socket.gethostname() == 'rwkv-82'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU
    assert os.environ.get('VLLM_RWKV_STATE_CACHE_MAX_BYTES') == '268435456'
    assert Path('/etc/machine-id').read_text().strip() == 'bcd164d5ad3a4ab3b0790412e32e69f3'
    gpu = subprocess.check_output(['nvidia-smi', '-i', GPU,
        '--query-gpu=index,uuid,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True).strip()
    index, uuid, used, total = [v.strip() for v in gpu.split(',')]
    assert index == '3' and uuid == GPU and int(total) - int(used) > 32000
    binding = json.loads((BASE / 'ENGINE-SOURCE.json').read_text())
    assert sha(BASE / 'ENGINE-SOURCE.json') == 'b8be05cd8dddcdb9eac88bca6af0ff2486982edd9407328fb6b064fd0c416168'
    for name, expected in binding['files'].items():
        assert sha(ENGINE / name) == expected, name
    for name, expected in binding['model_metadata'].items():
        assert sha(MODEL / name) == expected, name
    hashes = {}
    for expected, name in re.findall(r'([a-f0-9]{64})  (model-\S+\.safetensors)', (MODEL/'PROVENANCE.md').read_text()):
        actual = sha(MODEL / name)
        assert actual == expected, name
        hashes[name] = actual
        print(json.dumps({'verified_model_shard': name}), flush=True)
    assert len(hashes) == 6
    sys.path.insert(0, str(ENGINE))
    import torch
    import vllm
    import flashrwkv2
    from vllm.version import is_reduced_rwkv_build
    assert Path(vllm.__file__).resolve() == ENGINE / 'vllm/__init__.py'
    assert torch.cuda.device_count() == 1 and is_reduced_rwkv_build()
    assert flashrwkv2.__version__ == '0.1.0a13'
    assert torch.__version__ == '2.13.0+cu130'
    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', str(MODEL), '--served-model-name', 'rwkvrag-g1j72-mainline',
        '--host', '127.0.0.1', '--port', '18426', '--dtype', 'float16',
        '--mamba-ssm-cache-dtype', 'float32', '--max-model-len', '16384',
        '--max-num-seqs', '4', '--max-num-batched-tokens', '2048',
        '--gpu-memory-utilization', '0.30', '--enable-chunked-prefill',
        '--enable-prefix-caching', '--async-scheduling', '--generation-config', 'vllm']
    record = {'verified_at': datetime.now(timezone.utc).isoformat(), 'gpu': gpu,
        'engine_source_manifest_sha256': sha(BASE/'ENGINE-SOURCE.json'),
        'model_shards_sha256': hashes, 'vllm_file': vllm.__file__,
        'torch': torch.__version__, 'flashrwkv2': flashrwkv2.__version__,
        'flashrwkv2_file': flashrwkv2.__file__, 'command': command,
        'zero_state_at_startup': True, 'state_cache_max_bytes': 268435456,
        'local_engine_working_tree_snapshot': True}
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/'VERIFIED-DEPLOYMENT.json').open('x') as stream:
        json.dump(record, stream, indent=2)
    os.chdir(ENGINE)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
