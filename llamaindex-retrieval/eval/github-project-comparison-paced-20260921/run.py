"""Reuse frozen questions and runner, changing only endpoint and output version."""
import hashlib
import json
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for name, digest in json.loads((HERE / "PINS.json").read_text()).items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
source = ROOT / "llamaindex-retrieval/eval/github-project-comparison-20260921/run.py"
code = source.read_text().replace('http://127.0.0.1:18448', 'http://127.0.0.1:18450').replace(
    'data/quality-runs/github-project-comparison-20260921/run1',
    'data/quality-runs/github-project-comparison-paced-20260921/run1')
exec(compile(code, str(source), "exec"), {"__file__":str(source), "__name__":"__main__"})
