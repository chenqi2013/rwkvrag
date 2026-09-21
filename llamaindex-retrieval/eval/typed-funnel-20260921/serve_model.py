import importlib.util
import time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'data/experiments/typed-funnel-20260921/service1'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('launch', ROOT / 'llamaindex-retrieval/eval/model-size-paired-20260921/run_v2.py')
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    launch.OUT = OUT
    proc = None
    try:
        proc = launch.start('7.2b', 1)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError('engine exited')
            try:
                if httpx.get('http://127.0.0.1:18426/health', timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(2)
        else:
            raise TimeoutError('startup')
        launch.save(OUT / 'READY.json', {'model': '7.2b', 'state': 'zero', 'port': 18426})
        proc.wait(timeout=7200)
    finally:
        launch.stop(proc)


if __name__ == '__main__':
    main()
