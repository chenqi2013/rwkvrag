"""Serialize real-retrieval service after the frozen ablation releases GPU 3."""
import importlib.util
import json
import subprocess
import time
from pathlib import Path
import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "data/experiments/github-natural-comparison-20260921/service1"

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    prior = ROOT / "data/experiments/writer-convergence-20260921/run2/SUMMARY.json"
    deadline = time.monotonic() + 12000
    while True:
        active = subprocess.run(["systemctl","--user","is-active","rwkvrag-writer-convergence-v2-20260921.service"],capture_output=True,text=True).stdout.strip()
        if prior.exists() and active != "active": break
        if time.monotonic() > deadline: raise TimeoutError("waiting for controlled ablation")
        time.sleep(10)
    summary = json.loads(prior.read_text())
    assert summary["recorded"] == 752 and summary["error"] is None, "incomplete ablation"
    loader = importlib.util.spec_from_file_location("launch", HERE.parent / "model-size-paired-20260921/run_v2.py")
    launch = importlib.util.module_from_spec(loader); loader.loader.exec_module(launch)
    launch.OUT = OUT
    proc = None
    try:
        proc = launch.start("7.2b", 1)
        deadline = time.monotonic() + 600
        while True:
            if proc.poll() is not None: raise RuntimeError("engine exited")
            try:
                r = httpx.get("http://127.0.0.1:18426/health", timeout=2)
                if r.status_code == 200: break
            except httpx.HTTPError: pass
            if time.monotonic() > deadline: raise TimeoutError("startup")
            time.sleep(2)
        launch.save(OUT / "READY.json", {"prior_completed":752,"model":"7.2b","state":"zero","port":18426})
        print("Real comparison service ready", flush=True)
        proc.wait(timeout=10800)
    finally:
        launch.stop(proc)

if __name__ == "__main__":main()
