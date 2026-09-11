"""Start or stop only this project's GPU3 service; run on rwkv-8222."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

ROOT = Path('/home/chase/rwkvrag')
SERVER = ROOT/'llamaindex-retrieval/deploy/local/serve_gpu3.py'
DIRECTORY = ROOT/'data/services/rwkv-gpu3'
PYTHON = '/home/chase/chase/RWKV-PEFT/.venv/bin/python'

def process_identity(pid):
    proc = Path(f'/proc/{pid}')
    argv = proc.joinpath('cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    start_ticks = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]
    return [os.fsdecode(arg) for arg in argv], start_ticks

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'stop'])
    args = parser.parse_args()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    # Serialize launcher invocations, without taking ownership of any GPU workload.
    import fcntl
    with (DIRECTORY/'launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        identity_file = DIRECTORY/'launcher.json'
        saved = json.loads(identity_file.read_text()) if identity_file.exists() else None
        pid = saved['pid'] if saved else None
        if pid and Path(f'/proc/{pid}').exists():
            argv, start_ticks = process_identity(pid)
            expected = [PYTHON, '-u', str(SERVER), '--config', str(DIRECTORY/'settings.json'),
                        '--sha256', saved['config_sha256']]
            if argv != expected or start_ticks != saved['start_ticks']:
                raise RuntimeError('Saved PID identity changed; leaving it untouched')
            if args.action == 'stop':
                os.kill(pid, signal.SIGTERM)
                print(f'Stop requested for owned service PID {pid}')
            else:
                print(f'Owned service already running: PID {pid}; check /health for readiness')
            return
        if args.action == 'stop':
            print('Owned service is not running')
            return
        config = DIRECTORY/'settings.json'
        digest = hashlib.sha256(config.read_bytes()).hexdigest()
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': 'GPU-a9570da2-547a-c2b3-0cab-7bbdc1a8a8b0'}
        with (DIRECTORY/'console.log').open('ab') as log:
            process = subprocess.Popen([PYTHON, '-u', str(SERVER), '--config', str(config),
                '--sha256', digest], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=log, start_new_session=True)
        _, start_ticks = process_identity(process.pid)
        identity_file.write_text(json.dumps({'pid':process.pid, 'start_ticks':start_ticks,
            'config_sha256':digest})+'\n')
        print(f'Started PID {process.pid}; model loading must complete before /health is ready')

if __name__ == '__main__':
    main()
