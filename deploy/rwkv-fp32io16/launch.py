"""Explicit fp32io16 launch gate; defaults to check, never silently falls back."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess


def command(profile):
    if (profile["provider"], profile["kernel_mode"], profile["recurrent_state_dtype"], profile["weights_io_dtype"]) != (
            "FlashRWKV2", "fp32io16", "float32", "float16"):
        raise ValueError("This launch profile requires FlashRWKV2 fp32io16, FP32 recurrent State and FP16 IO")
    return [str(Path(profile["engine"]) / ".venv/bin/python"), "-m", "vllm.entrypoints.openai.api_server",
        "--model", profile["model"], "--served-model-name", "paired-eval", "--host", "127.0.0.1",
        "--port", str(profile["port"]), "--dtype", profile["weights_io_dtype"],
        "--mamba-ssm-cache-dtype", profile["recurrent_state_dtype"],
        "--max-model-len", "16384", "--max-num-seqs", "4", "--max-num-batched-tokens", "2048",
        "--gpu-memory-utilization", "0.30", "--enable-chunked-prefill", "--enable-prefix-caching",
        "--async-scheduling", "--generation-config", "vllm"]


def preflight(profile):
    args = command(profile)
    if Path("/etc/machine-id").read_text().strip() != profile["machine_id"]:
        raise RuntimeError("Not the authorized model host")
    hashes = {}
    for relative, expected in profile["engine_files"].items():
        actual = hashlib.sha256((Path(profile["engine"]) / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Engine binding changed: {relative}; re-audit before launching")
        hashes[relative] = actual
    probe = "import flashrwkv2,json; print(json.dumps({'version':flashrwkv2.__version__,'path':flashrwkv2.__file__,'apis':[n for n in dir(flashrwkv2) if n in " + repr([profile["kernel_api"], profile["state_api"]]) + "]}))"
    provider = json.loads(subprocess.check_output([args[0], "-c", probe], text=True, cwd=profile["engine"]))
    if provider["version"] != profile["provider_version"] or set(provider["apis"]) != {profile["kernel_api"], profile["state_api"]}:
        raise RuntimeError("Required fp32io16 provider/API unavailable; no FP16 fallback")
    return {"protocol": profile["protocol"], "command": args, "provider": provider,
        "engine_files": hashes, "kernel_mode": profile["kernel_mode"],
        "weights_io_dtype": profile["weights_io_dtype"], "recurrent_state_dtype": profile["recurrent_state_dtype"],
        "gpu_uuid": profile["gpu_uuid"], "preflight_passed": True,
        "model_started": False, "runtime_kernel_executed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["check", "start"], default="check")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(Path(__file__).with_name("PROFILE.json").read_text())
    report = preflight(profile)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2)
    if args.mode == "check":
        print(json.dumps(report))
        return
    row = subprocess.check_output(["nvidia-smi", "-i", profile["gpu_uuid"],
        "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True)
    index, uuid, used, total = [part.strip() for part in row.split(",")]
    if index != "3" or uuid != profile["gpu_uuid"] or int(total) - int(used) < 32000:
        raise RuntimeError("Authorized GPU3 capacity check failed")
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", profile["port"])) == 0:
            raise RuntimeError("Experiment port already in use")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": profile["gpu_uuid"], "PYTHONPATH": profile["engine"],
           "VLLM_RWKV_STATE_CACHE_MAX_BYTES": "268435456"}
    os.chdir(profile["engine"])
    os.execvpe(report["command"][0], report["command"], env)


if __name__ == "__main__":
    main()
