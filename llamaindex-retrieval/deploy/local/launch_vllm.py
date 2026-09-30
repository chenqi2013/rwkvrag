"""Start a pinned RWKV vLLM service with FP32 recurrent State and FP16 I/O."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    engine = Path(config["engine"])
    for relative, expected in config["source_pins"].items():
        actual = hashlib.sha256((engine / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Engine source changed: {relative}")
    for relative, expected in config["model_pins"].items():
        with (Path(config["model_path"]) / relative).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Model artifact changed: {relative}")
    gpu = config["gpu_uuid"]
    info = subprocess.check_output([
        "nvidia-smi", "-i", gpu, "--query-gpu=uuid,memory.free",
        "--format=csv,noheader,nounits"], text=True).strip().split(",")
    if info[0].strip() != gpu or int(info[1]) < config["minimum_free_mib"]:
        raise ValueError("GPU identity or available memory does not match deployment")
    # This engine selects the actual FP32IO16 operator from the State dtype.
    cmd = [str(engine / ".venv/bin/python"), "-m", "vllm.entrypoints.openai.api_server",
           "--model", config["model_path"], "--served-model-name", config["model"],
           "--host", "127.0.0.1", "--port", str(config["port"]),
           "--dtype", "float16", "--mamba-ssm-cache-dtype", "float32",
           "--max-model-len", "16384", "--max-num-seqs", "4",
           "--max-num-batched-tokens", "2048", "--gpu-memory-utilization", "0.30",
           "--enable-chunked-prefill", "--enable-prefix-caching", "--async-scheduling",
           "--generation-config", "vllm"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": str(engine),
           "VLLM_RWKV_STATE_CACHE_MAX_BYTES": "268435456",
           "SETUPTOOLS_SCM_PRETEND_VERSION": config["build_version"]}
    print(json.dumps({"command": cmd, "gpu_uuid": gpu,
                      "recurrent_state_dtype": "float32", "operator_mode": "fp32io16",
                      "verified_source_files": len(config["source_pins"]),
                      "verified_model_files": len(config["model_pins"])}), flush=True)
    os.chdir(engine)
    os.execve(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
