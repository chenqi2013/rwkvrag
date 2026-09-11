"""One bounded full-length state backward on server GPU2, with no optimizer.

Uses the original frozen PEFT import closure and the portable train-only export.
The executable source, data, checkpoint and environment are pinned in config.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "llamaindex-retrieval/src/llamaindex_retrieval"))
from state_training import read_training_tokens, accumulation_groups, longest_training_row
from state_tokens import Vocabulary

GPU = "GPU-5d61943c-0955-e221-92a8-318915f5a3a0"
CHECKPOINT = "966f3420f833532aae3fb1fd6326533b08d43d23b7b03eaa2f0694a30b64a239"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def bound(path):
    result = (PROJECT / path).resolve()
    if not result.is_relative_to(PROJECT):
        raise ValueError("path outside project")
    return result


def host_guard():
    if socket.gethostname() != "rwkv-260304" or os.environ.get("CUDA_VISIBLE_DEVICES") != GPU:
        raise ValueError("only rwkv-8222 physical GPU2 is allowed")
    info = subprocess.run(["nvidia-smi", "-i", GPU, "--query-gpu=index,uuid,memory.used",
                           "--format=csv,noheader,nounits"], check=True, capture_output=True,
                          text=True, timeout=20).stdout.strip()
    fields = [x.strip() for x in info.split(",")]
    if fields[:2] != ["2", GPU]:
        raise ValueError("GPU index/UUID mismatch")
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                           "--format=csv,noheader"], check=True, capture_output=True,
                          text=True, timeout=20).stdout
    if any(line.split(",")[0].strip() == GPU for line in apps.splitlines()):
        raise ValueError("GPU2 already has a compute process; no automatic retry")
    return {"hostname": socket.gethostname(), "physical_index": 2, "uuid": GPU,
            "memory_used_mib_before": int(fields[2])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--config-sha256", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if sha(args.config) != args.config_sha256:
        raise ValueError("config SHA mismatch")
    config = json.loads(args.config.read_text())
    if config["limits"] != {"optimizer_updates": 0, "model_forward_calls": 1,
                            "backward_calls": 1, "max_sequence_tokens": 4096, "wall_seconds": 1800}:
        raise ValueError("preflight limits differ from fixed contract")
    gpu = host_guard()  # Before Torch import or any CUDA access.
    output = bound(args.output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    def expired(signum, frame):
        raise TimeoutError("1800-second preflight wall limit")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(1800)
    forwards = backwards = 0
    stage = "bindings"
    try:
        for name, expected in config["code_bindings"].items():
            if sha(bound(name)) != expected:
                raise ValueError("source SHA mismatch: " + name)
        versions = {name: importlib.metadata.version(name) for name in config["packages"]}
        if versions != config["packages"]:
            raise ValueError("runtime package versions changed")
        source = bound(config["peft_source"])
        manifest = json.loads((source / "SOURCE-MANIFEST.json").read_text())
        for name, pin in manifest["files"].items():
            if sha(source / name) != pin["sha256"]:
                raise ValueError("PEFT source changed: " + name)
        rows = read_training_tokens(bound(config["train_tokens"]), config["train_sha256"], config["train_count"])
        vocab = Vocabulary(source / "rwkv_vocab_v20230424.txt")
        for row in rows:
            raw = b"".join(vocab.by_id[i] for i in row["input_ids"][:row["prompt_tokens"]])
            if hashlib.sha256(raw).hexdigest() != row["prompt_sha256"]:
                raise ValueError("encoded prompt SHA mismatch")
        row = longest_training_row(rows)
        if row["id"] != config["longest_id"] or len(row["input_ids"]) != config["longest_tokens"]:
            raise ValueError("longest training example changed")
        groups = accumulation_groups(rows, seed=20260909)
        write(output / "TRAIN-ADAPTER.json", {
            "samples": len(rows), "groups": [[x["id"] for x in g] for g in groups],
            "loss_divisors": [len(g) for g in groups], "training_schedule_executed": False,
            "dev_or_heldout_content_read": False, "longest_id": row["id"],
            "longest_sequence_tokens": len(row["input_ids"]),
        })
        env = {"WKV": "fla", "RWKV_MY_TESTING": "x070", "RWKV_TRAIN_TYPE": "state",
               "FUSED_KERNEL": "0", "RWKV_HEAD_SIZE_A": "64", "RWKV_JIT_ON": "0",
               "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "WANDB_MODE": "disabled", "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1",
               "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TRITON_CACHE_DIR": str(output / "cache/triton"),
               "TORCH_EXTENSIONS_DIR": str(output / "cache/extensions"), "XDG_CACHE_HOME": str(output / "cache/xdg")}
        for name in ("TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "XDG_CACHE_HOME"):
            Path(env[name]).mkdir(parents=True, exist_ok=True)
        os.environ.update(env)
        write(output / "STARTED.json", {"config": config, "config_sha256": args.config_sha256,
                                       "gpu": gpu, "packages": versions, "environment": env})
        stage = "model_load"
        import torch
        torch.set_num_threads(8)
        torch.manual_seed(20260909)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
        checkpoint = bound(config["checkpoint"])
        assert sha(checkpoint) == CHECKPOINT and checkpoint.stat().st_size == 5896273469
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        assert len(weights) == 1062 and sum(v.numel() for v in weights.values()) == 2948065280
        assert all(torch.isfinite(v).all() for v in weights.values())
        sys.path.insert(0, str(source))
        from rwkvt.rwkv7.model import RWKV7
        cfg = SimpleNamespace(vocab_size=65536, n_layer=32, n_embd=2560, dim_att=2560,
                              dim_ffn=10240, head_size_a=64, head_size_divisor=8, ctx_len=4096,
                              grad_cp=1, peft="state", train_type="state", my_testing="x070", dropout=0)
        with torch.device("meta"):
            model = RWKV7(cfg)
        names = [f"blocks.{i}.att.time_state" for i in range(32)]
        expected = model.state_dict()
        assert set(expected) - set(weights) == set(names) and not set(weights) - set(expected)
        assert all(expected[k].shape == v.shape for k, v in weights.items())
        loaded = {**weights, **{n: torch.zeros((40, 64, 64), dtype=torch.bfloat16) for n in names}}
        model.load_state_dict(loaded, strict=True, assign=True)
        del loaded, expected
        model.requires_grad_(False)
        model = model.to(device="cuda:0", dtype=torch.bfloat16)
        for block in model.blocks:
            block.att.time_state = torch.nn.Parameter(block.att.time_state.detach().float())
        state = [(n, p) for n, p in model.named_parameters() if n in names]
        base = [(n, p) for n, p in model.named_parameters() if n not in names]
        assert len(state) == 32 and sum(p.numel() for _, p in state) == 5242880
        assert all(p.requires_grad and p.dtype == torch.float32 for _, p in state)
        assert all(not p.requires_grad and p.grad is None for _, p in base)
        base_versions = {n: p._version for n, p in base}
        model.train()
        torch.cuda.reset_peak_memory_stats()
        stage = "forward_backward"
        ids = torch.tensor([row["input_ids"][:-1]], dtype=torch.long, device="cuda:0")
        labels = torch.tensor([row["labels"][1:]], dtype=torch.long, device="cuda:0")
        started = time.monotonic()
        forwards += 1
        logits = model(ids)
        mask = labels != -100
        loss = torch.nn.functional.cross_entropy(logits[mask].float(), labels[mask])
        assert torch.isfinite(loss)
        backwards += 1
        loss.backward()
        torch.cuda.synchronize()
        gradients = [{"name": n, "finite": bool(torch.isfinite(p.grad).all()) if p.grad is not None else False,
                      "nonzero": int(torch.count_nonzero(p.grad)) if p.grad is not None else 0,
                      "l2": float(p.grad.norm()) if p.grad is not None else None} for n, p in state]
        assert all(g["finite"] and g["nonzero"] > 0 for g in gradients)
        assert all(p.grad is None and not p.requires_grad and p._version == base_versions[n] for n, p in base)
        assert all(torch.count_nonzero(p) == 0 for _, p in state)
        probe = {"id": row["id"], "sequence_tokens": len(row["input_ids"]),
                 "forward_tokens": ids.shape[1], "supervised_tokens": int(mask.sum()),
                 "target_only_loss": float(loss.detach()), "gradients": gradients,
                 "seconds": time.monotonic() - started,
                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        write(output / "GRADIENTS.json", probe)
        stage = "base_integrity"
        equal = {n: torch.equal(p.detach().cpu(), weights[n]) for n, p in base}
        assert len(equal) == 1062 and all(equal.values())
        assert sha(checkpoint) == CHECKPOINT
        for name, expected in config["code_bindings"].items():
            assert sha(bound(name)) == expected
        write(output / "BASE-INTEGRITY.json", {"per_tensor_equal": equal, "all1062_equal": True,
                                                "base_gradients_present": 0, "state_unchanged_zero": True})
        write(output / "COMPLETED.json", {"status": "PREFLIGHT_COMPLETE", "optimizer_updates": 0,
              "model_forward_calls": forwards, "backward_calls": backwards,
              "gradient_checkpointing_internal_recomputation": True,
              "train_count": len(rows), "longest_id": row["id"], "longest_sequence_tokens": len(row["input_ids"]),
              "state_layers_with_finite_nonzero_gradients": 32, "all_base_weights_equal": True,
              "state_unchanged_zero": True, "dev_or_heldout_content_read": False,
              "seconds": time.monotonic() - began, "gpu_uuid": GPU, "probe": probe,
              "quality_evaluated": False, "production_promoted": False})
    except BaseException as error:
        write(output / "FAILED.json", {"stage": stage, "error_type": type(error).__name__,
              "message": str(error), "model_forward_attempts": forwards, "backward_attempts": backwards,
              "optimizer_updates": 0, "seconds": time.monotonic() - began})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
