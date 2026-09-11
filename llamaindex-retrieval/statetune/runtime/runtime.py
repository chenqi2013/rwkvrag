"""Pinned runtime adapter. Model creation is reachable only after server_guard."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
import time

from contracts import (
    CHECKPOINT_SHA, MAX_GENERATION, STATE_KEYS, checked_file, now,
    output_record, server_guard, sha_file, sync_directory, validate_state_metadata,
)


def open_runtime(config):
    device_receipt = server_guard(config)
    runtime_root = Path(config["runtime_dir"]).resolve()
    manifest_path = runtime_root / "RUNTIME-SOURCE.json"
    manifest = json.loads(checked_file(manifest_path, config["runtime_manifest_sha256"]).read_text())
    if manifest.get("installed_version") != "0.8.30" or manifest.get("compile_cuda_extension") is not False:
        raise ValueError("Only the pinned pureTorch rwkv0.8.30 runtime is allowed")
    expected_names = {"__init__.py", "model.py", "rwkv_tokenizer.py", "utils.py", "rwkv_vocab_v20230424.txt"}
    if set(manifest["files"]) != expected_names:
        raise ValueError("Unexpected runtime manifest entries")
    for name, binding in manifest["files"].items():
        checked_file(runtime_root / "rwkv" / name, binding["sha256"])
    vocab = checked_file(runtime_root / "rwkv" / "rwkv_vocab_v20230424.txt", config["vocab_sha256"])
    checkpoint = checked_file(config["checkpoint_path"], CHECKPOINT_SHA)
    if any(name == "rwkv" or name.startswith("rwkv.") for name in sys.modules):
        raise RuntimeError("An unverified rwkv module was already imported")
    os.environ.update({"RWKV_V7_ON": "1", "RWKV_CUDA_ON": "0", "RWKV_JIT_ON": "1",
                       "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                       "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    sys.path.insert(0, str(runtime_root))
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    module = importlib.import_module("rwkv.model")
    if Path(module.__file__).resolve() != runtime_root / "rwkv" / "model.py":
        raise ValueError("Unexpected imported runtime path")
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    if torch.cuda.device_count() != 1:
        raise ValueError("Exactly one visible CUDA device is required")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    begin = time.monotonic()
    if checkpoint.suffix != ".pth":
        raise ValueError("Pinned RWKV7 loader requires a .pth checkpoint")
    # RWKV_x070 always appends .pth; its constructor has no verbose parameter.
    model = module.RWKV(model=str(checkpoint)[:-4], strategy="cuda bf16")
    if (model.n_layer, model.n_embd, model.n_head, model.head_size) != (32, 2560, 40, 64):
        raise ValueError("Unexpected model dimensions")
    if tuple(model.z["head.weight"].shape) != (2560, 65536):
        raise ValueError("Unexpected output vocabulary")
    if any(value.dtype != torch.bfloat16 for value in model.z.values()):
        raise ValueError("Base model tensors must all be BF16")
    torch.cuda.synchronize()
    tokenizer = TRIE_TOKENIZER(str(vocab))
    receipt = {"started_at": now(), "device": device_receipt,
               "runtime_manifest_sha256": config["runtime_manifest_sha256"],
               "vocab_sha256": config["vocab_sha256"], "checkpoint_sha256": CHECKPOINT_SHA,
               "model_load_seconds": time.monotonic() - begin,
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "rwkv": "0.8.30", "strategy": "cuda bf16", "tf32": False,
               "custom_cuda_extension": False, "initial_wkv_axes": "H,V,K",
               "state_transposed": False, "sample_state_reset": True,
               "torch_num_threads": torch.get_num_threads(),
               "torch_num_interop_threads": torch.get_num_interop_threads()}
    return torch, model, tokenizer, receipt


def state_from_spec(torch, spec):
    if spec.get("step") not in [0, 8, 16]:
        raise ValueError("Only frozen steps 0, 8, 16 are allowed")
    path = checked_file(spec["path"], spec["sha256"])
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or set(loaded) != STATE_KEYS:
        raise ValueError("State file must directly map the 32 canonical time_state keys")
    metadata = {}
    for key, value in loaded.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError("State entries must be tensors")
        metadata[key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                         "finite": bool(torch.isfinite(value).all().item())}
    validate_state_metadata(metadata)
    if spec["step"] == 0 and any(torch.count_nonzero(value).item() for value in loaded.values()):
        raise ValueError("Step0 must be exactly zero")
    return {key: value.detach().clone() for key, value in loaded.items()}, metadata


def initial_state(torch, canonical=None):
    result = []
    for layer in range(32):
        key = f"blocks.{layer}.att.time_state"
        wkv = (torch.zeros((40, 64, 64), dtype=torch.float32, device="cuda:0")
               if canonical is None else canonical[key].to(device="cuda:0", dtype=torch.float32).clone())
        result.extend([torch.zeros(2560, dtype=torch.bfloat16, device="cuda:0"), wkv,
                       torch.zeros(2560, dtype=torch.bfloat16, device="cuda:0")])
    return result


def clone_state(state):
    return [value.detach().clone() for value in state]


def tensor_metadata(torch, tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "finite": bool(torch.isfinite(tensor).all().item())}


def tensor_difference(torch, reference, candidate):
    left, right = reference.detach().float(), candidate.detach().float()
    if left.shape != right.shape or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("Cannot compare mismatched or nonfinite tensors")
    delta = right - left
    left_norm, delta_norm = torch.linalg.vector_norm(left), torch.linalg.vector_norm(delta)
    return {"equal": bool(torch.equal(reference, candidate)),
            "max_absolute": float(delta.abs().max().item()),
            "l2": float(delta_norm.item()), "reference_l2": float(left_norm.item()),
            "relative_l2": float((delta_norm / left_norm).item()) if left_norm.item() else None}


def save_tensors(torch, path, tensors):
    path = Path(path)
    cpu = {key: value.detach().to("cpu").contiguous() for key, value in tensors.items()}
    with path.open("xb") as handle:
        torch.save(cpu, handle)
        handle.flush()
        os.fsync(handle.fileno())
    sync_directory(path.parent)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha_file(path)}


def greedy_from_prefill(torch, model, tokenizer, logits, state, decision_sink=None):
    """Generate on caller-owned state. Actual EOS0 consumes one of 128 steps."""
    ids, decisions = [], []
    started = time.monotonic()
    for step in range(MAX_GENERATION):
        if logits.numel() != 65536 or not torch.isfinite(logits).all():
            raise ValueError("Nonfinite or wrong-shape generation logits")
        token = int(torch.argmax(logits).item())
        top_values, top_ids = torch.topk(logits.float(), k=2)
        decision = {"step": step + 1, "token": token,
                    "top2_ids": top_ids.tolist(), "top2_logits": top_values.tolist(),
                    "top_margin": float((top_values[0] - top_values[1]).item())}
        decisions.append(decision)
        ids.append(token)
        if decision_sink is not None:
            decision_sink(decision)
        if token == 0:
            break
        if step + 1 < MAX_GENERATION:
            logits, state = model.forward([token], state)
    torch.cuda.synchronize()
    result = output_record(ids, tokenizer.idx2token, ids[-1] == 0)
    result.update({"decisions": decisions, "generation_seconds": time.monotonic() - started,
                   "sampling": "argmax; tie follows torch.argmax first index; no penalties"})
    return result
