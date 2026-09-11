"""Pinned PEFT-FLA runtime for the bounded v3 training and evaluation pilot."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from preflight_state import CHECKPOINT, bound, sha, write


def load_model(config, output):
    # Host/GPU and source bindings must be checked by caller before Torch import.
    source = bound(config["peft_source"])
    env = {"WKV": "fla", "RWKV_MY_TESTING": "x070", "RWKV_TRAIN_TYPE": "state",
           "FUSED_KERNEL": "0", "RWKV_HEAD_SIZE_A": "64", "RWKV_JIT_ON": "0",
           "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "WANDB_MODE": "disabled", "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1",
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TRITON_CACHE_DIR": str(output / "cache/triton"),
           "TORCH_EXTENSIONS_DIR": str(output / "cache/extensions"), "XDG_CACHE_HOME": str(output / "cache/xdg")}
    for name in ("TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "XDG_CACHE_HOME"):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    os.environ.update(env)
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

    return torch, model, state, base, weights, base_versions


def check_base(torch, base, weights, versions):
    equal = {n: bool(not p.requires_grad and p.grad is None
                     and p._version == versions[n]
                     and torch.equal(p.detach().cpu(), weights[n])) for n, p in base}
    if len(equal) != 1062 or not all(equal.values()):
        raise ValueError("frozen base weights or gradient status changed")
    return equal


def save_state(torch, state, output, step):
    tensors = {n: p.detach().cpu().contiguous().clone() for n, p in state}
    if len(tensors) != 32 or not all(t.shape == (40, 64, 64) and t.dtype == torch.float32
                                    and torch.isfinite(t).all() for t in tensors.values()):
        raise ValueError("invalid canonical state")
    path = output / f"state-step-{step:02d}.pth"
    with path.open("xb") as stream:
        torch.save(tensors, stream)
        stream.flush()
        os.fsync(stream.fileno())
    restored = torch.load(path, map_location="cpu", weights_only=True)
    if set(restored) != set(tensors) or not all(torch.equal(t, restored[n]) for n,t in tensors.items()):
        raise ValueError("state readback mismatch")
    result = {"step": step, "path": path.name, "sha256": sha(path),
              "axes": ["head", "value", "key"], "dtype": "float32", "layers":32,
              "state_l2": float(sum(t.double().square().sum() for t in tensors.values()).sqrt())}
    write(output / f"state-step-{step:02d}.json", result)
    return result
