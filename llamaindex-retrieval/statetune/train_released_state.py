"""Bounded train-only consumer of a SHA-pinned, independently reviewed release.

The v5 reproduction scripts remain frozen. This entry accepts the released row
count, validates on CPU, and requires a matching zero-update GPU preflight before
training. It never opens development or heldout examples or labels.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import signal
import time

from preflight_state import PROJECT, bound, host_guard, sha, write
from pilot_runtime import load_model, check_base, save_state
from state_training import accumulation_groups, longest_training_row, read_training_tokens
from state_tokens import Vocabulary

PACKAGES = {"torch": "2.9.0", "triton": "3.5.0", "rwkv-fla": "0.7.202508221413",
            "deepspeed": "0.18.1", "einops": "0.8.1", "numpy": "2.3.4"}
PEFT_MANIFEST_SHA = "ccdbb113427dad96e90d7cdef285f80eacb27dd02dedb63517fc7c903fc916ca"
REQUIRED_CODE = ["statetune/train_released_state.py", "statetune/preflight_state.py",
                 "statetune/pilot_runtime.py", "src/llamaindex_retrieval/state_training.py",
                 "src/llamaindex_retrieval/state_tokens.py"]


def validate(config):
    """No Torch import, GPU access, or evaluation-content reads."""
    if (config.get("schema") != "rwkv_released_training_v1"
            or config.get("mode") not in {"preflight", "train"}
            or config.get("packages") != PACKAGES
            or type(config.get("wall_seconds")) is not int
            or not 60 <= config["wall_seconds"] <= 3600):
        raise ValueError("unsupported execution contract")
    deadline = datetime.fromisoformat(config["deadline_utc"])
    if deadline.tzinfo is None:
        raise ValueError("deadline requires timezone")
    settings = config["training"]
    if (set(settings) != {"epochs", "learning_rate", "accumulation", "seed"}
            or type(settings["epochs"]) is not int or not 1 <= settings["epochs"] <= 6
            or settings["learning_rate"] not in {1e-4, 3e-5, 1e-5}
            or type(settings["accumulation"]) is not int or settings["accumulation"] != 2
            or type(settings["seed"]) is not int or settings["seed"] != 20260909):
        raise ValueError("training budget exceeds fixed contract")
    required = {"llamaindex-retrieval/" + name for name in REQUIRED_CODE}
    source = bound(config["runtime"]["peft_source"])
    required.add(str((source / "SOURCE-MANIFEST.json").relative_to(PROJECT)))
    if not required <= set(config["bindings"]):
        raise ValueError("missing runtime code bindings")
    for name, expected in config["bindings"].items():
        if sha(bound(name)) != expected:
            raise ValueError("bound source changed: " + name)
    if sha(source / "SOURCE-MANIFEST.json") != PEFT_MANIFEST_SHA:
        raise ValueError("complete frozen PEFT manifest required")
    manifest = json.loads((source / "SOURCE-MANIFEST.json").read_text())
    for name, pin in manifest["files"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or sha(path) != pin["sha256"]:
            raise ValueError("PEFT source changed")
    release_path = bound(config["release"])
    if sha(release_path) != config["release_sha256"]:
        raise ValueError("release receipt changed")
    release = json.loads(release_path.read_text())
    if (release.get("schema") != "rwkv_reader_release_v1"
            or release.get("training_data_admitted") is not True
            or release.get("input_layout") != "task_last"
            or release.get("prompt_protocol") != "rwkv_g1j_no_think_v1"
            or release.get("max_sequence_tokens") != 4096
            or release.get("max_generation_tokens") != 32):
        raise ValueError("release was not admitted under this Reader protocol")
    count = release["export"]["exported"]
    if type(count) is not int or not 2 <= count <= 256:
        raise ValueError("released sample count outside bounded range")
    token_path = release_path.parent / "train.tokens.jsonl"
    expected = release["files"]["train.tokens.jsonl"]
    if release["export"]["sha256"] != expected:
        raise ValueError("inconsistent released token hash")
    rows = read_training_tokens(token_path, expected, count)
    vocab_path = source / "rwkv_vocab_v20230424.txt"
    if sha(vocab_path) != release["vocabulary_sha256"]:
        raise ValueError("released vocabulary changed")
    vocab = Vocabulary(vocab_path)
    for row in rows:
        raw = b"".join(vocab.by_id[i] for i in row["input_ids"][:row["prompt_tokens"]])
        text = raw.decode("utf-8")
        suffix = "\n\nAssistant: <think></think>\n"
        lines = text.removesuffix(suffix).split("\n")
        if (hashlib.sha256(raw).hexdigest() != row["prompt_sha256"]
                or not text.endswith(suffix) or len(lines) != 6
                or any(not line.startswith(prefix) for line, prefix in zip(
                    lines, ["User: ", "来源：", "原文父级上下文：", "原文：", "任务：", "子问题："]))
                or row["prompt_tokens"] + 31 > 4096):
            raise ValueError("released prompt hash/layout/generation budget mismatch")
    longest = longest_training_row(rows)
    identity = {"release_sha256": config["release_sha256"], "train_sha256": expected,
                "train_count": count, "runtime": config["runtime"],
                "bindings": config["bindings"], "packages": config["packages"],
                "longest_id": longest["id"], "longest_tokens": len(longest["input_ids"])}
    if config["mode"] == "train":
        pin = config["preflight"]
        path = bound(pin["path"])
        if sha(path) != pin["sha256"]:
            raise ValueError("preflight receipt changed")
        receipt = json.loads(path.read_text())
        if (receipt.get("status") != "PREFLIGHT_COMPLETE" or receipt.get("identity") != identity
                or receipt.get("optimizer_updates") != 0 or receipt.get("backward_calls") != 1
                or receipt.get("base_tensors_unchanged") != 1062):
            raise ValueError("matching zero-update preflight required")
    return rows, identity


def loss_for(torch, model, row):
    ids = torch.tensor([row["input_ids"][:-1]], device="cuda:0")
    labels = torch.tensor([row["labels"][1:]], device="cuda:0")
    logits = model(ids)
    mask = labels != -100
    loss = torch.nn.functional.cross_entropy(logits[mask].float(), labels[mask])
    if not torch.isfinite(loss):
        raise ValueError("nonfinite target/EOS loss")
    return loss


def execute(torch, model, state, rows, config, output):
    parameters = [p for _, p in state]
    model.train()
    if config["mode"] == "preflight":
        loss = loss_for(torch, model, longest_training_row(rows))
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError("missing/nonfinite state gradient")
        norm = float(sum(p.grad.double().square().sum() for p in parameters).sqrt())
        if norm <= 0 or any(torch.count_nonzero(p).item() for p in parameters):
            raise ValueError("zero gradient or preflight state changed")
        model.zero_grad(set_to_none=True)
        return {"status": "PREFLIGHT_COMPLETE", "optimizer_updates": 0, "backward_calls": 1,
                "loss": float(loss.detach()), "gradient_l2": norm}
    settings = config["training"]
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"],
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    checkpoints = [save_state(torch, state, output, 0)]
    updates = backwards = 0
    for epoch in range(1, settings["epochs"] + 1):
        losses = []
        for group in accumulation_groups(rows, seed=settings["seed"] + epoch - 1, size=2):
            optimizer.zero_grad(set_to_none=True)
            values = []
            for row in group:
                loss = loss_for(torch, model, row)
                (loss / len(group)).backward()
                values.append(float(loss.detach()))
                backwards += 1
                del loss
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError("missing/nonfinite state gradient")
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
            if norm <= 0:
                raise ValueError("zero gradient")
            record = {"step": updates + 1, "epoch": epoch, "ids": [r["id"] for r in group],
                      "sample_losses": values, "loss_divisor": len(group),
                      "gradient_l2_before_clip": norm}
            write(output / f"step-{updates + 1:03d}-INTENT.json", record)
            optimizer.step()
            torch.cuda.synchronize()
            if not all(torch.isfinite(p).all() for p in parameters):
                raise ValueError("nonfinite state")
            write(output / f"step-{updates + 1:03d}-DONE.json", record)
            updates += 1
            losses.extend(values)
        checkpoints.append(save_state(torch, state, output, updates))
        print(json.dumps({"epoch": epoch, "updates": updates,
                          "online_mean_train_loss": sum(losses) / len(rows)}), flush=True)
    optimizer.zero_grad(set_to_none=True)
    if updates != settings["epochs"] * ((len(rows) + 1) // 2) or backwards != settings["epochs"] * len(rows):
        raise ValueError("epoch coverage mismatch")
    return {"status": "TRAIN_COMPLETE", "optimizer_updates": updates,
            "backward_calls": backwards, "checkpoints": checkpoints}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if sha(args.config) != args.config_sha256:
        raise ValueError("configuration changed")
    config = json.loads(args.config.read_text())
    rows, identity = validate(config)
    if args.validate_only:
        print(json.dumps({"status": "CPU_VALIDATED", "identity": identity,
                          "model_calls": 0, "optimizer_updates": 0}, indent=2))
        return
    if not args.output:
        parser.error("--output is required for GPU execution")
    remaining = lambda: (datetime.fromisoformat(config["deadline_utc"]) - datetime.now(timezone.utc)).total_seconds()
    if remaining() <= 60:
        raise ValueError("work window exhausted")
    gpu = host_guard()  # Before Torch import or CUDA access.
    if remaining() <= 60:
        raise ValueError("work window exhausted during GPU check")
    wall = min(config["wall_seconds"], int(remaining()) - 15)
    output = bound(args.output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    def expired(*unused):
        raise TimeoutError("released training deadline")
    def interrupted(*unused):
        raise InterruptedError("released training received SIGTERM")
    previous_alarm = signal.signal(signal.SIGALRM, expired)
    previous_term = signal.signal(signal.SIGTERM, interrupted)
    signal.alarm(wall)
    try:
        versions = {name: importlib.metadata.version(name) for name in PACKAGES}
        if versions != PACKAGES:
            raise ValueError("runtime package versions changed")
        write(output / "STARTED.json", {"config": config, "config_sha256": args.config_sha256,
                                        "identity": identity, "gpu": gpu, "effective_wall_seconds": wall})
        torch, model, state, base, weights, base_versions = load_model(config["runtime"], output)
        result = execute(torch, model, state, rows, config, output)
        equal = check_base(torch, base, weights, base_versions)
        write(output / "BASE-INTEGRITY.json", equal)
        write(output / "COMPLETED.json", {**result, "identity": identity,
              "config_sha256": args.config_sha256, "base_tensors_unchanged": len(equal),
              "seconds": time.monotonic() - began, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "dev_or_heldout_content_read": False, "production_promoted": False})
    except BaseException as error:
        write(output / "FAILED.json", {"error": repr(error), "seconds": time.monotonic() - began,
              "confirmed_updates": len(list(output.glob("step-*-DONE.json"))),
              "attempted_updates": len(list(output.glob("step-*-INTENT.json")))})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        signal.signal(signal.SIGTERM, previous_term)


if __name__ == "__main__":
    main()
