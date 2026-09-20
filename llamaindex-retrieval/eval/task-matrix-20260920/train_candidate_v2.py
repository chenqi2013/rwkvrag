"""Bounded experimental StateTune using the existing frozen PEFT/FLA runtime.

Author-reviewed synthetic data; not an independently reviewed production release.
Only physical GPU3, target/EOS loss, immutable base, validation-selected states.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

ROOT = Path("/home/chase/rwkvrag")
GPU = "GPU-a9570da2-547a-c2b3-0cab-7bbdc1a8a8b0"
sys.path.insert(0, str(ROOT / "llamaindex-retrieval/statetune"))
from pilot_runtime import load_model, check_base
from preflight_state import sha
from train_released_state import loss_for


def write(path, value):
    with path.open("x") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())


def load_rows(path, pin, vocab):
    if sha(path) != pin["sha256"]:
        raise ValueError("data hash changed")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != pin["rows"]:
        raise ValueError("row count changed")
    for row in rows:
        n, ids, labels = row["prompt_tokens"], row["input_ids"], row["labels"]
        if not 0 < n < len(ids) <= 4096 or ids[-1] != 0 or labels != [-100] * n + ids[n:]:
            raise ValueError("invalid target-only labels")
        prompt = b"".join(vocab.by_id[i] for i in ids[:n])
        target = b"".join(vocab.by_id[i] for i in ids[n:-1])
        if (hashlib.sha256(prompt).hexdigest() != row["prompt_sha256"] or
                hashlib.sha256(target).hexdigest() != row["target_sha256"] or
                prompt.decode() != row["prompt"] or target.decode() != row["target"]):
            raise ValueError("prompt/target byte binding failed")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if sha(args.config) != args.sha256:
        raise ValueError("configuration hash mismatch")
    config = json.loads(args.config.read_text())
    if socket.gethostname() != "rwkv-82" or os.environ.get("CUDA_VISIBLE_DEVICES") != GPU:
        raise ValueError("only authorized GPU3 on rwkv-8222")
    if Path("/etc/machine-id").read_text().strip() != "bcd164d5ad3a4ab3b0790412e32e69f3":
        raise ValueError("host identity changed")
    memory = subprocess.check_output(["nvidia-smi", "-i", GPU, "--query-gpu=index,uuid,memory.free",
        "--format=csv,noheader,nounits"], text=True).strip().split(",")
    if [s.strip() for s in memory[:2]] != ["3", GPU] or int(memory[2]) < 45000:
        raise ValueError("GPU identity changed or insufficient spare memory; no service is stopped")
    if config["stage_epochs"] != {"writer": 2, "planner": 4} or config["learning_rate"] != 5e-5 or config["accumulation"] != 4:
        raise ValueError("unsupported experimental budget")
    if not 60 <= config["wall_seconds"] <= 5400:
        raise ValueError("invalid deadline")
    for name, expected in config["bindings"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("runtime source changed: " + name)
    source = ROOT / config["runtime"]["peft_source"]
    for name, pin in json.loads((source / "SOURCE-MANIFEST.json").read_text())["files"].items():
        if sha(source / name) != pin["sha256"]:
            raise ValueError("PEFT import closure changed")
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    write(output / "START.json", {"config": config, "config_sha256": args.sha256,
        "gpu": GPU, "pid": os.getpid(), "independent_review": False, "production_promotion": False})
    data = ROOT / config["data"]
    manifest = json.loads((data / "MANIFEST.json").read_text())
    if sha(data / "MANIFEST.json") != config["manifest_sha256"]:
        raise ValueError("manifest changed")
    from preflight_state import Vocabulary
    vocab = Vocabulary(source / "rwkv_vocab_v20230424.txt")
    # Held-out files are intentionally not opened by the training process.
    rows = {stage: {split: load_rows(data / f"{split}.{stage}.jsonl",
        manifest["files"][f"{split}.{stage}.jsonl"], vocab) for split in ("train", "validation")}
        for stage in ("writer", "planner")}
    started = time.monotonic()
    torch, model, state, base, weights, versions = load_model(config["runtime"], output)
    torch.cuda.set_per_process_memory_fraction(0.60)
    parameters = [p for _, p in state]
    results = {}

    def validate(stage_rows):
        model.eval()
        with torch.no_grad():
            values = [float(loss_for(torch, model, row)) for row in stage_rows]
        model.train()
        return sum(values) / len(values)

    try:
        for stage in ("writer", "planner"):
            pin = config["initial_states"][stage]
            initial = None
            if pin:
                if sha(ROOT / pin["path"]) != pin["sha256"]:
                    raise ValueError("initial state changed")
                initial = torch.load(ROOT / pin["path"], map_location="cpu", weights_only=True)
            with torch.no_grad():
                for name, parameter in state:
                    parameter.copy_(initial[name] if initial else torch.zeros_like(parameter))
            model.zero_grad(set_to_none=True)
            longest = max(rows[stage]["train"], key=lambda r: len(r["input_ids"]))
            before = [p.detach().clone() for p in parameters]
            loss = loss_for(torch, model, longest)
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError("preflight gradient invalid")
            if not all(torch.equal(p, b) for p, b in zip(parameters, before, strict=True)):
                raise ValueError("preflight mutated state")
            model.zero_grad(set_to_none=True)
            del loss, before
            baseline = validate(rows[stage]["validation"])
            write(output / f"{stage}-PREFLIGHT.json", {"optimizer_updates": 0,
                "longest_id": longest["id"], "validation_loss": baseline})
            best = baseline
            result = {"baseline_validation_loss": baseline, "epochs": [], "selected": None}
            optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0)
            for epoch in range(1, config["stage_epochs"][stage] + 1):
                samples = list(rows[stage]["train"])
                random.Random(20260920 + epoch).shuffle(samples)
                for offset in range(0, len(samples), 4):
                    if time.monotonic() - started > config["wall_seconds"]:
                        raise TimeoutError("training wall budget reached")
                    group = samples[offset:offset + 4]
                    optimizer.zero_grad(set_to_none=True)
                    losses = []
                    for row in group:
                        loss = loss_for(torch, model, row)
                        losses.append(float(loss.detach()))
                        (loss / len(group)).backward()
                        del loss
                    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
                    if not all(p.grad is not None for p in parameters) or not all(p.grad is None for _, p in base):
                        raise ValueError("gradient ownership violated")
                    optimizer.step()
                    step = offset // 4 + 1
                    receipt = {"stage": stage, "epoch": epoch, "step": step,
                        "ids": [r["id"] for r in group], "loss": losses, "gradient_l2": norm,
                        "elapsed_seconds": round(time.monotonic() - started, 1)}
                    write(output / f"{stage}-e{epoch}-s{step:03}.json", receipt)
                    if step % 8 == 0:
                        print(json.dumps(receipt), flush=True)
                optimizer.zero_grad(set_to_none=True)
                val = validate(rows[stage]["validation"])
                path = output / f"{stage}-epoch-{epoch}.pth"
                tensors = {n: p.detach().float().cpu().contiguous().clone() for n, p in state}
                if not all(torch.isfinite(t).all() for t in tensors.values()):
                    raise ValueError("nonfinite trained state")
                with path.open("xb") as file:
                    torch.save(tensors, file)
                pin = {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "validation_loss": val}
                result["epochs"].append(pin)
                if val < best:
                    best, result["selected"] = val, pin
                write(output / f"{stage}-epoch-{epoch}.json", pin)
                print(json.dumps({"stage": stage, "epoch": epoch, **pin}), flush=True)
            del optimizer
            results[stage] = result
            write(output / f"{stage}-RESULT.json", result)
        check_base(torch, base, weights, versions)
        write(output / "RESULT.json", {"stages": results, "base_unchanged": True,
            "heldout_read": False, "generation_quality_verified": False,
            "elapsed_seconds": time.monotonic() - started,
            "peak_memory_bytes": torch.cuda.max_memory_allocated()})
    except BaseException as error:
        write(output / "FAILED.json", {"type": type(error).__name__, "error": str(error), "completed_stages": results})
        raise


if __name__ == "__main__":
    main()
