"""One fixed train-only pass: 21 examples, 11 AdamW updates, no automatic resume."""
import argparse
import importlib.metadata
import json
import signal
import time
from pathlib import Path

from preflight_state import bound, host_guard, sha, write
from state_training import accumulation_groups, read_training_tokens
from state_tokens import Vocabulary
from pilot_runtime import load_model, check_base, save_state


def check_config(config):
    expected = {"optimizer": "AdamW", "learning_rate": 0.001, "betas": [0.9, 0.999],
                "epsilon": 1e-8, "weight_decay": 0, "gradient_clip_l2": 1.0,
                "seed": 20260909, "accumulation": 2, "updates": 11, "examples": 21,
                "epochs": 1, "max_tokens": 4096, "wall_seconds": 1800, "saved_steps": [0, 11]}
    if config["training"] != expected:
        raise ValueError("training contract differs from fixed pilot")
    for path, digest in config["code_bindings"].items():
        if sha(bound(path)) != digest:
            raise ValueError("source SHA mismatch: " + path)
    if {n: importlib.metadata.version(n) for n in config["packages"]} != config["packages"]:
        raise ValueError("runtime package versions changed")
    for key in ("preflight", "preflight_started", "semantic_review"):
        pin = config[key]
        if sha(bound(pin["path"])) != pin["sha256"]:
            raise ValueError(key + " receipt hash mismatch")
        receipt = json.loads(bound(pin["path"]).read_text())
        if key == "preflight":
            if receipt["status"] != "PREFLIGHT_COMPLETE" or receipt["optimizer_updates"] != 0:
                raise ValueError("preflight incomplete")
        elif key == "preflight_started":
            prior = receipt["config"]
            for field in ("train_sha256", "train_count", "checkpoint", "peft_source", "packages",
                          "dataset_frozen_manifest_sha256"):
                if prior[field] != config[field]:
                    raise ValueError("preflight no longer matches training: " + field)
            if any(config["code_bindings"].get(p) != h for p,h in prior["code_bindings"].items()):
                raise ValueError("preflight implementation bindings changed")
        else:
            # Only a metadata certificate may enter the GPU process. The complete
            # review contains dev/heldout answers and stays on the local review side.
            fields = {"independent", "status", "reviewed_count", "dataset_frozen_sha256",
                      "review_report_sha256", "schema"}
            if (set(receipt) != fields or receipt["schema"] != "reader-semantic-certificate-v1"
                    or receipt["independent"] is not True or receipt["status"] != "PASS"
                    or receipt["reviewed_count"] != 43
                    or receipt["dataset_frozen_sha256"] != config["dataset_frozen_manifest_sha256"]):
                raise ValueError("review certificate invalid or contains extra content")


def train_groups(torch, model, state, base, rows, config, output):
    """Sequential backward frees each graph; final singleton uses denominator one."""
    groups = accumulation_groups(rows, seed=config["seed"], size=config["accumulation"])
    if len(rows) != config["examples"] or len(groups) != config["updates"]:
        raise ValueError("sample/update count mismatch")
    parameters = [p for _, p in state]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"],
        betas=tuple(config["betas"]), eps=config["epsilon"], weight_decay=config["weight_decay"])
    updates = []
    for step, group in enumerate(groups, 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        started = time.monotonic()
        for row in group:
            ids = torch.tensor([row["input_ids"][:-1]], dtype=torch.long, device="cuda:0")
            labels = torch.tensor([row["labels"][1:]], dtype=torch.long, device="cuda:0")
            logits = model(ids)
            mask = labels != -100
            loss = torch.nn.functional.cross_entropy(logits[mask].float(), labels[mask])
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite loss")
            (loss / len(group)).backward()
            losses.append(float(loss.detach()))
            del logits, loss, ids, labels, mask
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError("missing/nonfinite state gradient")
        if any(p.grad is not None or p.requires_grad for _, p in base):
            raise ValueError("base received gradient")
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip_l2"],
                                                  error_if_nonfinite=True))
        if norm == 0:
            raise ValueError("all state gradients are zero")
        record = {"step": step, "ids": [r["id"] for r in group], "loss_divisor": len(group),
                  "sample_losses": losses, "mean_loss": sum(losses)/len(group),
                  "gradient_l2_before_clip": norm}
        # A crash between these receipts leaves an explicitly unknown update, never a retry.
        write(output / f"step-{step:02d}-INTENT.json", record)
        optimizer.step()
        torch.cuda.synchronize()
        if not all(torch.isfinite(p).all() for p in parameters):
            raise ValueError("nonfinite updated state")
        record["seconds"] = time.monotonic() - started
        write(output / f"step-{step:02d}-DONE.json", record)
        print(json.dumps(record), flush=True)
        updates.append(record)
    optimizer.zero_grad(set_to_none=True)
    return updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if sha(args.config) != args.config_sha256:
        raise ValueError("configuration SHA mismatch")
    config = json.loads(args.config.read_text())
    check_config(config)
    gpu = host_guard()  # Still before importing Torch.
    output = bound(args.output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    def expired(signum, frame):
        raise TimeoutError("1800-second training limit; no resume")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(1800)
    try:
        rows = read_training_tokens(bound(config["train_tokens"]), config["train_sha256"], 21)
        vocab = Vocabulary(bound(config["peft_source"]) / "rwkv_vocab_v20230424.txt")
        import hashlib
        for row in rows:
            raw = b"".join(vocab.by_id[t] for t in row["input_ids"][:row["prompt_tokens"]])
            if hashlib.sha256(raw).hexdigest() != row["prompt_sha256"]:
                raise ValueError("prompt token bytes changed")
        write(output / "STARTED.json", {"config": config, "config_sha256": args.config_sha256,
              "gpu": gpu, "dev_or_heldout_content_read": False})
        torch, model, state, base, weights, versions = load_model(config, output)
        checkpoints = [save_state(torch, state, output, 0)]
        updates = train_groups(torch, model, state, base, rows, config["training"], output)
        checkpoints.append(save_state(torch, state, output, 11))
        equal = check_base(torch, base, weights, versions)
        write(output / "BASE-INTEGRITY.json", equal)
        write(output / "COMPLETED.json", {"status": "TRAIN_COMPLETE", "optimizer_updates": len(updates),
              "config_sha256": args.config_sha256,
              "examples": sum(len(x["ids"]) for x in updates), "backward_calls": 21,
              "base_tensors_unchanged": len(equal), "checkpoints": checkpoints,
              "seconds": time.monotonic()-began, "dev_or_heldout_content_read": False,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "production_promoted": False, "quality_evaluated": False})
    except BaseException as error:
        write(output / "FAILED.json", {"error": repr(error), "seconds": time.monotonic()-began,
              "confirmed_updates": len(list(output.glob("step-*-DONE.json"))),
              "attempted_updates": len(list(output.glob("step-*-INTENT.json"))), "retry": False})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
