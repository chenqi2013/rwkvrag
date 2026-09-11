"""One bounded Writer state epoch; frozen Reader runtime and math are reused.

CPU validation reads only the Writer train release. A matching longest-row
zero-update preflight is mandatory before the fixed 54-backward/27-update run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import signal
import time

from preflight_state import PROJECT, CHECKPOINT, bound, host_guard, sha, write
from pilot_runtime import load_model, check_base
from train_released_state import execute, PACKAGES, PEFT_MANIFEST_SHA, REQUIRED_CODE
from state_training import read_training_tokens, longest_training_row
from state_tokens import Vocabulary

TRAINING = {"epochs": 1, "learning_rate": 1e-5, "accumulation": 2, "seed": 20260909}


def validate(config):
    if (config.get("schema") != "rwkv_writer_training_v1"
            or config.get("mode") not in {"preflight", "train"}
            or config.get("training") != TRAINING or config.get("packages") != PACKAGES
            or type(config.get("wall_seconds")) is not int
            or not 60 <= config["wall_seconds"] <= 3600
            or config.get("max_sequence_tokens") != 8192):
        raise ValueError("unsupported Writer execution budget")
    if datetime.fromisoformat(config["deadline_utc"]).tzinfo is None:
        raise ValueError("timezone required")
    source = bound(config["runtime"]["peft_source"])
    required = {"llamaindex-retrieval/" + name for name in REQUIRED_CODE}
    required.add("llamaindex-retrieval/statetune/train_writer_state.py")
    required.add(str((source / "SOURCE-MANIFEST.json").relative_to(PROJECT)))
    if not required <= set(config["bindings"]):
        raise ValueError("missing executed source binding")
    for name, expected in config["bindings"].items():
        if sha(bound(name)) != expected:
            raise ValueError("executed source changed: " + name)
    if sha(source / "SOURCE-MANIFEST.json") != PEFT_MANIFEST_SHA:
        raise ValueError("unrecognized PEFT runtime")
    for name, pin in json.loads((source / "SOURCE-MANIFEST.json").read_text())["files"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or sha(path) != pin["sha256"]:
            raise ValueError("PEFT source changed")
    path = bound(config["release"])
    if sha(path) != config["release_sha256"]:
        raise ValueError("Writer release changed")
    release = json.loads(path.read_text())
    if (release.get("schema") != "rwkv_writer_release_v1"
            or release.get("training_data_admitted") is not True
            or release.get("checkpoint_sha256") != CHECKPOINT
            or release.get("prompt_protocol") != "rwkv_g1j_no_think_v1"
            or release.get("writer_prompt") != "writer_prompt_v2"
            or release.get("max_sequence_tokens") != 8192
            or release.get("max_generation_tokens") != 2048 or release.get("train_count") != 54
            or release.get("dev_or_heldout_content_included") is not False):
        raise ValueError("Writer release contract mismatch")
    for name in ("train.tokens.jsonl", "INDEPENDENT-REVIEW.json"):
        if sha(path.parent / name) != release["files"][name]:
            raise ValueError("released file changed")
    review = json.loads((path.parent / "INDEPENDENT-REVIEW.json").read_text())
    if (review.get("independent_of_author") is not True
            or review.get("reviewer") == review.get("author")
            or review.get("decisions") != release["rows"]):
        raise ValueError("independent review binding mismatch")
    approved = {r["id"]: r for r in release["rows"]}
    if len(approved) != 54 or any(r["decision"] != "approve" for r in approved.values()):
        raise ValueError("all 54 independent decisions required")
    rows = read_training_tokens(path.parent / "train.tokens.jsonl", release["files"]["train.tokens.jsonl"], 54, max_tokens=8192)
    if set(approved) != {r["id"] for r in rows}:
        raise ValueError("review does not cover training IDs")
    vocab_path = source / "rwkv_vocab_v20230424.txt"
    if sha(vocab_path) != release["vocabulary_sha256"]:
        raise ValueError("vocabulary changed")
    vocab = Vocabulary(vocab_path)
    for row in rows:
        n = row["prompt_tokens"]
        prompt = b"".join(vocab.by_id[i] for i in row["input_ids"][:n])
        target = b"".join(vocab.by_id[i] for i in row["input_ids"][n:-1])
        pin = approved[row["id"]]
        if (hashlib.sha256(prompt).hexdigest() != row["prompt_sha256"]
                or row["prompt_sha256"] != pin["prompt_sha256"]
                or hashlib.sha256(target).hexdigest() != pin["target_sha256"]
                or not prompt.endswith(b"\n\nAssistant: <think></think>\n")
                or n + 2047 > 8192):
            raise ValueError("reviewed prompt/target/budget mismatch")
    longest = longest_training_row(rows)
    identity = {"release_sha256": config["release_sha256"], "train_sha256": release["files"]["train.tokens.jsonl"],
        "train_count": 54, "runtime": config["runtime"], "bindings": config["bindings"],
        "packages": config["packages"], "max_sequence_tokens": 8192,
        "checkpoint_sha256": CHECKPOINT, "longest_id": longest["id"], "longest_tokens": len(longest["input_ids"])}
    if config["mode"] == "train":
        pin = config["preflight"]
        preflight = bound(pin["path"])
        if sha(preflight) != pin["sha256"]:
            raise ValueError("preflight receipt changed")
        receipt = json.loads(preflight.read_text())
        if (receipt.get("status") != "PREFLIGHT_COMPLETE" or receipt.get("identity") != identity
                or receipt.get("optimizer_updates") != 0 or receipt.get("backward_calls") != 1
                or receipt.get("base_tensors_unchanged") != 1062):
            raise ValueError("matching Writer zero-update preflight required")
    return rows, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if sha(args.config) != args.config_sha256:
        raise ValueError("config changed")
    config = json.loads(args.config.read_text())
    rows, identity = validate(config)
    if args.validate_only:
        print(json.dumps({"status": "CPU_VALIDATED", "identity": identity, "optimizer_updates": 0}, indent=2))
        return
    if not args.output:
        parser.error("GPU execution requires --output")
    remaining = lambda: (datetime.fromisoformat(config["deadline_utc"]) - datetime.now(timezone.utc)).total_seconds()
    if remaining() <= 60:
        raise ValueError("execution deadline exhausted")
    gpu = host_guard()
    if remaining() <= 60:
        raise ValueError("execution deadline exhausted during host guard")
    output = bound(args.output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    def expired(*unused):
        raise TimeoutError("bounded Writer execution interrupted or expired")
    previous = {s: signal.signal(s, expired) for s in (signal.SIGALRM, signal.SIGTERM)}
    signal.alarm(min(config["wall_seconds"], int(remaining()) - 15))
    try:
        if {name: importlib.metadata.version(name) for name in PACKAGES} != PACKAGES:
            raise ValueError("package versions changed")
        write(output / "STARTED.json", {"config": config, "identity": identity, "gpu": gpu,
            "config_sha256": args.config_sha256, "optimizer": {"betas": [0.9, 0.999], "eps": 1e-8,
            "weight_decay": 0, "clip_gradient_l2": 1.0}})
        torch, model, state, base, weights, versions = load_model(config["runtime"], output)
        if model.args.ctx_len != 4096:
            raise ValueError("unexpected frozen runtime context")
        model.args.ctx_len = 8192  # This instance only; before its first forward.
        write(output / "WRITER-CONTEXT.json", {"effective_ctx_len": model.args.ctx_len,
            "longest_actual_sequence": identity["longest_tokens"], "full8192_probe_claimed": False})
        result = execute(torch, model, state, rows, config, output)
        equal = check_base(torch, base, weights, versions)
        write(output / "BASE-INTEGRITY.json", equal)
        validate(config)  # Inputs and executed source must remain unchanged.
        write(output / "COMPLETED.json", {**result, "identity": identity,
            "base_tensors_unchanged": len(equal), "config_sha256": args.config_sha256,
            "seconds": time.monotonic() - began, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "dev_or_heldout_content_read": False, "production_promoted": False})
    except BaseException as error:
        write(output / "FAILED.json", {"error": repr(error), "seconds": time.monotonic() - began,
            "confirmed_updates": len(list(output.glob("step-*-DONE.json"))),
            "attempted_updates": len(list(output.glob("step-*-INTENT.json")))})
        raise
    finally:
        signal.alarm(0)
        for s, handler in previous.items():
            signal.signal(s, handler)


if __name__ == "__main__":
    main()
