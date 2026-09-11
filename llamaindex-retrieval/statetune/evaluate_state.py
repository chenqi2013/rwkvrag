"""Fixed dev inputs, zero/final states, greedy FLA generation; no gold access."""
import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

from preflight_state import bound, host_guard, sha, write
from pilot_runtime import load_model, check_base
from state_tokens import Vocabulary
from train_state import check_config


def generate(prompt, vocab, decide, *, cap=32):
    """Recompute each full prefix from the same initial state; never repair output."""
    if not prompt or len(prompt) + cap - 1 > 4096:
        raise ValueError("complete generation would exceed context")
    generated = []
    for _ in range(cap):
        token = decide(prompt + generated)
        if type(token) is not int or (token != 0 and token not in vocab):
            raise ValueError("invalid generated token")
        generated.append(token)
        if token == 0:
            break
    raw = b"".join(vocab[t] for t in generated if t != 0)
    try:
        text = raw.decode("utf-8")
        utf8 = True
    except UnicodeDecodeError:
        text, utf8 = raw.decode("utf-8", errors="replace"), False
    return {"token_ids": generated, "raw_hex": raw.hex(), "raw_text": text,
            "utf8_valid": utf8, "eos_observed": generated[-1] == 0,
            "cap_reached": generated[-1] != 0, "gold_opened": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--config-sha256", required=True)
    ap.add_argument("--training-run", required=True)
    ap.add_argument("--training-receipt-sha256", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if sha(args.config) != args.config_sha256:
        raise ValueError("configuration SHA mismatch")
    config = json.loads(args.config.read_text())
    check_config(config)
    ev = config["evaluation"]
    if (ev["split"], ev["count"], ev["steps"], ev["max_output_tokens"], ev["wall_seconds"]) != (
            "dev", 11, [0, 11], 32, 900):
        raise ValueError("evaluation contract differs")
    run = bound(args.training_run)
    if sha(run / "COMPLETED.json") != args.training_receipt_sha256:
        raise ValueError("training receipt changed")
    receipt = json.loads((run / "COMPLETED.json").read_text())
    if (receipt["status"] != "TRAIN_COMPLETE" or receipt["optimizer_updates"] != 11
            or receipt["config_sha256"] != args.config_sha256):
        raise ValueError("training incomplete or from different configuration")
    gpu = host_guard()
    output = bound(args.output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    def expired(signum, frame):
        raise TimeoutError("900-second evaluation limit; no retry")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(900)
    forwards = 0
    try:
        raw = bound(ev["inputs"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != ev["inputs_sha256"]:
            raise ValueError("evaluation inputs changed")
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) != 11 or len({r["id"] for r in rows}) != 11 or any(r["split"] != "dev" for r in rows):
            raise ValueError("wrong evaluation cases")
        vocab = Vocabulary(bound(config["peft_source"]) / "rwkv_vocab_v20230424.txt")
        prepared = []
        for row in rows:
            if hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
                raise ValueError("prompt hash mismatch")
            tokens = vocab.encode(row["prompt"])
            if len(tokens) != row["input_token_count"] or len(tokens) + 31 > 4096:
                raise ValueError("complete prompt token budget changed")
            prepared.append((row, tokens))
        write(output / "STARTED.json", {"config_sha256": args.config_sha256, "gpu": gpu,
              "training_receipt_sha256": args.training_receipt_sha256,
              "gold_opened": False, "heldout_read": False, "decode": "full_prefix_peft_fla"})
        torch, model, state, base, weights, versions = load_model(config, output)
        model.eval().requires_grad_(False)
        completed = []
        if [s["step"] for s in receipt["checkpoints"]] != [0,11]:
            raise ValueError("wrong checkpoint set")
        with torch.no_grad():
            for spec in receipt["checkpoints"]:
                path = run / spec["path"]
                if path.parent != run or sha(path) != spec["sha256"]:
                    raise ValueError("state checkpoint changed")
                canonical = torch.load(path, map_location="cpu", weights_only=True)
                if set(canonical) != {n for n,p in state}:
                    raise ValueError("state keys changed")
                for n,p in state:
                    value = canonical[n]
                    if value.shape != p.shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                        raise ValueError("invalid state tensor")
                    p.copy_(value)
                for row, tokens in prepared:
                    def decide(prefix):
                        nonlocal forwards
                        forwards += 1
                        if forwards > 704:
                            raise ValueError("forward call cap")
                        ids = torch.tensor([prefix], dtype=torch.long, device="cuda:0")
                        logits = model(ids)[0,-1,:]
                        if not torch.isfinite(logits).all():
                            raise ValueError("nonfinite logits")
                        return int(torch.argmax(logits))
                    result = {"id": row["id"], "step": spec["step"],
                              "prompt_sha256": row["prompt_sha256"], "state_sha256": spec["sha256"],
                              "output": generate(tokens, vocab.by_id, decide)}
                    name = f"step-{spec['step']:02d}-{row['id']}.json"
                    write(output / name, result)
                    completed.append({"path": name, "sha256": sha(output/name)})
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                if not all(torch.equal(p.detach().cpu(), canonical[n]) for n,p in state):
                    raise ValueError("initial state mutated by evaluation")
        equal = check_base(torch, base, weights, versions)
        write(output / "BASE-INTEGRITY.json", equal)
        write(output / "COMPLETED.json", {"status": "GENERATION_COMPLETE", "records": completed,
              "config_sha256": args.config_sha256,
              "training_receipt_sha256": args.training_receipt_sha256,
              "forward_calls": forwards, "gold_opened": False, "heldout_read": False,
              "optimizer_updates": 0, "seconds": time.monotonic()-began,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
    except BaseException as error:
        write(output / "FAILED.json", {"error": repr(error), "forward_calls": forwards,
                                      "seconds": time.monotonic()-began, "retry": False})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
