"""Frozen train-only 2x2 newline/state diagnostic, with zero parameter updates."""
import argparse
import json
from pathlib import Path
import signal
import time

from preflight_state import bound, host_guard, sha, write
from train_state import check_config
from pilot_runtime import load_model, check_base
from state_training import read_training_tokens
from state_tokens import Vocabulary
from evaluate_state import generate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--config-sha256", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if sha(args.config) != args.config_sha256:
        raise ValueError("diagnostic config changed")
    config = json.loads(args.config.read_text())
    if config["limits"] != {"updates":0,"examples":4,"variants":2,"states":2,
                            "max_generation_tokens":32,"max_forward_calls":528,"wall_seconds":900}:
        raise ValueError("diagnostic limits changed")
    for path, digest in config["bindings"].items():
        if sha(bound(path)) != digest:
            raise ValueError("diagnostic binding mismatch: " + path)
    pilot = json.loads(bound(config["pilot"]).read_text())
    check_config(pilot)
    gpu = host_guard()
    output = bound(args.output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    def expired(signum, frame):
        raise TimeoutError("900-second diagnostic limit")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(900)
    forwards = 0
    try:
        rows = read_training_tokens(bound(config["inputs"]),config["bindings"][config["inputs"]],8)
        if len({r["source_id"] for r in rows}) != 4:
            raise ValueError("wrong original case count")
        vocab = Vocabulary(bound(pilot["peft_source"]) / "rwkv_vocab_v20230424.txt")
        torch, model, state, base, weights, versions = load_model(pilot,output)
        model.eval().requires_grad_(False)
        write(output / "STARTED.json", {"config_sha256":args.config_sha256,"gpu":gpu,
              "train_only":True,"optimizer_updates":0,"dev_or_heldout_content_read":False})
        results = []
        with torch.no_grad():
            for spec in config["states"]:
                canonical = torch.load(bound(spec["path"]),map_location="cpu",weights_only=True)
                if set(canonical) != {n for n,p in state}:
                    raise ValueError("wrong state keys")
                for n,p in state:
                    t = canonical[n]
                    if t.shape != p.shape or t.dtype != torch.float32 or not torch.isfinite(t).all():
                        raise ValueError("invalid state")
                    p.copy_(t)
                for row in rows:
                    ids = torch.tensor([row["input_ids"][:-1]],device="cuda:0",dtype=torch.long)
                    labels = torch.tensor([row["labels"][1:]],device="cuda:0",dtype=torch.long)
                    forwards += 1
                    logits = model(ids)
                    mask = labels != -100
                    losses = torch.nn.functional.cross_entropy(logits[mask].float(),labels[mask],reduction="none")
                    if not torch.isfinite(losses).all():
                        raise ValueError("nonfinite diagnostic loss")
                    loss_values = losses.tolist()
                    del ids, labels, logits, losses, mask
                    def decide(prefix):
                        nonlocal forwards
                        forwards += 1
                        if forwards > 528:
                            raise ValueError("forward call cap")
                        logits = model(torch.tensor([prefix],device="cuda:0",dtype=torch.long))[0,-1,:]
                        if not torch.isfinite(logits).all():
                            raise ValueError("nonfinite final logits")
                        return int(torch.argmax(logits))
                    record = {"id":row["id"],"source_id":row["source_id"],"variant":row["variant"],
                              "step":spec["step"],"prompt_sha256":row["prompt_sha256"],
                              "state_sha256":config["bindings"][spec["path"]],
                              "mean_target_eos_loss":sum(loss_values)/len(loss_values),
                              "per_token_loss":loss_values,
                              "output":generate(row["input_ids"][:row["prompt_tokens"]],vocab.by_id,decide)}
                    name = f"step-{spec['step']:02d}-{row['id']}.json"
                    write(output/name,record)
                    results.append({"path":name,"sha256":sha(output/name)})
                    print(json.dumps({"id":row["id"],"step":spec["step"],"loss":record["mean_target_eos_loss"]}),flush=True)
                if not all(torch.equal(p.detach().cpu(),canonical[n]) for n,p in state):
                    raise ValueError("state changed during diagnostic")
        equal = check_base(torch,base,weights,versions)
        write(output/"BASE-INTEGRITY.json",equal)
        write(output/"COMPLETED.json",{"status":"DIAGNOSTIC_COMPLETE","config_sha256":args.config_sha256,
              "records":results,"forward_calls":forwards,"optimizer_updates":0,
              "train_only":True,"seconds":time.monotonic()-began,"state_unchanged":True,
              "base_tensors_unchanged":1062,"peak_allocated_bytes":torch.cuda.max_memory_allocated()})
    except BaseException as error:
        write(output/"FAILED.json",{"error":repr(error),"forward_calls":forwards,"optimizer_updates":0,
                                    "seconds":time.monotonic()-began})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
