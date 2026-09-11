"""One canonical-format training example, one update, identical before/after probe."""
import argparse
import json
from pathlib import Path
import signal
import time

from preflight_state import bound, host_guard, sha, write
from train_state import check_config, train_groups
from pilot_runtime import load_model, check_base, save_state
from state_training import read_training_tokens, longest_training_row
from state_tokens import Vocabulary
from evaluate_state import generate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--config-sha256", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if sha(args.config) != args.config_sha256:
        raise ValueError("probe configuration changed")
    config = json.loads(args.config.read_text())
    for p,h in config["bindings"].items():
        if sha(bound(p)) != h: raise ValueError("probe binding changed: " + p)
    pilot = json.loads(bound(config["pilot"]).read_text())
    check_config(pilot)
    if config["limits"] != {"updates":1,"examples":1,"max_generation_tokens":32,"wall_seconds":900}:
        raise ValueError("probe limits changed")
    gpu = host_guard()
    output = bound(args.output)
    output.mkdir(parents=True,exist_ok=False)
    began = time.monotonic()
    def expired(signum,frame): raise TimeoutError("900-second single-update limit")
    signal.signal(signal.SIGALRM,expired)
    signal.alarm(900)
    try:
        rows = read_training_tokens(bound(pilot["train_tokens"]),pilot["train_sha256"],21)
        row = longest_training_row(rows)
        if row["id"] != config["sample_id"]: raise ValueError("sample selection changed")
        vocab = Vocabulary(bound(pilot["peft_source"])/"rwkv_vocab_v20230424.txt")
        prompt = b"".join(vocab.by_id[t] for t in row["input_ids"][:row["prompt_tokens"]])
        if not prompt.endswith(b"Assistant: <think></think>\n"):
            raise ValueError("canonical prompt boundary missing")
        write(output/"STARTED.json",{"config_sha256":args.config_sha256,"gpu":gpu,
              "sample_id":row["id"],"train_only":True,"learning_rate":pilot["training"]["learning_rate"]})
        torch,model,state,base,weights,versions = load_model(pilot,output)
        checkpoints = [save_state(torch,state,output,0)]
        forwards = 0
        def measure():
            nonlocal forwards
            model.eval()
            with torch.no_grad():
                ids = torch.tensor([row["input_ids"][:-1]],device="cuda:0",dtype=torch.long)
                labels = torch.tensor([row["labels"][1:]],device="cuda:0",dtype=torch.long)
                forwards += 1
                logits = model(ids)
                mask = labels != -100
                losses = torch.nn.functional.cross_entropy(logits[mask].float(),labels[mask],reduction="none")
                if not torch.isfinite(losses).all(): raise ValueError("nonfinite probe loss")
                values = losses.tolist()
                del ids,labels,logits,mask,losses
                def decide(prefix):
                    nonlocal forwards
                    forwards += 1
                    if forwards > 66: raise ValueError("probe evaluation forward cap")
                    logits = model(torch.tensor([prefix],device="cuda:0",dtype=torch.long))[0,-1,:]
                    if not torch.isfinite(logits).all(): raise ValueError("nonfinite generation logits")
                    return int(torch.argmax(logits))
                return {"mean_target_eos_loss":sum(values)/len(values),"per_token_loss":values,
                        "output":generate(row["input_ids"][:row["prompt_tokens"]],vocab.by_id,decide)}
        before = measure()
        write(output/"BEFORE.json",before)
        model.train()
        settings = {**pilot["training"],"examples":1,"updates":1,"accumulation":1}
        updates = train_groups(torch,model,state,base,[row],settings,output)
        after = measure()
        write(output/"AFTER.json",after)
        checkpoints.append(save_state(torch,state,output,1))
        equal = check_base(torch,base,weights,versions)
        write(output/"BASE-INTEGRITY.json",equal)
        write(output/"COMPLETED.json",{"status":"SINGLE_UPDATE_COMPLETE","config_sha256":args.config_sha256,
              "optimizer_updates":len(updates),"sample_id":row["id"],"before":before,"after":after,
              "loss_decreased":after["mean_target_eos_loss"] < before["mean_target_eos_loss"]*(1-1e-4),
              "checkpoints":checkpoints,"base_tensors_unchanged":1062,
              "evaluation_forward_calls":forwards,"training_backward_calls":1,
              "seconds":time.monotonic()-began,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
              "full_training_authorized_by_this_receipt":False,"dev_or_heldout_content_read":False})
    except BaseException as error:
        write(output/"FAILED.json",{"error":repr(error),"seconds":time.monotonic()-began,
              "confirmed_updates":len(list(output.glob('step-*-DONE.json'))),
              "attempted_updates":len(list(output.glob('step-*-INTENT.json')))})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__": main()
