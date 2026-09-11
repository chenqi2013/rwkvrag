"""Local scoring after every dev generation is durable. Never starts a GPU run."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llamaindex_retrieval.rwkv_pipeline import parse_task_selections


def score_case(output, expected, unit_count):
    selected, valid = [], False
    if output["eos_observed"] and not output["cap_reached"] and output["utf8_valid"]:
        try:
            selected = [f"E{x}" for x in parse_task_selections(output["raw_text"], unit_count)]
            valid = True
        except ValueError:
            pass
    return {"valid": valid, "selected": selected, "expected": expected,
            "correct": valid and set(selected) == set(expected),
            "negative_false_positive": not expected and bool(selected)}


def select_state(scores):
    best = min(scores, key=lambda s: (-s["correct"], s["negative_false_positives"], s["step"]))
    zero = next(s for s in scores if s["step"] == 0)
    return {"selected_step": best["step"], "heldout_eligible": best["step"] == 11
            and best["correct"] > zero["correct"]
            and best["positive_correct"] > zero["positive_correct"]
            and best["negative_correct"] >= zero["negative_correct"]
            and best["negative_false_positives"] <= zero["negative_false_positives"],
            "production_promoted": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--training-run", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(__file__).resolve().parents[2]
    for path, digest in config["scoring_bindings"].items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest() != digest:
            raise ValueError("local scoring implementation changed")
    ev = config["evaluation"]
    receipt = json.loads((args.run/"COMPLETED.json").read_text())
    config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
    train_raw = (args.training_run/"COMPLETED.json").read_bytes()
    training = json.loads(train_raw)
    if (receipt["config_sha256"] != config_sha or training["config_sha256"] != config_sha
            or hashlib.sha256(train_raw).hexdigest() != receipt["training_receipt_sha256"]
            or training["status"] != "TRAIN_COMPLETE" or training["optimizer_updates"] != 11):
        raise ValueError("training/generation/configuration receipt chain differs")
    states = {s["step"]:s["sha256"] for s in training["checkpoints"]}
    if set(states) != {0,11}:
        raise ValueError("checkpoint set differs")
    if receipt["status"] != "GENERATION_COMPLETE" or len(receipt["records"]) != 22 or receipt["gold_opened"]:
        raise ValueError("all 22 raw generations must be durable before gold access")
    raw = (args.dataset/"dev.inputs.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != ev["inputs_sha256"]:
        raise ValueError("inputs hash mismatch")
    inputs = {r["id"]:r for r in map(json.loads, raw.splitlines())}
    records = {}
    for pin in receipt["records"]:
        path = args.run/pin["path"]
        raw = path.read_bytes()
        if path.parent != args.run or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
            raise ValueError("generation hash mismatch")
        row = json.loads(raw)
        if row["state_sha256"] != states[row["step"]]:
            raise ValueError("generation state does not match training")
        if row["prompt_sha256"] != inputs[row["id"]]["prompt_sha256"]:
            raise ValueError("generation prompt mismatch")
        key = (row["step"],row["id"])
        if key in records:
            raise ValueError("duplicate generation")
        records[key] = row
    if set(records) != {(step, identity) for step in [0,11] for identity in inputs}:
        raise ValueError("generation coverage differs")
    # First gold content access is after verifying the complete raw generation set.
    raw = (args.dataset/"dev.gold.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != ev["gold_sha256"]:
        raise ValueError("gold hash mismatch")
    gold = {r["id"]:r for r in map(json.loads, raw.splitlines())}
    if set(gold) != set(inputs):
        raise ValueError("gold coverage differs")
    scores = []
    for step in [0,11]:
        cases = [{"id": identity, **score_case(records[step,identity]["output"],
                    gold[identity]["expected_unit_ids"], len(item["units"]))}
                 for identity,item in inputs.items()]
        scores.append({"step": step, "count":len(cases), "correct":sum(c["correct"] for c in cases),
            "positive_correct":sum(c["correct"] and bool(c["expected"]) for c in cases),
            "negative_correct":sum(c["correct"] and not c["expected"] for c in cases),
            "invalid":sum(not c["valid"] for c in cases),
            "negative_false_positives":sum(c["negative_false_positive"] for c in cases), "cases":cases})
    result = {"scores":scores, **select_state(scores), "gold_opened_after_durable_raw": True,
              "heldout_read":False, "independent_dataset_author_review":True,
              "blind_benchmark":False}
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k != "scores"}))


if __name__ == "__main__":
    main()
