"""Bounded zero-state Reader diagnostic: score complete legal labels + EOS.

Not a production decoder or natural-generation accuracy. No training, retries,
prompt changes, length normalization or candidate pruning based on gold.
"""

import argparse
from itertools import combinations
import json
from pathlib import Path
import signal
import sys
import time


def candidate_labels(units):
    if type(units) is not int or not 1 <= units <= 3:
        raise ValueError("diagnostic supports 1..3 units, at most eight candidates")
    ids = [f"E{i}" for i in range(1, units + 1)]
    return ["NONE"] + [",".join(group) for n in range(1, units + 1)
                       for group in combinations(ids, n)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; no automatic rerun or overwrite")
    sys.path.insert(0, str(Path(__file__).resolve().parent / "runtime"))
    from contracts import checked_file, sha_file, write_json
    from runtime import open_runtime, initial_state, clone_state
    config = json.loads(args.config.read_text())
    limits = {"cases": 8, "max_units": 3, "max_candidates_per_case": 8,
              "max_input_tokens": 4096, "wall_seconds": 1800, "optimizer_steps": 0}
    if config.get("limits") != limits or config.get("mode") != "dev":
        raise ValueError("only the fixed eight-case development diagnostic is allowed")
    input_path = checked_file(config["inputs_path"], config["inputs_sha256"])
    inputs = [json.loads(line) for line in input_path.read_text().splitlines()]
    if len(inputs) != 8 or len({r['id'] for r in inputs}) != 8:
        raise ValueError("must contain exactly eight distinct cases")
    for row in inputs:
        candidate_labels(len(row["units"]))
    signal.alarm(1800)
    args.output.mkdir(parents=True)
    write_json(args.output / "STARTED.json", {"config": config, "config_sha256": sha_file(args.config),
               "script_sha256": sha_file(Path(__file__)), "limits": limits})
    began = time.monotonic()
    try:
        torch, model, tokenizer, receipt = open_runtime(config)
        write_json(args.output / "RUNTIME.json", receipt)
        results, forwards = [], 0
        for row in inputs:
            prompt_ids = tokenizer.encode(row["prompt"])
            labels = candidate_labels(len(row["units"]))
            sequences = [tokenizer.encode(label) + [0] for label in labels]
            if len(prompt_ids) + max(map(len, sequences)) > 4096:
                raise ValueError("complete input exceeds the fixed budget; no truncation")
            logits, state = model.forward(prompt_ids, initial_state(torch))
            forwards += 1
            scores = []
            for label, ids in zip(labels, sequences, strict=True):
                current, copied, steps = logits.clone(), clone_state(state), []
                for index, token in enumerate(ids):
                    if not torch.isfinite(current).all():
                        raise ValueError("nonfinite model logits")
                    steps.append({"token": token, "log_probability":
                                  float(torch.log_softmax(current.float(), dim=-1)[token].item())})
                    if index + 1 < len(ids):
                        current, copied = model.forward([token], copied)
                        forwards += 1
                scores.append({"label": label, "tokens": steps,
                               "log_probability": sum(s["log_probability"] for s in steps)})
            selected = max(scores, key=lambda s: s["log_probability"])["label"]
            result = {"id": row["id"], "prompt_sha256": row["prompt_sha256"],
                      "candidates": scores, "selected_label": selected,
                      "selection_method": "maximum complete-sequence likelihood including EOS"}
            write_json(args.output / (row["id"] + ".json"), result)
            results.append(result)
            print(json.dumps({"completed_cases": len(results), "id": row["id"]}), flush=True)
        # Read scoring labels only after every independent prediction is on disk.
        gold_path = checked_file(config["gold_path"], config["gold_sha256"])
        golds = [json.loads(line) for line in gold_path.read_text().splitlines()]
        if [g["id"] for g in golds] != [r["id"] for r in results]:
            raise ValueError("gold identities differ from predictions")
        positive = sum(bool(g["expected_unit_ids"]) for g in golds)
        exact = sum(r["selected_label"] == g["target"] for r, g in zip(results, golds))
        positive_support = sum(bool(set(r["selected_label"].split(",")) & set(g["expected_unit_ids"]))
                               for r, g in zip(results, golds))
        negative_correct = sum(not g["expected_unit_ids"] and r["selected_label"] == "NONE"
                               for r, g in zip(results, golds))
        write_json(args.output / "COMPLETED.json", {
            "status": "COMPLETE", "cases": 8, "forward_calls": forwards, "optimizer_steps": 0,
            "positive": positive, "positive_any_support": positive_support,
            "negative": 8 - positive, "negative_correct": negative_correct,
            "exact_selection": exact, "seconds": time.monotonic() - began,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "production_promoted": False, "heldout_evaluated": False,
            "caveat": "Exposed dev set; forced candidate likelihood, not free-generation quality."})
    except BaseException as error:
        write_json(args.output / "FAILED.json", {"error_type": type(error).__name__,
                   "message": str(error), "seconds": time.monotonic() - began,
                   "optimizer_steps": 0})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
