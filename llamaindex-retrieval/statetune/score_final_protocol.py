"""Verify every final-protocol raw run before the first gold read; never tune."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

from preflight_state import bound, sha, write
from score_exploration import score


def verify_all(protocol_path, expected_sha, run_root):
    if sha(protocol_path) != expected_sha:
        raise ValueError("final protocol changed")
    protocol = json.loads(protocol_path.read_text())
    if (protocol["schema"] != "rwkv_reader_final_heldout_v1"
            or protocol["heldout_trial_count"] != 1 or protocol["optimizer_updates"] != 0
            or protocol["total_generation_cap"] != 66 or len(protocol["configs"]) != 3
            or not protocol["score_after_all_raw_complete"] or not protocol["no_post_heldout_tuning"]):
        raise ValueError("unsupported final protocol")
    verified = []
    total = 0
    for trial in protocol["configs"]:
        config_path = bound(trial["config"])
        if sha(config_path) != trial["sha256"]:
            raise ValueError("final configuration changed")
        config = json.loads(config_path.read_text())
        for path, digest in config["bindings"].items():
            if sha(bound(path)) != digest:
                raise ValueError("final source binding changed")
        spec = config["evaluation"]
        if spec["split"] != "heldout" or spec["max_output_tokens"] != 32:
            raise ValueError("final evaluation protocol mismatch")
        inputs_path = bound(spec["inputs"])
        if sha(inputs_path) != spec["inputs_sha256"]:
            raise ValueError("final inputs changed")
        items = [json.loads(s) for s in inputs_path.read_text().splitlines()]
        inputs = {r["id"]:r for r in items}
        states = {s["name"]:s for s in spec["states"]}
        if (len(inputs) != 11 or len(items) != 11 or set(states) != {"zero", "epoch-11"}
                or len(spec["states"]) != 2 or any(r["split"] != "heldout" for r in items)
                or any(not isinstance(r.get("units"), list) or not r["units"] for r in items)
                or states["epoch-11"]["sha256"] != protocol["candidate_state_sha256"]):
            raise ValueError("final input/state coverage changed")
        for snapshot in states.values():
            if sha(bound(snapshot["path"])) != snapshot["sha256"]:
                raise ValueError("final state file changed")
        run = run_root / trial["run"]
        if run.parent != run_root:
            raise ValueError("run path escapes root")
        receipt_path = run / "COMPLETED.json"
        receipt = json.loads(receipt_path.read_text())
        if (receipt["status"] != "GENERATION_COMPLETE" or receipt["gold_opened"]
                or receipt["optimizer_updates"] != 0 or receipt["split"] != "heldout"
                or receipt["config_sha256"] != trial["sha256"]):
            raise ValueError("final completion receipt mismatch")
        seen = set()
        for pin in receipt["records"]:
            path = run / pin["path"]
            if path.parent != run or sha(path) != pin["sha256"]:
                raise ValueError("final raw output changed")
            record = json.loads(path.read_text())
            raw = record.get("output")
            if (not isinstance(raw, dict) or not isinstance(raw.get("raw_text"), str)
                    or any(type(raw.get(k)) is not bool
                           for k in ("utf8_valid", "eos_observed", "cap_reached"))):
                raise ValueError("final raw output lacks scoring fields")
            key = (record["state"], record["id"])
            if (key in seen or record["state_sha256"] != states[record["state"]]["sha256"]
                    or record["prompt_sha256"] != inputs[record["id"]]["prompt_sha256"]):
                raise ValueError("final raw identity changed")
            seen.add(key)
        if seen != {(s,i) for s in states for i in inputs}:
            raise ValueError("incomplete final raw coverage")
        total += len(seen)
        verified.append({**trial, "receipt_sha256": sha(receipt_path), "raw_records": len(seen)})
    if total != 66 or len({r["run"] for r in verified}) != 3 or sum(r["primary"] for r in verified) != 1:
        raise ValueError("final run coverage/primary mismatch")
    return protocol, verified


def run(protocol_path, expected_sha, run_root, gold, output):
    if output.exists():
        raise ValueError("final score output already exists")
    protocol, verified = verify_all(protocol_path, expected_sha, run_root)
    # Durable certificate precedes score(), the first function that opens gold.
    write(output.with_suffix(".raw-ready.json"), {"protocol_sha256": expected_sha,
          "orchestrator_sha256": sha(Path(__file__)),
          "runs": verified, "raw_records": 66, "gold_opened": False,
          "verified_at_utc": datetime.now(timezone.utc).isoformat()})
    results = {trial["run"]: score(bound(trial["config"]), run_root / trial["run"], gold)
               for trial in verified}
    primary = next(t["run"] for t in verified if t["primary"])
    result = {"protocol_sha256": expected_sha, "all_raw_verified_before_gold": True,
              "orchestrator_sha256": sha(Path(__file__)),
              "fixed_candidate_state_sha256": protocol["candidate_state_sha256"],
              "primary_run": primary, "runs": results,
              "primary_gate_passed": results[primary]["improves_over_zero"],
              "candidate_reselected": False, "post_heldout_optimizer_updates": 0,
              "production_promoted": False}
    write(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run(args.protocol, args.protocol_sha256, args.run_root, args.gold, args.output)
    print(json.dumps({"primary_run":result["primary_run"],"primary_gate_passed":result["primary_gate_passed"],
        "runs":{name:[{k:v for k,v in s.items() if k!="cases"} for s in r["scores"]]
                for name,r in result["runs"].items()}},ensure_ascii=False))


if __name__ == "__main__": main()
