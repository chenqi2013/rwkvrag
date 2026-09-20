"""Post-run export and byte-level audit; not part of the frozen inference/scoring."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil


def digest(value):
    return sha256(value.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    bindings = json.loads((args.run / "BINDINGS.json").read_text())
    assert all(sha256(Path(p).read_bytes()).hexdigest() == h for p, h in bindings["files"].items())
    assert json.loads((args.run / "EXECUTION.json").read_text())["complete"]
    summary = json.loads((args.run / "SUMMARY.json").read_text())
    calls = [json.loads(p.read_text()) for p in sorted((args.run / "calls").glob("*.json"))]
    assert len(calls) == summary["calls"] == 640
    keyed = {(c["result"]["round"], c["case"]["id"], c["result"]["arm"]): c for c in calls}
    assert len(keyed) == 640
    finishes, parameters = Counter(), set()
    for c in calls:
        t = c["trace"]
        assert digest(t["prompt"]) == t["prompt_sha256"] == t["budget"]["counted_prompt_sha256"]
        assert digest(c["case"]["text"]) == t["source_sha256"]
        assert digest(t["raw_text"]) == t["raw_text_sha256"]
        assert t["model"] == bindings["model"] and t["state_id"] is None
        assert t["requested_temperature"] == 0 and t["parameters"]["max_tokens"] == 32
        assert t["budget"]["application_policy_fits"]
        assert t["prefill_mode"] == "complete" and t["stop_tokens_requested"] == [0]
        parameters.add(json.dumps(t["parameters"], sort_keys=True))
        finishes[t["provider_finish_reason"]] += 1
    assert len(parameters) == 1
    raw_agreement = Counter()
    for case_id in {c["case"]["id"] for c in calls}:
        for round_id in (1, 2):
            a = keyed[round_id, case_id, "yes-no"]["trace"]["messages"][0]["content"]
            b = keyed[round_id, case_id, "neutral-labels"]["trace"]["messages"][0]["content"]
            instruction, separator, data = a.partition("\n")
            instruction = instruction.replace("输出NO。", "输出INSUFFICIENT。")
            instruction = instruction.replace('"YES"', '"ANSWERABLE"').replace('"NO"', '"INSUFFICIENT"')
            assert instruction + separator + data == b
        for arm in ("yes-no", "neutral-labels"):
            a, b = (keyed[r, case_id, arm]["trace"] for r in (1, 2))
            assert a["prompt"] == b["prompt"]
            raw_agreement[arm] += a["raw_text"] == b["raw_text"]
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("BINDINGS.json", "EXECUTION.json", "SUMMARY.json", "ROWS.json"):
        shutil.copyfile(args.run / name, args.output / name)
    for path in args.run.glob("HEALTH-*.json"):
        shutil.copyfile(path, args.output / path.name)
    with (args.output / "calls.jsonl").open("w") as stream:
        for c in calls:
            stream.write(json.dumps(c, ensure_ascii=False) + "\n")
    audit = {"calls": len(calls), "label_only_pairs_verified": 320,
             "unchanged_repeat_prompts": 320, "bound_files_unchanged": True,
             "prompt_source_raw_hashes_verified": True,
             "provider_finish_reasons": finishes, "identical_raw_outputs_across_rounds": raw_agreement,
             "requested_temperature": 0, "wire_parameters": json.loads(next(iter(parameters))),
             "sampling_note": "Client clamps temperature to 0.001; dedicated server selects logits.argmax(). Both arms identical.",
             "input_token_range": [min(c["trace"]["budget"]["input_tokens"] for c in calls),
                                   max(c["trace"]["budget"]["input_tokens"] for c in calls)],
             "call_file_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted((args.run / "calls").glob("*.json"))},
             "export_sha256": sha256((args.output / "calls.jsonl").read_bytes()).hexdigest()}
    (args.output / "AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in audit.items() if k != "call_file_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
