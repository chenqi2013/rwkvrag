"""Compare explicitly reviewed arms without pooling development and holdout."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from llamaindex_retrieval.quality_gate import audit, read_json, require, write_new


def summarize(root):
    plan = read_json(root / "PLAN.json")
    comparisons = {}
    reports = {}
    for split in ("development", "holdout"):
        arms = []
        for protocol in ("evidence_first", "evidence_checked"):
            name = f"{split}-{protocol}"
            folder = root / name
            report, _ = audit(folder, folder / "review.json")
            manifest = read_json(folder / "manifest.json")
            require(report["reviewed_cases"] == report["planned_cases"], "review is incomplete")
            require(report["configuration"]["writer_prompt_protocol"] == protocol,
                    "arm prompt does not match its name")
            reports[name] = report
            records = read_json(folder / "results.json")
            summary = read_json(folder / "summary.json")
            expected_split = "development_smoke_not_blind" if split == "development" else "frozen_holdout"
            fixture_name = ("llamaindex-retrieval/eval/native-smoke/fixtures.jsonl" if split == "development"
                            else "llamaindex-retrieval/eval/writer-p0-20260916/holdout.jsonl")
            require(report["evaluation_split"] == expected_split
                    and report["fixture_sha256"] == plan["bindings"][fixture_name], "arm split mismatch")
            require(report["configuration"]["model"] == plan["model"]
                    and report["configuration"]["state_id"] == plan["state_id"], "model/state mismatch")
            arms.append({"quality": report, "manifest": manifest, "elapsed_ms": summary["elapsed_ms"],
                         "output_characters": sum(len(r["response"]["answer"])
                                                  for r in records if r["response"] is not None)})
        baseline, candidate = arms
        a, b = baseline["quality"], candidate["quality"]
        for key in ("production_sources_sha256", "runner_sha256"):
            require(baseline["manifest"][key] == candidate["manifest"][key], "arm source mismatch")
        require(a["fixture_sha256"] == b["fixture_sha256"], "arms used different materials")
        require({k: v for k, v in a["configuration"].items() if k != "writer_prompt_protocol"}
                == {k: v for k, v in b["configuration"].items() if k != "writer_prompt_protocol"},
                "more than one configuration variable changed")
        changes = [{"id": x["id"], "baseline": x["decision"], "candidate": y["decision"]}
                   for x, y in zip(a["cases"], b["cases"], strict=True)]
        comparisons[split] = {"baseline_passed": a["passed_cases"],
            "candidate_passed": b["passed_cases"], "total": a["planned_cases"],
            "improvements": [x["id"] for x in changes
                             if x["baseline"] != "pass" and x["candidate"] == "pass"],
            "regressions": [x["id"] for x in changes
                            if x["baseline"] == "pass" and x["candidate"] != "pass"],
            "elapsed_ms": {"baseline": baseline["elapsed_ms"], "candidate": candidate["elapsed_ms"]},
            "output_characters": {"baseline": baseline["output_characters"],
                                  "candidate": candidate["output_characters"]},
            "cases": changes}
    dev, held = comparisons["development"], comparisons["holdout"]
    held_cases = {r["id"]: r for r in reports["holdout-evidence_checked"]["cases"]}
    admitted = (dev["candidate_passed"] >= dev["baseline_passed"] + 1
                and held["candidate_passed"] >= held["baseline_passed"]
                and all(held_cases[key]["decision"] == "pass" for key in ("smoke_003", "smoke_008")))
    return {"schema": "rwkvrag-writer-prompt-comparison-v1",
            "plan_sha256": sha256((root / "PLAN.json").read_bytes()).hexdigest(),
            "preregistered_criteria": plan["candidate_acceptance"],
            "passes_material_candidate_criteria": admitted,
            "default_promoted": False, "full_rag_verified": False,
            "comparisons": comparisons, "arms": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.run_root)
    write_new(args.output, report)
    print(json.dumps({"passes_material_candidate_criteria": report["passes_material_candidate_criteria"],
                      "comparisons": report["comparisons"]}, ensure_ascii=False, indent=2))
