"""Execute the four preregistered Writer arms sequentially without retries.

Each arm uses the existing native-material runner and its complete receipts.
This script does not score answers or promote a prompt to the default.
"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "llamaindex-retrieval"


def check_bindings(plan):
    for name, expected in plan["bindings"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen input changed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--plan", type=Path,
                        default=ROOT / "artifacts/writer-p0-20260916/PLAN.json")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    check_bindings(plan)
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (args.output / "PLAN.json").write_bytes(args.plan.read_bytes())
    # Prevent a local application setting from changing one arm's configuration.
    # The local comparison service does not require credentials.
    env = {k: v for k, v in os.environ.items() if not k.startswith("RWKVRAG_")}
    results = []
    for split in ("development", "holdout"):
        fixture = MODULE / ("eval/native-smoke/fixtures.jsonl" if split == "development"
                            else "eval/writer-p0-20260916/holdout.jsonl")
        for protocol in ("evidence_first", "evidence_checked"):
            check_bindings(plan)
            arm = f"{split}-{protocol}"
            command = [sys.executable, str(MODULE / "eval/native-smoke/run_smoke.py"),
                "--module-root", str(MODULE), "--fixtures", str(fixture),
                "--base-url", args.base_url, "--model", plan["model"],
                "--transport", "rwkvos_batch", "--writer-prefill", plan["prefill"],
                "--writer-prompt-protocol", protocol, "--prefill-mode", plan["prefill_mode"],
                "--state-id", plan["state_id"], "--stop-tokens", json.dumps(plan["stop_tokens"]),
                "--concurrency", str(plan["concurrency"]), "--batch-size", str(plan["batch_size"]),
                "--max-output-tokens", str(plan["max_output_tokens"]),
                "--timeout-seconds", str(plan["timeout_seconds"]),
                "--output", str(args.output / arm)]
            print(f"Starting {arm}", flush=True)
            with (args.output / f"{arm}.log").open("xb") as log:
                result = subprocess.run(command, cwd=MODULE, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
            results.append({"arm": arm, "exit_code": result.returncode})
            print(f"Finished {arm}: runner exit {result.returncode}", flush=True)
            # Exit 1 retains model failures in the denominator. A missing summary
            # is an infrastructure failure, so stop instead of fabricating scores.
            if not (args.output / arm / "summary.json").exists():
                raise RuntimeError(f"arm did not finalize: {arm}; inspect its log")
    (args.output / "ARMS.json").write_text(json.dumps(results, indent=2) + "\n")
    check_bindings(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
