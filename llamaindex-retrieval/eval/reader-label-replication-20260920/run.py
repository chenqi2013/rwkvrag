"""Two repeated, counterbalanced arms using the frozen label-only intervention."""
import argparse
import asyncio
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import perf_counter
from uuid import uuid4

import httpx

from llamaindex_retrieval.model_client import model_answer_bounds
from llamaindex_retrieval.rwkvos_batch import RwkvosBatchClient
from protocol import prompt, parse_label, validate_cases
from analysis import summarize

HERE = Path(__file__).resolve().parent
MODEL = "rwkv7-g1j-7.2b-20260831-ctx16384"
BASE_URL = "http://127.0.0.1:18425/v1"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


async def health():
    async with httpx.AsyncClient(timeout=10) as client:
        reply = await client.get(BASE_URL.removesuffix("/v1") + "/health")
        reply.raise_for_status()
        value = reply.json()
    if not value.get("ready") or value.get("model") != MODEL or value.get("state_sha256") is not None:
        raise ValueError("model_or_zero_state_identity_mismatch")
    return value


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    cases = [json.loads(line) for line in (HERE / "cases.jsonl").read_text().splitlines()]
    validate_cases(cases)
    schedule = json.loads((HERE / "schedule.json").read_text())
    by_id = {c["id"]: c for c in cases}
    assert len(schedule) == len(cases) * 4
    assert {(s["round"], s["case_id"], s["arm"]) for s in schedule} == {
        (r, c["id"], a) for r in (1, 2) for c in cases for a in ("yes-no", "neutral-labels")}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "calls").mkdir()
    (args.output / "http").mkdir()
    bound_files = [*sorted(HERE.glob("*.py")), HERE / "cases.jsonl", HERE / "schedule.json", HERE / "README.md",
        HERE.parent / "reader-label-20260920/protocol.py", HERE.parent / "evidence-support-20260920/protocol.py",
        HERE.parents[1] / "src/llamaindex_retrieval/reader_prompt.py",
        HERE.parents[1] / "src/llamaindex_retrieval/rwkvos_batch.py"]
    bindings = {str(p): sha256(p.read_bytes()).hexdigest() for p in bound_files}
    save(args.output / "BINDINGS.json", {"files": bindings,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_at": datetime.now(timezone.utc).isoformat(), "model": MODEL, "state": None,
        "checkpoint_sha256_from_verified_prior_deployment": "e3091a579c23ea7ebce9a0ad1ecfbda27082eeecd64d7f0474016e626df8f9c3",
        "temperature": 0, "max_tokens": 32, "input_limit": 4096, "stop_tokens": [0],
        "unique_cases": len(cases), "scheduled_calls": len(schedule), "independent_label_review": False,
        "sole_variable": "output label words and necessary output declarations; data bytes unchanged"})
    save(args.output / "HEALTH-BEFORE.json", await health())

    async def recorder(event, receipt):
        save(args.output / "http" / (uuid4().hex + ".json"), {"event": event, "receipt": receipt})

    client = RwkvosBatchClient(base_url=BASE_URL, model=MODEL, state_id=None, reader_state_id=None,
        stop_tokens=[0], reader_prompt_protocol="rwkv_g1j_no_think_v1", reader_input_layout="original",
        prefill_mode="complete", count_input_tokens=True, input_token_limit=4096,
        max_concurrency=1, batch_size=1, timeout_seconds=60, recorder=recorder)
    rows = []
    try:
        for i, item in enumerate(schedule):
            case = by_id[item["case_id"]]
            exact_prompt = prompt(case, item["arm"])
            trace = {"prompt": exact_prompt, "prompt_sha256": sha256(exact_prompt.encode()).hexdigest(),
                     "source_sha256": sha256(case["text"].encode()).hexdigest()}
            row = {**item, "pair": case["pair"], "family": case["family"], "category": case["category"],
                   "expected": case["expected"], "prediction": None, "status": "invalid", "error": None}
            began = perf_counter()
            try:
                async with asyncio.timeout(75):
                    result = await client.complete([{"role": "user", "content": exact_prompt}],
                        stage="resolver", max_tokens=32, temperature=0, assistant_prefill="<think></think>",
                        evidence_ids=[case["id"]], trace=trace)
                bounds = model_answer_bounds(result.raw_text, result.trace)
                if result.status != "completed" or result.trace.get("provider_finish_reason") != "stop" or not bounds:
                    raise ValueError("model_call_not_completed")
                row["prediction"] = parse_label(result.raw_text[bounds[0]:bounds[1]], item["arm"])
                row["status"] = "valid"
            except Exception as error:
                row["error"] = {"type": type(error).__name__, "detail": str(error)}
            finally:
                row["elapsed_ms"] = round((perf_counter() - began) * 1000, 2)
                save(args.output / "calls" / f"{i:04d}.json", {"case": case, "result": row, "trace": trace})
            rows.append(row)
            if (i + 1) % 40 == 0:
                save(args.output / f"HEALTH-{i + 1:04d}.json", await health())
                print(json.dumps({"completed_calls": i + 1, "total_calls": len(schedule), "round": item["round"]}), flush=True)
    finally:
        await client.aclose()
        save(args.output / "ROWS.json", rows)
        save(args.output / "EXECUTION.json", {"complete": len(rows) == len(schedule), "completed_calls": len(rows)})
    assert all(sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in bindings.items()), "frozen files changed"
    report = summarize(rows, cases)
    save(args.output / "SUMMARY.json", report)
    print(json.dumps({"complete": True, "limited_experimental_improvement_gate": report["limited_experimental_improvement_gate"]}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
