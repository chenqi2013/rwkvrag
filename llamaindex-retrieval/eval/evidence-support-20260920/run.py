"""Fixed, labelled support benchmark. Never rewrites or retries a model answer."""
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
from llamaindex_retrieval.reader_prompt import parse_binary_decision
from llamaindex_retrieval.rwkvos_batch import RwkvosBatchClient
from protocol import metrics, prompt, validate_cases


HERE = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, default=HERE / "profiles.json")
    args = parser.parse_args()
    cases = [json.loads(line) for line in (HERE / "cases.jsonl").read_text().splitlines()]
    validate_cases(cases)
    profiles = json.loads(args.profiles.read_text())
    assert len({p["id"] for p in profiles}) == len(profiles)
    assert all(p["id"].replace("-", "").isalnum() for p in profiles)
    args.output.mkdir(parents=True, exist_ok=False)
    bound = [HERE / "cases.jsonl", HERE / "protocol.py", Path(__file__).resolve(), args.profiles,
             HERE.parents[1] / "src/llamaindex_retrieval/reader_prompt.py",
             HERE.parents[1] / "src/llamaindex_retrieval/rwkvos_batch.py"]
    bindings = {str(p): sha256(p.read_bytes()).hexdigest() for p in bound}
    save(args.output / "BINDINGS.json", {"files": bindings,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "created_at": datetime.now(timezone.utc).isoformat(), "profiles": profiles,
        "index": "fixed-material-no-retrieval", "temperature": 0, "max_tokens": 32,
        "gate": {"invalid": 0, "false_positives": 0, "min_recall": .9, "min_pair_accuracy": .9},
        "annotation": "synthetic agent-authored labels; not independently adjudicated"})
    results = []
    for profile in profiles:
        folder = args.output / profile["id"]
        folder.mkdir()
        async with httpx.AsyncClient(timeout=10) as http:
            response = await http.get(profile["base_url"].removesuffix("/v1") + "/health")
            response.raise_for_status()
            health = response.json()
        save(folder / "HEALTH.json", health)
        assert health.get("ready") and health.get("model") == profile["model"], "model identity mismatch"
        if profile["state"]:
            assert any(s["id"] == profile["state"] and s["stage"] == "resolver"
                       for s in health.get("available_states", [])), "Reader State not available on this checkpoint"

        async def recorder(event, receipt):
            save(folder / (uuid4().hex + ".http.json"), {"event": event, "receipt": receipt})

        client = RwkvosBatchClient(base_url=profile["base_url"], model=profile["model"],
            state_id=None, reader_state_id=profile["state"], stop_tokens=[0],
            reader_prompt_protocol="rwkv_g1j_no_think_v1", reader_input_layout="original",
            prefill_mode="complete", count_input_tokens=True, input_token_limit=4096,
            max_concurrency=1, batch_size=1, timeout_seconds=60, recorder=recorder)
        rows = []
        try:
            # Both splits and all profiles are frozen before any model is run.
            for split in ["development", "validation"]:
                for case in [c for c in cases if c["split"] == split]:
                    exact_prompt = prompt(case, profile["protocol"])
                    trace = {"prompt": exact_prompt, "prompt_sha256": sha256(exact_prompt.encode()).hexdigest(),
                             "source_sha256": sha256(case["text"].encode()).hexdigest()}
                    row = {"case_id": case["id"], "pair": case["pair"], "split": split,
                           "category": case["category"], "expected": case["expected"],
                           "prediction": None, "status": "invalid", "error": None}
                    started = perf_counter()
                    try:
                        async with asyncio.timeout(75):
                            reply = await client.complete([{"role": "user", "content": exact_prompt}],
                                stage="resolver", max_tokens=32, temperature=0,
                                assistant_prefill="<think></think>", evidence_ids=[case["id"]], trace=trace)
                        bounds = model_answer_bounds(reply.raw_text, reply.trace)
                        if reply.status != "completed" or reply.trace.get("provider_finish_reason") != "stop" or not bounds:
                            raise ValueError("model_call_not_completed")
                        row["prediction"] = parse_binary_decision(reply.raw_text[bounds[0]:bounds[1]])
                        row["status"] = "valid"
                    except Exception as error:
                        row["error"] = {"type": type(error).__name__, "detail": str(error)}
                    finally:
                        row["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
                        save(folder / (case["id"] + ".json"), {"case": case, "result": row, "trace": trace})
                    rows.append(row)
                print(json.dumps({"profile": profile["id"], "split": split,
                                  **metrics([r for r in rows if r["split"] == split])}), flush=True)
        finally:
            await client.aclose()
            save(folder / "ROWS.json", rows)
            report = {"profile": profile, "expected_cases": len(cases),
                "execution_status": "completed" if len(rows) == len(cases) else "incomplete",
                "overall": metrics(rows),
                "splits": {s: metrics([r for r in rows if r["split"] == s]) for s in ["development", "validation"]},
                "categories": {c: metrics([r for r in rows if r["category"] == c])
                               for c in sorted({r["category"] for r in rows})}}
            save(folder / "SUMMARY.json", report)
            results.append(report)
            save(args.output / "SUMMARY.json", results)
    assert all(sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in bindings.items()), "experiment files changed during run"


if __name__ == "__main__":
    asyncio.run(main())
