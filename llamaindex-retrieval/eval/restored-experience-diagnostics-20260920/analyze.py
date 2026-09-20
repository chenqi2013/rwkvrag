"""Post-run observations of waiting, work volume and incomplete answers.

No semantic scoring, model calls, answer editing or changes to frozen inputs.
Concurrent model call timings overlap; never add them into request wall time.
"""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "data/quality-runs/restored-retrieval-v2-20260920/run1"
OUT = ROOT / "artifacts/broad-regression-20260920/restored-retrieval-v2"


def distribution(values):
    values = sorted(float(v) for v in values if isinstance(v, (int, float)) and isfinite(v))
    if not values:
        return {"n": 0}
    def percentile(p):
        position = (len(values) - 1) * p
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return round(values[lower] + (values[upper] - values[lower]) * (position - lower), 2)
    return {"n": len(values), "min": values[0], "p50": percentile(.5), "p90": percentile(.9),
            "p95": percentile(.95), "max": values[-1]}


def main():
    rows = json.loads((RUN / "ROWS.json").read_text())
    assert len(rows) == len({r["case_id"] for r in rows}) == 434
    assert OUT.exists(), "run the full evidence export first"
    records, timing = [], defaultdict(lambda: defaultdict(list))
    for row in rows:
        data = json.loads(row.get("raw_response") or "{}")
        calls = data.get("generation", {}).get("model_calls", [])
        by_stage = Counter(c.get("stage", "unspecified") for c in calls)
        reader_groups = defaultdict(set)
        repeated_inputs = Counter()
        writers = []
        for call in calls:
            stage = call.get("stage", "unspecified")
            for key in ("queue_ms", "active_ms", "elapsed_ms"):
                timing[stage][key].append(call.get(key))
            if stage == "resolver":
                unit = json.dumps([call.get("source_id"), call.get("source_sha256"), call.get("units")],
                                  sort_keys=True, ensure_ascii=False)
                reader_groups[unit].add(json.dumps(call.get("task_group"), ensure_ascii=False))
            if call.get("prompt_sha256"):
                signature = json.dumps([stage, call.get("state_id"), call.get("state_role"),
                    call["prompt_sha256"], call.get("parameters"), call.get("budget", {}).get("reserved_output_tokens")],
                    sort_keys=True, ensure_ascii=False)
                repeated_inputs[signature] += 1
            if stage == "writer":
                budget = call.get("budget", {})
                writers.append({"call_id": call.get("call_id"), "status": call.get("status"),
                    "input_tokens": budget.get("input_tokens"),
                    "reserved_output_tokens": budget.get("reserved_output_tokens"),
                    "configured_context_window": budget.get("configured_context_window"),
                    "server_context_window": budget.get("server_context_window"),
                    "budget_verification": budget.get("verification"),
                    "application_policy_fits": budget.get("application_policy_fits"),
                    "active_ms": call.get("active_ms"), "queue_ms": call.get("queue_ms")})
        records.append({"case_id": row["case_id"], "suite": row["suite"], "category": row["category"],
            "http_status": row.get("http_status"), "elapsed_ms": row["elapsed_ms"],
            "generation_status": data.get("generation", {}).get("status"),
            "selected_sources": len(data.get("sources", [])), "model_calls": dict(by_stage),
            "same_unit_multiple_reader_task_groups": sum(len(groups) > 1 for groups in reader_groups.values()),
            "same_call_input_repetitions": sum(n - 1 for n in repeated_inputs.values()),
            "writers": writers, "raw_file": f"calls/{row['index']:04d}.json"})

    def summarize(items):
        completed_http = [r for r in items if r["http_status"] == 200]
        return {"n": len(items), "http_statuses": dict(Counter(str(r["http_status"]) for r in items)),
            "request_wall_ms_http200": distribution(r["elapsed_ms"] for r in completed_http),
            "request_wall_ms_http_failure": distribution(r["elapsed_ms"] for r in items if r["http_status"] != 200),
            "generation_statuses": dict(Counter(str(r["generation_status"]) for r in items)),
            "selected_sources": distribution(r["selected_sources"] for r in completed_http),
            "native_call_count": distribution(sum(r["model_calls"].values()) for r in completed_http),
            "reader_call_count": distribution(r["model_calls"].get("resolver", 0) for r in completed_http),
            "same_unit_multiple_reader_task_groups": distribution(r["same_unit_multiple_reader_task_groups"] for r in completed_http),
            "same_call_input_repetitions": sum(r["same_call_input_repetitions"] for r in items)}

    report = {"scope": "434 actual-retrieval requests, frozen 4-way concurrency; not interactive latency SLA or semantic accuracy",
        "limits": ["call timings overlap and are not summed into request duration",
                   "same input and multiple task-group counts describe work, not proof that it can safely be removed",
                   "configured context window is not independent verification of the provider window",
                   "length termination is not by itself a semantic repetition judgment"],
        "overall": summarize(records),
        "by_suite": {suite: summarize([r for r in records if r["suite"] == suite]) for suite in sorted({r["suite"] for r in records})},
        "native_call_timing_ms": {stage: {key: distribution(values) for key, values in grouped.items()}
                                  for stage, grouped in timing.items()},
        "rows_sha256": sha256((RUN / "ROWS.json").read_bytes()).hexdigest(), "cases": records}
    destination = OUT / "EXPERIENCE-DIAGNOSTICS.json"
    with destination.open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
