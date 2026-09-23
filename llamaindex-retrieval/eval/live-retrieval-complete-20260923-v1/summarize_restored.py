"""Compact, non-semantic diagnostics for the new 434-case restored-index replay."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "restored-retrieval-v2-20260920"
CASES = json.loads((OLD / "cases.json").read_text(encoding="utf-8"))


def digest(data):
    return sha256(data).hexdigest()


def reviewed_ids():
    groups = defaultdict(list)
    for case in CASES:
        groups[case["suite"]].append(case["id"])
    return {name: sorted(ids, key=lambda case_id: digest(("review-20260923:" + case_id).encode()))[:5]
            for name, ids in sorted(groups.items())}


def row_for(index, case, run):
    path = run / "calls" / f"{index:04d}.json"
    if not path.exists():
        return {"index": index, "case_id": case["id"], "suite": case["suite"],
                "category": case["category"], "style": case["style"], "recorded": False}
    raw = path.read_bytes()
    record = json.loads(raw)
    info = record.get("diagnostics") or {}
    answer = ""
    if record.get("raw_response"):
        try:
            answer = json.loads(record["raw_response"]).get("answer") or ""
        except (json.JSONDecodeError, TypeError):
            pass
    return {
        "index": index, "case_id": case["id"], "suite": case["suite"],
        "category": case["category"], "style": case["style"], "recorded": True,
        "http_status": record.get("http_status"), "completed": record.get("completed"),
        "elapsed_ms": record.get("elapsed_ms"), "generation_status": info.get("generation_status"),
        "source_count": info.get("source_count"), "answer_characters": len(answer),
        "empty_answer": info.get("empty_answer"),
        "citation_out_of_range": info.get("citation_out_of_range"),
        "has_numeric_citation": info.get("has_numeric_citation"),
        "restored_expected_document_returned": info.get("restored_expected_document_returned"),
        "reference_terms_matched": info.get("reference_terms_matched"),
        "reference_terms_total": info.get("reference_terms_total"),
        "semantic_accuracy_verified": False,
        "raw_sha256": digest(raw), "answer_sha256": digest(answer.encode()),
        "error": record.get("error"),
    }


def group(rows, dimension):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[dimension]].append(row)
    return {key: {
        "n": len(items), "recorded": sum(row["recorded"] for row in items),
        "http_completed": sum(row.get("completed") is True for row in items),
        "with_sources": sum((row.get("source_count") or 0) > 0 for row in items),
        "expected_document_in_final_sources": sum(row.get("restored_expected_document_returned") is True for row in items),
        "out_of_range_citations": sum(row.get("citation_out_of_range") is True for row in items),
        "generation_statuses": dict(Counter(row.get("generation_status") or "not_recorded" for row in items)),
    } for key, items in sorted(grouped.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [row_for(index, case, args.run) for index, case in enumerate(CASES)]
    elapsed = sorted(row["elapsed_ms"] for row in rows if row.get("elapsed_ms") is not None)
    report = {
        "input_cases_sha256": digest((OLD / "cases.json").read_bytes()),
        "run_binding_sha256": digest((args.run / "BINDINGS.json").read_bytes()),
        "planned": len(rows), "recorded": sum(row["recorded"] for row in rows),
        "groups": {dimension: group(rows, dimension) for dimension in ("suite", "category", "style")},
        "elapsed_median_ms": median(elapsed) if elapsed else None,
        "elapsed_p95_observed_ms": elapsed[int(0.95 * (len(elapsed) - 1))] if elapsed else None,
        "manual_sample_ids": reviewed_ids(), "rows": rows,
        "limits": ["Automated diagnostics do not verify semantic accuracy.",
                   "Expected article in final sources does not mean answer is correct.",
                   "Partial runs must not be interpreted as a completed 434-case regression."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
