"""Compare fresh restored-index answers with the frozen 2026-09-20 raw run."""

import argparse
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[3]
PRIOR = ROOT / "artifacts/broad-regression-20260920/restored-retrieval-v2"
N = 434


def response(row):
    try:
        return json.loads(row.get("raw_response") or "{}")
    except json.JSONDecodeError:
        return {}


def identity(sources):
    return [(item.get("id"), item.get("document_id"), item.get("title"), item.get("snippet"))
            for item in sources or []]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prior_binding = json.loads((PRIOR / "BINDINGS.json").read_text(encoding="utf-8"))
    fresh_binding = json.loads((args.run / "BINDINGS.json").read_text(encoding="utf-8"))
    binding_match = {
        key: prior_binding.get(key) == fresh_binding.get(key)
        for key in ("settings_sha256", "files", "origins")
    }
    rows = []
    with tarfile.open(PRIOR / "raw-calls.tar.gz", "r:gz") as archive:
        for index in range(N):
            path = args.run / "calls" / f"{index:04d}.json"
            if not path.exists():
                continue
            previous = json.load(archive.extractfile(f"calls/{index:04d}.json"))
            current = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("case_id") != current.get("case_id"):
                raise ValueError(f"case order changed at {index}")
            a, b = response(previous), response(current)
            rows.append({
                "index": index, "case_id": current["case_id"],
                "request_equal": previous.get("payload") == current.get("payload"),
                "http_equal": previous.get("http_status") == current.get("http_status"),
                "generation_status_equal": (previous.get("diagnostics") or {}).get("generation_status") ==
                                           (current.get("diagnostics") or {}).get("generation_status"),
                "answer_equal": a.get("answer") == b.get("answer"),
                "final_sources_equal": identity(a.get("sources")) == identity(b.get("sources")),
                "previous_elapsed_ms": previous.get("elapsed_ms"),
                "current_elapsed_ms": current.get("elapsed_ms"),
            })
    report = {"planned": N, "compared": len(rows), "binding_match": binding_match,
              "equal_counts": {key: sum(row[key] for row in rows) for key in
                               ("request_equal", "http_equal", "generation_status_equal",
                                "answer_equal", "final_sources_equal")},
              "mismatches": {key: [row["case_id"] for row in rows if not row[key]] for key in
                             ("request_equal", "http_equal", "generation_status_equal",
                              "answer_equal", "final_sources_equal")},
              "limits": ["Equality on completed members does not replace the remaining fresh requests.",
                         "Matching answer text does not prove semantic correctness."],
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
