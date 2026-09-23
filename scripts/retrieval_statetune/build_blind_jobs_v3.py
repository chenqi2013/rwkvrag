"""Register heldout jobs only in four directly comparable project cohorts."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from build_curated_jobs import build


COHORTS = {"note_apps", "uptime_monitors", "api_clients", "terminal_emulators"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.sources.read_bytes()
    data = json.loads(raw)
    if data.get("schema") != "retrieval_curated_sources_v3":
        raise ValueError("source schema mismatch")
    selected = [row for row in data["sources"] if row["split"] == "heldout"
                and row["cohort"] in COHORTS]
    if {row["cohort"] for row in selected} != COHORTS:
        raise ValueError("one blind cohort has no source")
    jobs = build(selected, "heldout", 0, 20260924)
    if len(jobs) != 16:
        raise ValueError("expected exactly sixteen blind jobs")
    with args.out.open("x", encoding="utf-8") as destination:
        for job in jobs:
            destination.write(json.dumps(job, ensure_ascii=False) + "\n")
    print(json.dumps({"jobs": len(jobs), "planned_rows": len(jobs) * 8,
                      "source_sha256": sha256(raw).hexdigest(),
                      "jobs_sha256": sha256(args.out.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
