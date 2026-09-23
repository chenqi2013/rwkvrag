"""Snapshot exposed evaluation questions and known source identities as hashes.

The output is an exclusion registry, never a training dataset. Existing frozen
files are read only. New evaluations must be added before data admission.
"""

from hashlib import sha256
import json
from pathlib import Path

from audit_candidates import digest, normalize

ROOT = Path(__file__).resolve().parents[2]
FILES = {
    "live": ROOT / "llamaindex-retrieval/eval/live-retrieval-complete-20260923-v1/INPUTS.json",
    "fresh": ROOT / "llamaindex-retrieval/eval/state-fresh-github-20260922-v1/CASES.jsonl",
    "old434": ROOT / "artifacts/broad-regression-20260920/restored-retrieval-v2/ANSWERS.json",
    "fresh_sources": ROOT / "llamaindex-retrieval/eval/state-fresh-github-20260922-v1/MANIFEST.json",
    "prior_engineering_sources": ROOT / "llamaindex-retrieval/statetune/progression-engineering-v4-20260922/MANIFEST.json",
}
OUT = ROOT / "llamaindex-retrieval/statetune/retrieval-v1-20260923/EXCLUSIONS.json"


def records(path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    live = records(FILES["live"])["cases"]
    fresh = records(FILES["fresh"])
    old = records(FILES["old434"])
    questions = {digest(normalize(row["question"])) for row in [*live, *fresh, *old]}
    source_rows = [*records(FILES["fresh_sources"]), *records(FILES["prior_engineering_sources"])]
    families = {"finewiki-zh"}
    sources = set()
    for row in source_rows:
        families.add("repo:" + row["repo"].casefold())
        sources.add(row["sha256"])
        if row.get("snippet_sha256"):
            sources.add(row["snippet_sha256"])
    result = {
        "schema": "retrieval_state_exclusions_v1",
        "inputs": {name: sha256(path.read_bytes()).hexdigest() for name, path in FILES.items()},
        "question_hashes": sorted(questions),
        "source_families": sorted(families),
        "source_hashes": sorted(sources),
        "limits": "Exact normalized questions and named source snapshots only; manual alias, mirror and topic review remains required."
    }
    with OUT.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({"question_hashes": len(questions), "source_families": len(families),
                      "source_hashes": len(sources)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
