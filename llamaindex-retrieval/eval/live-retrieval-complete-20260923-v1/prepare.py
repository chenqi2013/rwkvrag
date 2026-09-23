"""Freeze the complete live-retrieval question manifest without editing sources."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
OUTPUT = HERE / "INPUTS.json"
SOURCES = [
    ("live-retrieval-20260923-v1", "cases.json"),
    ("live-retrieval-20260923-v2", "cases.json"),
    ("github-natural-comparison-20260921", "CASES.json"),
    ("github-project-comparison-20260921", "CASES.json"),
    ("live-comparison-20260921", "CASES.json"),
    ("live-retrieval-complete-20260923-v1", "NEW-CASES.json"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite frozen manifest: {OUTPUT}")
    result = {"protocol": "live-retrieval-complete-20260923-v1", "sources": {}, "cases": []}
    for source, filename in SOURCES:
        path = EVAL / source / filename
        rows = json.loads(path.read_text(encoding="utf-8"))
        result["sources"][str(path.relative_to(EVAL.parent))] = sha(path)
        for row in rows:
            case_id = row["id"]
            if source == "live-retrieval-20260923-v1":
                mode = "web"
            elif source == "live-retrieval-20260923-v2":
                mode = "web"
            else:
                mode = row.get("retrieval_mode", "web")
            result["cases"].append({
                "ordinal": len(result["cases"]),
                "uid": f"{source}:{case_id}",
                "source_suite": source,
                "source_id": case_id,
                "question": row["question"],
                "retrieval_mode": mode,
                "knowledge_base_id": row.get("knowledge_base_id"),
                "parent_uid": f"{source}:{row['parent_id']}" if row.get("parent_id") else None,
                "history": row.get("history", []),
                "kind": row.get("kind"),
                "objects": row.get("objects", row.get("projects")),
                "rubric": row.get("rubric"),
            })
    assert len(result["cases"]) == 65
    assert len({row["uid"] for row in result["cases"]}) == 65
    assert [sum(x["source_suite"] == s for x in result["cases"]) for s, _ in SOURCES] == [11, 6, 8, 8, 12, 20]
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT, len(result["cases"]))


if __name__ == "__main__":
    main()
