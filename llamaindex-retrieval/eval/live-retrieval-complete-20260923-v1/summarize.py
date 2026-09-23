"""Summarize the immutable live replay without treating format checks as correctness."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
INPUTS = HERE / "INPUTS.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def summarize_case(case, directory):
    ordinal = case["ordinal"]
    search_path = directory / f"{ordinal:02d}.search.json"
    ask_path = directory / f"{ordinal:02d}.ask.json"
    if not search_path.exists() or not ask_path.exists():
        return {"uid": case["uid"], "ordinal": ordinal, "missing_record": True}
    search, ask = read(search_path), read(ask_path)
    search_data, ask_data = search.get("response") or {}, ask.get("response") or {}
    retrieval, generation = ask_data.get("retrieval") or {}, ask_data.get("generation") or {}
    search_retrieval = search_data.get("retrieval") or {}
    sources = ask_data.get("sources") or []
    answer = ask_data.get("answer") or ""
    audit = generation.get("citation_audit") or {}
    queried = retrieval.get("web_search") or []
    search_queried = search_retrieval.get("web_search") or []
    official_hint = sum("github.com/" in (r.get("uri") or "") or
                        "readthedocs.io" in (r.get("uri") or "") for r in search_data.get("results") or [])
    return {
        "uid": case["uid"], "ordinal": ordinal, "suite": case["source_suite"],
        "kind": case["kind"], "mode": case["retrieval_mode"],
        "objects": case.get("objects") or [], "search_http": search.get("http_status"),
        "ask_http": ask.get("http_status"), "search_ms": search.get("elapsed_ms"),
        "ask_ms": ask.get("elapsed_ms"), "search_candidates": len(search_data.get("results") or []),
        "search_web_calls": len(search_queried),
        "search_web_failures": sum(q.get("status") == "failed" for q in search_queried),
        "search_official_domain_hints": official_hint,
        "ask_web_calls": len(queried),
        "ask_web_failures": sum(q.get("status") == "failed" for q in queried),
        "ask_web_executed": sum(q.get("status") == "completed" for q in queried),
        "ask_web_skipped_query_budget": sum(q.get("status") == "skipped_query_budget" for q in queried),
        "planner_query_count": len((retrieval.get("plan") or {}).get("queries") or []),
        "source_count": len(sources),
        "source_types": dict(Counter(s.get("source") or "unknown" for s in sources)),
        "snippet_only_sources": sum((s.get("metadata") or {}).get("content_status") == "snippet_only" for s in sources),
        "generation_status": generation.get("status") or "none",
        "planner_fallback": bool(generation.get("planner_fallback")),
        "answer_characters": len(answer), "answer_empty": not bool(answer.strip()),
        "answer_mentions_objects": sum(str(obj).lower() in answer.lower() for obj in (case.get("objects") or [])),
        "answer_object_count": len(case.get("objects") or []),
        "cited_label_count": len(audit.get("label_ids") or []),
        "unknown_citation_ids": audit.get("unknown_label_ids") or [],
        "invalid_citations": audit.get("invalid_labels") or [],
        "semantic_support_verified": False,
        "raw_sha256": {"search": digest(search_path), "ask": digest(ask_path)},
    }


def group(rows):
    result = {}
    for dimension in ("suite", "kind", "mode"):
        buckets = defaultdict(list)
        for row in rows:
            buckets[row.get(dimension, "missing")].append(row)
        result[dimension] = {
            name: {
                "n": len(items),
                "search_with_candidates": sum(r.get("search_candidates", 0) > 0 for r in items),
                "search_with_web_failure": sum(r.get("search_web_failures", 0) > 0 for r in items),
                "ask_with_web_failure": sum(r.get("ask_web_failures", 0) > 0 for r in items),
                "ask_with_query_budget_skip": sum(r.get("ask_web_skipped_query_budget", 0) > 0 for r in items),
                "ask_with_sources": sum(r.get("source_count", 0) > 0 for r in items),
                "ask_with_unknown_citation": sum(bool(r.get("unknown_citation_ids")) for r in items),
                "ask_with_all_object_names": sum(
                    r.get("answer_object_count", 0) > 0 and
                    r.get("answer_mentions_objects") == r.get("answer_object_count") for r in items),
                "generation_statuses": dict(Counter(r.get("generation_status", "none") for r in items)),
            } for name, items in buckets.items()
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    inputs = read(INPUTS)
    rows = [summarize_case(case, args.run) for case in inputs["cases"]]
    report = {
        "protocol": inputs["protocol"], "inputs_sha256": digest(INPUTS),
        "run_binding_sha256": digest(args.run / "BINDING.json"),
        "planned": len(rows), "recorded_pairs": sum(not r.get("missing_record") for r in rows),
        "groups": group(rows), "rows": rows,
        "limits": ["Official-domain hints do not establish relevance or primary-source support.",
                   "Object-name mentions do not establish comparison coverage.",
                   "Citation numbering does not establish semantic support.",
                   "No automated semantic-accuracy score is claimed."],
    }
    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
