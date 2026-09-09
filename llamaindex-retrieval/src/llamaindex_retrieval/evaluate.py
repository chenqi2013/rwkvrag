import json
import re
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def score_titles(case: dict, titles: list[str]) -> tuple[bool, list[bool]]:
    groups = case.get("expected_title_groups")
    if not groups:
        expected_titles = case.get("expected_titles")
        groups = [expected_titles] if expected_titles else []
    matched_groups = [any(title in group for title in titles) for group in groups]
    return bool(matched_groups) and all(matched_groups), matched_groups


def _source_identity(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _phrase_present(phrase: str, snippet: str) -> bool:
    needle = "".join(unicodedata.normalize("NFKC", phrase).casefold().split())
    haystack = " ".join(unicodedata.normalize("NFKC", snippet).casefold().split())
    if not needle:
        return False
    left = r"(?<![a-z0-9_])" if needle[0].isascii() and needle[0].isalnum() else ""
    right = r"(?![a-z0-9_])" if needle[-1].isascii() and needle[-1].isalnum() else ""
    pattern = r"\s*".join(re.escape(character) for character in needle)
    return re.search(left + pattern + right, haystack) is not None


def _result_sources(result: dict) -> set[str]:
    metadata = result.get("metadata") or {}
    uri = result.get("uri") or ""
    values = [
        result.get("document_id"), result.get("title"), uri,
        Path(urllib.parse.unquote(urllib.parse.urlsplit(uri).path)).name,
        metadata.get("document_id"), metadata.get("file_id"), metadata.get("file"),
    ]
    return {
        _source_identity(value) for value in values
        if isinstance(value, str) and value.strip()
    }


def score_evidence(case: dict, results: list[dict]) -> tuple[bool | None, list[bool]]:
    groups = case.get("evidence_groups")
    if groups is None or groups == []:
        return None, []
    if not isinstance(groups, list) or not all(isinstance(group, dict) for group in groups):
        raise ValueError("evidence_groups must be a list of objects")
    aliases = case.get("source_aliases", {})
    matched_groups = []
    for group in groups:
        document_ids = group.get("document_ids") or []
        all_terms = group.get("must_include_all") or []
        any_terms = group.get("must_include_any") or []
        if not document_ids or not (all_terms or any_terms):
            raise ValueError("evidence groups require source document_ids and terms")
        allowed_sources = {
            _source_identity(value)
            for document_id in document_ids
            for value in [document_id, *aliases.get(document_id, [])]
        }
        matched = any(
            allowed_sources & _result_sources(result)
            and all(_phrase_present(term, result["snippet"]) for term in all_terms)
            and (not any_terms or any(_phrase_present(term, result["snippet"]) for term in any_terms))
            for result in results
        )
        matched_groups.append(matched)
    return all(matched_groups), matched_groups


def evaluate(url: str, cases_path: Path, top_k: int) -> dict:
    cases = load_cases(cases_path)
    details = []
    title_cases = title_hits = all_title_groups_hit = 0
    matched_title_groups = expected_title_groups = 0
    evidence_cases = all_evidence_groups_hit = 0
    matched_evidence_groups = expected_evidence_groups = 0
    for case in cases:
        payload = json.dumps(
            {"question": case["question"], "top_k": top_k},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            url.rstrip("/") + "/v1/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        titles = [item["title"] for item in result["results"]]
        if case.get("expected_titles_all") and not case.get("expected_title_groups"):
            case = {
                **case,
                "expected_title_groups": [
                    [title] for title in case["expected_titles_all"]
                ],
            }
        title_complete, title_groups = score_titles(case, titles)
        evidence_complete, evidence_groups = score_evidence(case, result["results"])
        if title_groups:
            title_cases += 1
            title_hits += int(any(title_groups))
            all_title_groups_hit += int(title_complete)
            matched_title_groups += sum(title_groups)
            expected_title_groups += len(title_groups)
        if evidence_complete is not None:
            evidence_cases += 1
            all_evidence_groups_hit += int(evidence_complete)
            matched_evidence_groups += sum(evidence_groups)
            expected_evidence_groups += len(evidence_groups)
        details.append(
            {
                "question": case["question"],
                "title_hit": any(title_groups) if title_groups else None,
                "all_title_groups_hit": title_complete if title_groups else None,
                "passed": title_complete if title_groups else False,
                "all_snippet_evidence_groups_hit": evidence_complete,
                "titles": titles,
                "expected_titles": case.get("expected_titles"),
                "expected_title_groups": case.get("expected_title_groups"),
                "matched_title_groups": title_groups,
                "matched_snippet_evidence_groups": evidence_groups,
                "case_type": case.get("case_type", "unspecified"),
            }
        )
    type_totals = Counter(case.get("case_type", "unspecified") for case in cases)
    type_passed = Counter(
        detail["case_type"] for detail in details if detail["all_title_groups_hit"]
    )
    return {
        "schema_version": "retrieval-evaluation.v2",
        "total": len(cases),
        # Backward-compatible aliases for existing CLI/report consumers.
        "passed": all_title_groups_hit,
        "recall_at_k": all_title_groups_hit / title_cases if title_cases else 0,
        "title_cases": title_cases,
        "title_hit_at_k": title_hits / title_cases if title_cases else None,
        "all_title_groups_hit_rate": (
            all_title_groups_hit / title_cases if title_cases else None
        ),
        "title_group_recall": (
            matched_title_groups / expected_title_groups if expected_title_groups else None
        ),
        "snippet_evidence_cases": evidence_cases,
        "all_snippet_evidence_groups_hit_rate": (
            all_evidence_groups_hit / evidence_cases if evidence_cases else None
        ),
        "snippet_evidence_group_coverage": (
            matched_evidence_groups / expected_evidence_groups
            if expected_evidence_groups else None
        ),
        "evidence_metric": (
            "Source-bound literal snippet matching; not semantic entailment or answer correctness."
        ),
        "by_type": {
            case_type: {
                "passed": type_passed[case_type],
                "total": total,
                "all_title_groups_hit_rate": type_passed[case_type] / total,
                "recall_at_k": type_passed[case_type] / total,
            }
            for case_type, total in sorted(type_totals.items())
        },
        "details": details,
    }
