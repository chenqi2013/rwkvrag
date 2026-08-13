import json
import urllib.request
from collections import Counter
from pathlib import Path


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def evaluate(url: str, cases_path: Path, top_k: int) -> dict:
    cases = load_cases(cases_path)
    details = []
    passed = 0
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
        expected_titles = case.get("expected_titles", [])
        expected_titles_all = case.get("expected_titles_all", [])
        matched = (
            all(title in titles for title in expected_titles_all)
            if expected_titles_all
            else any(title in expected_titles for title in titles)
        )
        passed += int(matched)
        details.append(
            {
                "question": case["question"],
                "passed": matched,
                "titles": titles,
                "expected_titles": case["expected_titles"],
                "case_type": case.get("case_type", "unspecified"),
            }
        )
    type_totals = Counter(case.get("case_type", "unspecified") for case in cases)
    type_passed = Counter(detail["case_type"] for detail in details if detail["passed"])
    return {
        "passed": passed,
        "total": len(cases),
        "recall_at_k": passed / len(cases) if cases else 0,
        "by_type": {
            case_type: {
                "passed": type_passed[case_type],
                "total": total,
                "recall_at_k": type_passed[case_type] / total,
            }
            for case_type, total in sorted(type_totals.items())
        },
        "details": details,
    }
