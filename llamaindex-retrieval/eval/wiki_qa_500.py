#!/usr/bin/env python3
"""Build and run a reproducible, source-grounded FineWiki QA evaluation."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?])\s*")
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
ASCII_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
YEAR = re.compile(r"(?<!\d)(\d{3,4})年")
NOISE = re.compile(r"(?:thumb\||Category:|参考资料|外部链接|^\s*$)", re.I)
BAD_ANSWER_MARKERS = (
    "未检索到可用于回答",
    "根据检索到的资料，无法确定",
    "<think>",
    "[助手",
    "[assistant",
    "tool_call",
)


def request_json(url: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def clean_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" \n\t")


def useful_sentences(text: str, title: str) -> list[str]:
    sentences = []
    for sentence in SENTENCE_SPLIT.split(text[:5000]):
        sentence = clean_sentence(sentence)
        if not 18 <= len(sentence) <= 240 or NOISE.search(sentence):
            continue
        if title not in sentence:
            continue
        sentences.append(sentence)
    relation_markers = ("是", "为", "指", "位于", "成立于", "创建于", "出生于")
    return sorted(sentences, key=lambda value: (not any(marker in value for marker in relation_markers), len(value)))


def make_case(source: dict[str, Any]) -> dict[str, Any] | None:
    title = clean_sentence(str(source.get("title") or ""))
    text = str(source.get("text") or "")
    if not 2 <= len(title) <= 48:
        return None
    sentences = useful_sentences(text, title)
    if not sentences:
        return None
    sentence = sentences[0]
    question_type = "definition"
    if "出生于" in sentence:
        question = f"{title}出生于哪里？"
        question_type = "birthplace"
    elif "位于" in sentence:
        question = f"{title}位于哪里？"
        question_type = "location"
    elif "成立于" in sentence or "创建于" in sentence:
        marker = "成立" if "成立于" in sentence else "创建"
        question = f"{title}是哪一年{marker}的？"
        question_type = "year"
    elif "逝世于" in sentence:
        question = f"{title}逝世于哪一年？"
        question_type = "year"
    elif "是" in sentence or "为" in sentence or "指" in sentence:
        variants = (f"{title}是什么？", f"请简要介绍{title}。", f"{title}指的是什么？")
        question = variants[int(hashlib.md5(title.encode()).hexdigest(), 16) % len(variants)]
    else:
        question = f"请简要介绍{title}。"
    return {
        "question": question,
        "question_type": question_type,
        "expected_title": title,
        "expected_document_id": source.get("document_id"),
        "reference": sentence,
        "uri": source.get("uri"),
    }


def sample_cases(opensearch_url: str, index: str, count: int, seed: int) -> list[dict[str, Any]]:
    payload = {
        "size": min(count * 4, 5000),
        "query": {
            "function_score": {
                "query": {"term": {"source": "finewiki-zh"}},
                "random_score": {"seed": seed, "field": "_seq_no"},
            }
        },
        "collapse": {
            "field": "document_id",
            "inner_hits": {"name": "first_chunk", "size": 1, "sort": [{"_seq_no": "asc"}]},
        },
        "_source": ["title", "text", "document_id", "uri"],
    }
    result = request_json(f"{opensearch_url.rstrip('/')}/{index}/_search", payload)
    cases: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for hit in result.get("hits", {}).get("hits", []):
        inner = hit.get("inner_hits", {}).get("first_chunk", {}).get("hits", {}).get("hits", [])
        source = inner[0].get("_source", {}) if inner else hit.get("_source", {})
        case = make_case(source)
        if case and case["question"] not in seen_questions:
            cases.append(case)
            seen_questions.add(case["question"])
        if len(cases) >= count:
            break
    if len(cases) < count:
        raise RuntimeError(f"only generated {len(cases)} valid cases; requested {count}")
    return cases


def tokens(text: str) -> set[str]:
    normalized = "".join(CJK.findall(text))
    cjk_bigrams = {normalized[index:index + 2] for index in range(max(0, len(normalized) - 1))}
    return cjk_bigrams | {token.lower() for token in ASCII_TOKEN.findall(text)}


def score_case(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    answer = str(response.get("answer") or "")
    sources = response.get("sources") or []
    source_ids = [source.get("document_id") for source in sources]
    expected_tokens = tokens(case["reference"])
    answer_without_citations = re.sub(r"\[资料\s*\d+\]", "", answer)
    answer_tokens = tokens(answer_without_citations)
    evidence_tokens = tokens(" ".join(str(source.get("snippet") or "") for source in sources))
    reference_coverage = len(expected_tokens & answer_tokens) / max(1, min(len(expected_tokens), 12))
    unsupported_ratio = len(answer_tokens - evidence_tokens) / max(1, len(answer_tokens))
    answer_numbers = set(re.findall(r"\d+(?:\.\d+)?", answer_without_citations))
    evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?", " ".join(str(source.get("snippet") or "") for source in sources)))
    reasons = []
    if case["expected_document_id"] not in source_ids:
        reasons.append("retrieval_miss")
    if any(marker in answer for marker in BAD_ANSWER_MARKERS):
        reasons.append("refused_or_malformed")
    if not re.search(r"\[资料\s*[1-9]\d*\]", answer):
        reasons.append("missing_citation")
    if reference_coverage < 0.2:
        reasons.append("low_reference_coverage")
    if unsupported_ratio > 0.55 and len(answer_tokens) >= 10:
        reasons.append("weak_evidence_support")
    if answer_numbers - evidence_numbers:
        reasons.append("unsupported_number")
    if answer.startswith(">"):
        reasons.append("format_artifact")
    return {
        **case,
        "answer": answer,
        "sources": [{"title": source.get("title"), "document_id": source.get("document_id"), "score": source.get("score")} for source in sources],
        "retrieval": response.get("retrieval"),
        "generation": response.get("generation"),
        "reference_coverage": round(reference_coverage, 4),
        "unsupported_ratio": round(unsupported_ratio, 4),
        "passed": not reasons,
        "failure_reasons": reasons,
    }


def run_case(api_url: str, case: dict[str, Any], top_k: int) -> dict[str, Any]:
    started = time.monotonic()
    try:
        response = request_json(
            f"{api_url.rstrip('/')}/v1/ask",
            {"question": case["question"], "knowledge_base_id": "default", "top_k": top_k},
        )
        scored = score_case(case, response)
        scored["latency_seconds"] = round(time.monotonic() - started, 3)
        return scored
    except Exception as error:  # preserve failures in the report
        return {**case, "passed": False, "failure_reasons": ["request_error"], "error": str(error), "latency_seconds": round(time.monotonic() - started, 3)}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    failures = Counter(reason for result in results for reason in result.get("failure_reasons", []))
    types = Counter(result["question_type"] for result in results)
    passed = sum(bool(result.get("passed")) for result in results)
    latencies = sorted(float(result.get("latency_seconds", 0)) for result in results)
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / len(results), 4) if results else 0,
        "question_types": dict(types),
        "failure_reasons": dict(failures.most_common()),
        "latency_seconds": {
            "average": round(sum(latencies) / len(latencies), 3) if latencies else 0,
            "p50": latencies[len(latencies) // 2] if latencies else 0,
            "p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--api-url", default="http://127.0.0.1:8090")
    parser.add_argument("--opensearch-url", default="http://127.0.0.1:9200")
    parser.add_argument("--index", default="rwkvrag-knowledge-v1")
    parser.add_argument("--output", type=Path, default=Path("eval/wiki_qa_500_results.json"))
    args = parser.parse_args()

    cases = sample_cases(args.opensearch_url, args.index, args.count, args.seed)
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_case, args.api_url, case, args.top_k) for case in cases]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result())
            if index % 25 == 0:
                print(f"completed {index}/{len(cases)}", flush=True)
    results.sort(key=lambda item: item["question"])
    report = {"summary": summarize(results), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
