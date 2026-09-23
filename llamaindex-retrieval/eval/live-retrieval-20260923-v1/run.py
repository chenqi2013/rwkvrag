"""Collect a frozen, serial read-only retrieval diagnostic from local services."""

import hashlib
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[3]
CASES = Path(__file__).with_name("cases.json")
OUTPUT = ROOT / "artifacts/live-retrieval-20260923-v1/RUN-v1.json"
SETTINGS = ROOT / "data/services/local-app/settings.json"
API = "http://127.0.0.1:18440"
SEARXNG = "http://127.0.0.1:18448"


def request(url, payload=None):
    body = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if body else {}
    started = time.monotonic()
    try:
        with urlopen(Request(url, body, headers), timeout=90) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError("response_too_large")
            status = response.status
        data = json.loads(raw)
        return {"http_status": status, "elapsed_ms": round((time.monotonic() - started) * 1000),
                "response": data}
    except (HTTPError, URLError, TimeoutError, ValueError) as error:
        return {"http_status": getattr(error, "code", None),
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "error_type": type(error).__name__}


def search(case, mode):
    return {"case_id": case["id"], "mode": mode,
            **request(API + "/v1/search", {"question": case["question"],
                "retrieval_mode": mode, "top_k": 20})}


def raw_searxng(case):
    raw = request(SEARXNG + "/search?" + urlencode({"q": case["question"], "format": "json"}))
    if isinstance(raw.get("response"), dict):
        response = raw["response"]
        raw["response"] = {"result_count": len(response.get("results", [])),
                           "top_results": [
                               {key: hit.get(key) for key in ("url", "title", "content", "engine", "score")}
                               for hit in response.get("results", [])[:10]],
                           "unresponsive_engines": response.get("unresponsive_engines", [])}
    return {"case_id": case["id"], "mode": "searxng_top10", **raw}


def main():
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite frozen run: {OUTPUT}")
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    run = {
        "protocol": "live-retrieval-20260923-v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "cases_sha256": hashlib.sha256(CASES.read_bytes()).hexdigest(),
        "settings_sha256": hashlib.sha256(SETTINGS.read_bytes()).hexdigest(),
        "index": settings.get("opensearch_index"),
        "web_search_provider": settings.get("web_search_provider"),
        "web_search_results": settings.get("web_search_results"),
        "web_search_max_queries": settings.get("web_search_max_queries"),
        "calls": [],
    }
    for case in cases:
        run["calls"].append(search(case, "web"))
        print("web", case["id"], flush=True)
        if case["kind"] == "compare":
            time.sleep(1)
            run["calls"].append(raw_searxng(case))
            print("searxng", case["id"], flush=True)
        time.sleep(1)
    by_id = {case["id"]: case for case in cases}
    for case_id in ("compare_inference_3", "compare_rag_3", "compare_storage_3", "single_repo"):
        run["calls"].append(search(by_id[case_id], "hybrid"))
        print("hybrid", case_id, flush=True)
        time.sleep(1)
    for case_id in ("compare_inference_3", "single_repo"):
        run["calls"].append(search(by_id[case_id], "auto"))
        print("auto", case_id, flush=True)
        time.sleep(1)
    run["finished_at"] = datetime.now(timezone.utc).isoformat()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
