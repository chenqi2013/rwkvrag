"""Serial project-specific retrieval and one full-answer observation."""

import hashlib
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[3]
CASES = Path(__file__).with_name("cases.json")
OUTPUT = ROOT / "artifacts/live-retrieval-20260923-v2/RUN-v2.json"
SETTINGS = ROOT / "data/services/local-app/settings.json"
URL = "http://127.0.0.1:18440"
COMPARISON = "vLLM、SGLang 和 llama.cpp 在部署方式、硬件需求和推理优化上分别适合什么场景？"


def post(path, body, timeout=90):
    started = time.monotonic()
    request = Request(URL + path, json.dumps(body, ensure_ascii=False).encode(),
                      {"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(16 * 1024 * 1024 + 1)
            if len(raw) > 16 * 1024 * 1024:
                raise ValueError("response_too_large")
            status = response.status
        return {"http_status": status, "elapsed_ms": round((time.monotonic() - started) * 1000),
                "response": json.loads(raw)}
    except (HTTPError, URLError, TimeoutError, ValueError) as error:
        return {"http_status": getattr(error, "code", None),
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "error_type": type(error).__name__}


def main():
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite frozen run: {OUTPUT}")
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    result = {
        "protocol": "live-retrieval-20260923-v2",
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
        result["calls"].append({"case_id": case["id"], "mode": "web",
            **post("/v1/search", {"question": case["question"], "retrieval_mode": "web", "top_k": 20})})
        print("web", case["id"], flush=True)
        time.sleep(1)
    result["calls"].append({"case_id": "compare_inference_3", "mode": "ask_web",
        **post("/v1/ask", {"question": COMPARISON, "retrieval_mode": "web"}, timeout=240)})
    print("ask_web compare_inference_3", flush=True)
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
