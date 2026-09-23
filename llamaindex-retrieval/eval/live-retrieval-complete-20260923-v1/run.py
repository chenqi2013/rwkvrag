"""Resumable, serial full-suite replay against the unchanged local API."""

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INPUTS = HERE / "INPUTS.json"
SETTINGS = ROOT / "data/services/local-app/settings.json"
OUT = ROOT / "data/quality-runs/live-retrieval-complete-20260923-v1/run1"
BASE_URL = "http://127.0.0.1:18440"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save_new(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def call(path, payload, timeout):
    started = time.monotonic()
    request = Request(BASE_URL + path, json.dumps(payload, ensure_ascii=False).encode(),
                      {"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.status
            raw = response.read(32 * 1024 * 1024 + 1)
            if len(raw) > 32 * 1024 * 1024:
                raise ValueError("response_too_large")
    except HTTPError as error:
        status = error.code
        raw = error.read(32 * 1024 * 1024 + 1)
    except (URLError, TimeoutError, OSError, ValueError) as error:
        return {"status": "failed", "http_status": None,
                "error_type": type(error).__name__,
                "elapsed_ms": round((time.monotonic() - started) * 1000)}
    result = {"http_status": status, "elapsed_ms": round((time.monotonic() - started) * 1000),
              "response_sha256": sha(raw), "response_body_base64": base64.b64encode(raw).decode()}
    try:
        result["response"] = json.loads(raw)
        result["status"] = "recorded" if status == 200 else "http_error"
    except (ValueError, UnicodeDecodeError):
        result["status"] = "invalid_json"
    return result


def health():
    with urlopen(BASE_URL + "/v1/admin/health", timeout=10) as response:
        if response.status != 200:
            raise RuntimeError("health_failed")
        return json.load(response)


def main():
    inputs = json.loads(INPUTS.read_text(encoding="utf-8"))
    for name, expected in inputs["sources"].items():
        if sha((ROOT / name).read_bytes()) != expected:
            raise RuntimeError(f"source_changed: {name}")
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    observed_health = health()
    if observed_health.get("status") != "ok" or observed_health.get("lexical", {}).get("index_version") != settings["opensearch_index"]:
        raise RuntimeError("service_or_index_binding_changed")
    binding = {
        "protocol": inputs["protocol"], "started_at": datetime.now(timezone.utc).isoformat(),
        "inputs_sha256": sha(INPUTS.read_bytes()), "settings_sha256": sha(SETTINGS.read_bytes()),
        "runner_sha256": sha(Path(__file__).read_bytes()),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "api": BASE_URL, "model": settings["native_model"],
        "provider": settings["web_search_provider"],
        "web_search_results": settings["web_search_results"],
        "web_search_max_queries": settings["web_search_max_queries"],
        "opensearch_index": settings["opensearch_index"],
        "health": observed_health,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    binding_path = OUT / "BINDING.json"
    if binding_path.exists():
        old = json.loads(binding_path.read_text(encoding="utf-8"))
        for key in ("protocol", "inputs_sha256", "settings_sha256", "runner_sha256", "api", "model", "provider", "opensearch_index"):
            if binding[key] != old[key]:
                raise RuntimeError(f"resume_binding_changed: {key}")
    else:
        save_new(binding_path, binding)
    completed = {}
    for case in inputs["cases"]:
        ordinal, uid = case["ordinal"], case["uid"]
        search_path = OUT / f"{ordinal:02d}.search.json"
        ask_path = OUT / f"{ordinal:02d}.ask.json"
        history = list(case["history"])
        parent_failed = False
        if case["parent_uid"]:
            parent = completed[case["parent_uid"]]
            history = [*parent["request"]["history"],
                {"role": "user", "content": parent["request"]["question"]}]
            answer = parent.get("response", {}).get("answer")
            if isinstance(answer, str):
                history.append({"role": "assistant", "content": answer})
            else:
                parent_failed = True
        payload = {"question": case["question"], "retrieval_mode": case["retrieval_mode"],
                   "knowledge_base_id": case["knowledge_base_id"], "history": history}
        if search_path.exists():
            search_row = json.loads(search_path.read_text(encoding="utf-8"))
        else:
            search_row = {"uid": uid, "request": {**payload, "top_k": 20},
                          "started_at": datetime.now(timezone.utc).isoformat(),
                          **call("/v1/search", {**payload, "top_k": 20}, 240)}
            save_new(search_path, search_row)
        if ask_path.exists():
            ask_row = json.loads(ask_path.read_text(encoding="utf-8"))
        else:
            ask_row = {"uid": uid, "request": payload, "parent_failed": parent_failed,
                       "started_at": datetime.now(timezone.utc).isoformat(),
                       **call("/v1/ask", payload, 240)}
            save_new(ask_path, ask_row)
        completed[uid] = ask_row
        response = ask_row.get("response", {})
        print(json.dumps({"ordinal": ordinal, "uid": uid,
            "search_http": search_row.get("http_status"),
            "ask_http": ask_row.get("http_status"),
            "ask_status": response.get("generation", {}).get("status"),
            "ask_ms": ask_row.get("elapsed_ms"),
            "sources": len(response.get("sources", []))}, ensure_ascii=False), flush=True)
        time.sleep(1)
    summary = {"planned": len(inputs["cases"]), "recorded_search": len(list(OUT.glob("*.search.json"))),
               "recorded_ask": len(list(OUT.glob("*.ask.json"))),
               "finished_at": datetime.now(timezone.utc).isoformat()}
    if not (OUT / "SUMMARY.json").exists():
        save_new(OUT / "SUMMARY.json", summary)
    print("completed", json.dumps(summary), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(type(error).__name__, str(error), file=sys.stderr)
        raise
