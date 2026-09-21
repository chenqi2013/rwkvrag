"""Run every frozen real-world question against the isolated application."""
import base64
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "data/quality-runs/github-natural-comparison-20260921/run1"

def save(path, value):
    with path.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)

def main():
    for name, digest in json.loads((HERE / "PINS.json").read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    OUT.mkdir(parents=True, exist_ok=False)
    cases = json.loads((HERE / "CASES.json").read_text())
    records = []; completed = {}
    with httpx.Client(base_url="http://127.0.0.1:18451", timeout=900) as client:
        client.get("/health").raise_for_status()
        for ordinal, case in enumerate(cases):
            history = []; parent_failed = False
            if case["parent_id"]:
                parent = completed[case["parent_id"]]
                history = [*parent["request"]["history"], {"role":"user","content":parent["request"]["question"]}]
                if "answer" in parent.get("response", {}):
                    history.append({"role":"assistant","content":parent["response"]["answer"]})
                else: parent_failed = True
            payload = {k:case[k] for k in ["question", "retrieval_mode", "knowledge_base_id"]}
            payload["history"] = history
            payload.update(top_k=8, candidate_k=40)
            row = {"ordinal":ordinal, "id":case["id"], "request":payload,
                   "started_at":datetime.now(timezone.utc).isoformat(), "parent_id":case["parent_id"], "parent_failed":parent_failed}
            save(OUT / f"{ordinal:02d}.request.json", row)
            tick = time.monotonic()
            try:
                response = client.post("/v1/ask", json=payload)
                row.update(http_status=response.status_code,
                           response_sha256=hashlib.sha256(response.content).hexdigest(),
                           response_body_base64=base64.b64encode(response.content).decode())
                response.raise_for_status()
                row.update(status="recorded", response=response.json())
            except Exception as exc:
                row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            row["elapsed_s"] = time.monotonic() - tick
            save(OUT / f"{ordinal:02d}.json", row)
            completed[case["id"]] = row
            records.append({k:row.get(k) for k in ["ordinal","id","status","http_status","elapsed_s"]})
            print(json.dumps(records[-1]), flush=True)
    save(OUT / "SUMMARY.json", {"planned":len(cases),"records":records})

if __name__ == "__main__":
    main()
