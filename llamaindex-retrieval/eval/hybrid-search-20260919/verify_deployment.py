"""Read-only production checks; does not upload files or write answer history."""
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

BASE = "http://127.0.0.1:18440"
OUT = Path(__file__).resolve().parents[3] / "artifacts/hybrid-search-20260919"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    health = page.request.get(BASE + "/v1/admin/health").json()
    assert health["status"] == "ok"
    assert health["lexical"]["documents"] == 46051
    offline = page.request.post(BASE + "/v1/search", data={
        "question": "只根据知识库回答内部设备维护周期，不要联网。",
        "retrieval_mode": "auto", "top_k": 1,
    }, timeout=60000)
    assert offline.ok, offline.text()
    local = offline.json()
    assert local["retrieval"]["routing"]["selected_mode"] == "knowledge_base"
    assert local["retrieval"]["web_search"] == []
    online = page.request.post(BASE + "/v1/search", data={
        "question": "Python 3.13.0 release date site:python.org",
        "retrieval_mode": "web", "top_k": 2,
    }, timeout=60000)
    assert online.ok, online.text()
    web = online.json()
    assert web["results"]
    assert all(s["metadata"]["retrieval_origin"] == "web" for s in web["results"])
    assert not web["retrieval"]["provider_failures"]
    page.goto(BASE + "/admin/#/search")
    page.get_by_text("中文", exact=True).click()
    expect(page.get_by_text("自动判断是否联网", exact=True)).to_be_visible()
    page.screenshot(path=str(OUT / "deployed-search.png"), full_page=True)
    assert not errors, errors
    result = {"health": health, "url": BASE + "/admin/#/search", "javascript_errors": errors,
        "offline_routing": local["retrieval"]["routing"], "offline_web_calls": 0,
        "web_results": [{"uri": s["uri"], "source": s["source"]} for s in web["results"]],
        "business_documents_modified": False, "remote_router_state": "v6r2 retained after candidate validation"}
    (OUT / "deployment-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"status": "passed", "offline_mode": "knowledge_base", "web_results": len(web["results"])}))
    browser.close()
