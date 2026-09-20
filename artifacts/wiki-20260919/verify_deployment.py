"""Read-only deployed UI verification; requires Python Playwright and Chromium."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

output = Path(__file__).resolve().parent
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    base = "http://127.0.0.1:18440"
    health = page.request.get(base + "/v1/admin/health")
    assert health.ok
    wiki = page.request.get(base + "/v1/admin/wiki")
    assert wiki.ok
    page.goto(base + "/admin/#/wiki")
    expect(page.get_by_text("Wiki", exact=True).first).to_be_visible()
    page.wait_for_timeout(1000)
    page.screenshot(path=str(output / "deployed-wiki.png"), full_page=True)
    assert not errors, errors
    (output / "deployment-verification.json").write_text(json.dumps({
        "health": health.json(), "wiki_api_status": wiki.status,
        "wiki_pages": len(wiki.json()), "javascript_errors": errors,
        "url": base + "/admin/#/wiki",
        "release": "data/services/local-app/releases/wiki-20260919",
        "existing_documents_preserved": health.json()["lexical"]["documents"] == 46051,
        "note": "Read-only deployment check; real-model upload and revision verified in isolated preview."
    }, ensure_ascii=False, indent=2))
    browser.close()
