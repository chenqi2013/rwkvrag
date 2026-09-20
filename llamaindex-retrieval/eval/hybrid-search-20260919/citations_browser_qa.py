"""Presentation QA using saved real model responses and labelled error fixtures.

HTTP API responses are intercepted; no model calls or business writes occur.
Set CITATION_QA_BASE to test the deployed static bundle after release.
"""
import copy
import json
import os
from pathlib import Path
import re
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/citations-20260919"
OUT.mkdir(exist_ok=True)
BASE = os.environ.get("CITATION_QA_BASE", "http://127.0.0.1:18443")
real = json.loads((ROOT / "artifacts/hybrid-search-20260919/live-hybrid-answer.json").read_text())
wiki = json.loads((ROOT / "artifacts/wiki-20260919/preview-wiki-final-v2.json").read_text())
fixture = copy.deepcopy(real)
fixture["answer"] += "\n缺失来源展示测试[资料 999]"
fixture["generation"]["answer_span"] = [0, len(fixture["answer"])]
stamp = "2026-09-19T12:00:00Z"
run = {"id": "fixture-run", "run_number": 1, "created_at": stamp,
       "request": {"question": "引用展示验收"}, "response": real}
item = {"id": "fixture-history", "question": "引用展示验收", "created_at": stamp, "updated_at": stamp,
        "run_count": 1, "request": run["request"], "latest_run": run}
checks = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.add_init_script("localStorage.setItem('rwkvrag-language','zh')")
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def respond(route):
        path = urlparse(route.request.url).path
        if path == "/v1/ask":
            data = fixture
        elif path == "/v1/admin/knowledge-bases":
            data = []
        elif path == "/v1/admin/search-history":
            data = {"items": [item], "page": 1, "page_size": 20, "total": 1}
        elif path == "/v1/admin/search-history/fixture-history":
            data = {**item, "runs": [run]}
        elif path == "/v1/admin/wiki":
            data = [wiki]
        elif path.startswith("/v1/admin/wiki/files/"):
            data = [wiki]
        elif path.startswith("/v1/admin/wiki/versions/"):
            data = wiki
        else:
            raise AssertionError("Unexpected API call: " + path)
        route.fulfill(json=data)

    page.route("**/v1/**", respond)
    page.goto(BASE + "/admin/#/search")
    page.get_by_role("textbox").fill("引用展示验收")
    page.get_by_role("button", name="检索并生成答案").click()
    citation = page.get_by_role("button", name="查看引用 1", exact=True).first
    expect(citation).to_be_visible()
    citation.focus()
    page.keyboard.press("Enter")
    drawer = page.get_by_role("dialog").last
    expect(drawer.get_by_text("回答使用的逐字证据", exact=True)).to_be_visible()
    expect(drawer.locator(".citation-full-text").first).to_contain_text("2026 年 10 月 15 日")
    assert page.locator('a[href^="file:"]').count() == 0
    page.wait_for_timeout(350)
    page.screenshot(path=str(OUT / "search-citation.png"), full_page=True)
    checks.append("search: keyboard activation opens saved local evidence; no broken file links")
    page.keyboard.press("Escape")
    page.get_by_role("button", name=re.compile(r"^\[资料 2\]")).click()
    drawer = page.get_by_role("dialog").last
    expect(drawer.get_by_role("link", name=re.compile("打开来源网页"))).to_be_visible()
    assert drawer.get_by_role("link", name=re.compile("打开来源网页")).get_attribute("href") == real["sources"][1]["uri"]
    expect(drawer.locator(".citation-full-text").first).to_contain_text(real["sources"][1]["snippet"].strip())
    drawer.get_by_text("查看当时保存的来源材料", exact=True).click()
    expect(drawer.locator("details").filter(has_text="查看当时保存的来源材料")).to_have_attribute("open", "")
    checks.append("web: correct URL, full evidence and saved source material")
    page.keyboard.press("Escape")
    page.get_by_role("button", name="查看引用 999", exact=True).click()
    expect(page.get_by_role("dialog").last.get_by_text("引用没有对应来源", exact=True)).to_be_visible()
    checks.append("missing citation: explicit error, no substituted source")
    page.keyboard.press("Escape")

    page.goto(BASE + "/admin/#/search-history")
    page.get_by_role("button", name="展开答案与引用", exact=True).click()
    page.get_by_role("button", name="查看引用 1", exact=True).first.click()
    expect(page.get_by_role("dialog").last.locator(".citation-full-text").first).to_contain_text("2026 年 10 月 15 日")
    checks.append("history: expanded historical run opens its saved evidence")
    page.keyboard.press("Escape")

    page.goto(BASE + "/admin/#/wiki")
    page.get_by_role("button", name="查看与历史", exact=True).click()
    page.get_by_role("button", name="查看引用 1", exact=True).first.click()
    expect(page.get_by_role("dialog").last.locator(".citation-full-text").first).to_contain_text("24 V")
    checks.append("Wiki: nested citation drawer uses the selected version")
    page.set_viewport_size({"width": 390, "height": 844})
    page.wait_for_timeout(350)
    panel = page.get_by_role("dialog").last.bounding_box()
    assert panel and panel["width"] <= 391
    page.screenshot(path=str(OUT / "wiki-citation-mobile.png"), full_page=False)
    checks.append("mobile: source drawer fits 390px viewport")
    assert not errors, errors
    (OUT / "browser-qa.json").write_text(json.dumps({"base": BASE, "checks": checks,
        "javascript_errors": errors, "api_intercepted": True,
        "fixture": "saved real hybrid/Wiki responses; missing citation is a synthetic UI failure fixture"}, ensure_ascii=False, indent=2))
    print(json.dumps({"passed": len(checks), "javascript_errors": errors}))
    browser.close()
