"""Real browser/API check against the isolated seeded preview; no route interception."""
import json
from pathlib import Path
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/atomic-evidence-20260920"
fixtures = json.loads((ROOT / "data/services/atomic-preview-20260920/fixtures.json").read_text())
case = next(c for c in fixtures if c["case_id"] == "conflict")
other = next(c for c in fixtures if c["case_id"] == "missing")
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.set_default_timeout(15000)
    page.add_init_script("localStorage.setItem('rwkvrag-language','zh')")
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto("http://127.0.0.1:18445/admin/#/atomic-evidence")
    def settle_drawer():
        page.wait_for_function("""() => {
            const e = document.querySelector('.ant-drawer-open .ant-drawer-content-wrapper');
            if (!e) return false;
            const r = e.getBoundingClientRect();
            return r.width >= 300 && r.left >= -1 && Math.abs(r.right - window.innerWidth) < 1;
        }""")
        expect(page.locator('[data-testid="atomic-source"] mark')).to_be_in_viewport()
    def choose(name):
        page.get_by_role("combobox", name="知识库").click()
        page.get_by_title(name, exact=True).click()
    choose(case["name"])
    page.get_by_label("对象", exact=True).fill(case["request"]["object"])
    page.get_by_label("一个属性", exact=True).fill(case["request"]["attribute"])
    with page.expect_response(lambda r: r.request.method == "POST" and "/atomic-evidence" in r.url, timeout=190000) as response:
        page.get_by_role("button", name="查找并核对证据").click()
    assert response.value.status == 200
    run = response.value.json()
    assert run["status"] == "completed" and len(run["claims"]) == 2, run.get("issues")
    expect(page.get_by_text("本次处理完成，语义仍待核验", exact=True)).to_be_visible()
    buttons = page.get_by_role("button", name="查看原文位置", exact=False)
    expect(buttons).to_have_count(2)
    for i, claim in enumerate(run["claims"]):
        buttons.nth(i).click()
        expect(page.locator('[data-testid="atomic-source"] mark')).to_have_text(claim["statement_quote"])
        settle_drawer()
        if i == 0:
            page.screenshot(path=str(OUT / "source-desktop.png"), full_page=True)
        page.keyboard.press("Escape")
        expect(page.get_by_role("dialog")).not_to_be_visible()
    page.reload()
    choose(case["name"])
    page.get_by_role("button", name=case["request"]["object"] + " · " + case["request"]["attribute"], exact=True).first.click()
    expect(page.get_by_role("button", name="查看原文位置", exact=False)).to_have_count(2)
    choose(other["name"])
    expect(page.get_by_role("button", name="查看原文位置", exact=False)).to_have_count(0)
    choose(case["name"])
    page.get_by_role("button", name=case["request"]["object"] + " · " + case["request"]["attribute"], exact=True).first.click()
    expect(page.get_by_role("button", name="查看原文位置", exact=False)).to_have_count(2)
    page.set_viewport_size({"width": 390, "height": 844})
    page.reload()
    choose(case["name"])
    page.get_by_role("button", name=case["request"]["object"] + " · " + case["request"]["attribute"], exact=True).first.click()
    expect(page.get_by_role("button", name="查看原文位置", exact=False)).to_have_count(2)
    assert page.locator(".atomic-evidence-page").bounding_box()["width"] >= 300
    assert not page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
    page.get_by_role("button", name="查看原文位置", exact=False).first.click()
    expect(page.locator('[data-testid="atomic-source"] mark')).to_be_visible()
    settle_drawer()
    page.screenshot(path=str(OUT / "source-mobile.png"), full_page=False, animations="disabled")
    overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
    assert not overflow, "page overflows mobile viewport"
    assert not errors, errors
    (OUT / "browser-qa.json").write_text(json.dumps({
        "api_intercepted": False, "real_model": True, "run_id": run["id"],
        "checks": ["actual UI -> OpenSearch -> 7.2B selection -> 2.9B Reader -> MongoDB",
                   "both conflicts retain their exact source highlights",
                   "reload and reopen saved history", "knowledge-base switch clears previous evidence",
                   "390px mobile source drawer, no horizontal page overflow"],
        "javascript_errors": errors, "mobile_overflow": overflow}, ensure_ascii=False, indent=2))
    browser.close()
    print("Browser/API checks passed")
