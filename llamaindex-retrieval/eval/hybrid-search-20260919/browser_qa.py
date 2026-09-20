"""Writes a fictional fixture ONLY to the isolated :18442 preview, then uses the UI."""
import json
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

BASE = "http://127.0.0.1:18442"
OUT = Path(__file__).resolve().parents[3] / "artifacts/hybrid-search-20260919"
QUESTION = "内部迁移计划里的升级日期是什么？Python 3.13.0 的官方发布日期是什么？"

preview_config = json.loads(Path("/tmp/rwkvrag-hybrid-preview-20260919/settings.json").read_text())
assert preview_config["mongo_database"] == "rwkvrag_hybrid_preview_20260919"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    health = page.request.get(BASE + "/v1/admin/health").json()
    assert health["lexical"]["configured_index"] == "rwkvrag-hybrid-preview-20260919"
    files = page.request.get(BASE + "/v1/admin/files").json()
    if not any(item["filename"] == "hybrid-acceptance.md" for item in files):
        response = page.request.post(BASE + "/v1/admin/files", multipart={
            "knowledge_base_id": "default",
            "file": {"name": "hybrid-acceptance.md", "mimeType": "text/markdown",
                     "buffer": "# 内部迁移计划（虚构验收）\n\n本页仅用于隔离环境验收。团队计划在 2026 年 10 月 15 日升级到 Python 3.13.0。内部实施日期不是该版本的公开发布日期。\n".encode()},
        })
        assert response.status == 202, response.text()
        file_id = response.json()["file_id"]
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            item = page.request.get(BASE + "/v1/admin/files/" + file_id).json()
            if item["status"] == "ready":
                break
            assert item["status"] != "failed", item
            time.sleep(1)
        else:
            raise AssertionError("file ingestion did not complete")
    page.goto(BASE + "/admin/#/search")
    page.get_by_text("中文", exact=True).click()
    expect(page.get_by_text("自动判断是否联网", exact=True)).to_be_visible()
    page.get_by_role("textbox").fill(QUESTION)
    with page.expect_response(lambda r: r.url == BASE + "/v1/ask" and r.request.method == "POST", timeout=240000) as pending:
        page.get_by_role("button", name="检索并生成答案").click()
    response = pending.value
    assert response.ok, response.text()
    result = response.json()
    (OUT / "live-hybrid-answer.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    page.wait_for_timeout(1000)
    page.screenshot(path=str(OUT / "hybrid-answer.png"), full_page=True)
    origins = sorted({source["metadata"]["retrieval_origin"] for source in result["sources"]})
    audit = {"requested_mode": response.request.post_data_json["retrieval_mode"],
        "selected_mode": result["retrieval"].get("routing", {}).get("selected_mode"),
        "generation_status": result["generation"]["status"], "evidence_origins": origins,
        "javascript_errors": errors, "semantic_reviewed": False}
    (OUT / "browser-qa.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    assert audit["selected_mode"] == "hybrid"
    assert origins == ["knowledge_base", "web"]
    assert not errors, errors
    browser.close()
