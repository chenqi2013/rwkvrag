"""Browser regression with every /v1 request mocked; no real search or save is sent.

Run Vite separately, then use a Python environment containing Playwright:
  python tests/saveWebSource.browser.py --base-url http://127.0.0.1:18506/admin/ \
    --browser /path/to/cached/chrome --output /tmp/save-source-ui
No browser or package is installed by this test.
"""

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--browser", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, executable_path=args.browser)
        for case in ("cancel", "confirm", "snippet", "legacy", "get-error", "wrong-id", "existing-failed", "cited-answer", "save-retry", "provider-failure", "provider-partial"):
            context = browser.new_context(viewport={"width": 1440, "height": 1100}, locale="en-US")
            calls, errors, unexpected = [], [], []
            source = {"id": "web-result", "document_id": "web-doc", "source": "web", "title": "Snapshot Example",
                      "uri": "https://example.org/docs", "score": 1, "snippet": "Selected quote only.",
                      "metadata": {"web_snapshot_id": "snapshot-full-001", "content_status": "fetched"}}
            receipt = {"id": "snapshot-full-001", "title": "Snapshot Example", "url": "https://example.org/docs",
                       "text": "FULL CAPTURED PAGE\nA complete source paragraph beyond the selected quote.",
                       "sha256": "a" * 64, "retrieved_at": "2026-09-24T00:00:00Z",
                       "content_status": "fetched", "provider": "test-provider", "save_allowed": True}
            if case == "snippet":
                source["metadata"]["content_status"] = receipt["content_status"] = "snippet_only"
                receipt["save_allowed"] = False
            if case == "legacy":
                source["metadata"] = {}
            if case == "wrong-id":
                receipt["id"] = "another-snapshot"
            answer = {"answer": "Captured claim [资料 1]", "sources": [source], "retrieval": {}, "generation": {}}

            def mock_api(route):
                request = route.request
                path = urlparse(request.url).path
                calls.append({"method": request.method, "path": path, "body": request.post_data_json if request.post_data else None})
                status = 200
                if path == "/v1/admin/knowledge-bases":
                    body = [{"id": "kb-a", "name": "First KB"}, {"id": "kb-b", "name": "Second KB"}]
                elif path == "/v1/search":
                    body = {"results": [source], "retrieval": {}}
                    if case in ("provider-failure", "provider-partial"):
                        body["retrieval"] = {"all_providers_failed": case == "provider-failure",
                                             "provider_failures": [{"error": "temporary provider failure"}]}
                        if case == "provider-failure":
                            body["results"] = []
                elif path == "/v1/ask":
                    body = answer
                elif path == "/v1/admin/web-snapshots/snapshot-full-001":
                    status, body = (404, {"detail": "snapshot_not_found"}) if case == "get-error" else (200, receipt)
                elif path == "/v1/admin/web-snapshots/snapshot-full-001/save":
                    body = {"file_id": "file-1", "job_id": "job-1", "status": "failed" if case == "existing-failed" else "pending",
                            "existing": case == "existing-failed"}
                    if case == "save-retry" and len([c for c in calls if c["path"].endswith("/save")]) == 1:
                        status, body = 503, {"detail": "temporary save failure"}
                else:
                    unexpected.append(path)
                    status, body = 501, {"detail": "Unexpected request blocked by browser test"}
                route.fulfill(status=status, content_type="application/json", body=json.dumps(body))

            context.route("**/v1/**", mock_api)
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(args.base_url + "#/search", wait_until="networkidle")
            page.get_by_label("Question", exact=True).fill("Preview a source")
            page.get_by_role("button", name="Search and generate answer" if case == "cited-answer" else "Retrieve sources only").click()
            if case == "provider-failure":
                expect(page.get_by_text("Web retrieval failed", exact=True)).to_be_visible()
                expect(page.get_by_text("No sources found", exact=True)).to_have_count(0)
                assert not unexpected and not errors
                page.screenshot(path=str(args.output / f"{case}.png"), full_page=True)
                records.append({"case": case, "passed": True, "calls": calls, "page_errors": errors})
                context.close()
                print(f"PASS {case}", flush=True)
                continue
            if case == "provider-partial":
                expect(page.get_by_text("Some web retrieval requests failed", exact=True)).to_be_visible()
            expect(page.get_by_text("Snapshot Example", exact=False).first).to_be_visible()
            save_calls = lambda: [c for c in calls if c["path"].endswith("/save")]
            search_count = len([c for c in calls if c["path"] in ("/v1/search", "/v1/ask")])
            assert search_count == 1
            if case == "legacy":
                expect(page.get_by_role("button", name="Save to knowledge base", exact=True)).to_have_count(0)
                expect(page.get_by_text("This source has no verifiable saved snapshot.", exact=False)).to_be_visible()
                assert not save_calls()
            else:
                page.get_by_role("button", name="Save to knowledge base", exact=True).first.click()
                dialog = page.get_by_role("dialog")
                confirm = dialog.get_by_role("button", name=re.compile("Confirm saving this snapshot"))
                if case == "get-error":
                    expect(dialog.get_by_text("snapshot_not_found", exact=True)).to_be_visible()
                    expect(confirm).to_be_disabled()
                else:
                    expect(dialog.get_by_text("FULL CAPTURED PAGE", exact=False)).to_be_visible()
                    expect(confirm).to_be_disabled()
                    if case in ("snippet", "wrong-id"):
                        expect(dialog.get_by_role("combobox")).to_be_disabled()
                    else:
                        dialog.get_by_role("combobox").click()
                        page.get_by_text("Second KB", exact=True).last.click()
                        expect(confirm).to_be_enabled()
                if case in ("confirm", "existing-failed", "cited-answer", "save-retry"):
                    if case == "save-retry":
                        confirm.click()
                        expect(dialog.get_by_text("temporary save failure", exact=True)).to_be_visible()
                        expect(confirm).to_be_enabled()
                    confirm.click()
                    expect(dialog.get_by_role("link", name="View ingestion tasks", exact=True)).to_have_attribute("href", "#/imports")
                    assert len(save_calls()) == (2 if case == "save-retry" else 1)
                    assert save_calls()[0]["body"] == {"knowledge_base_id": "kb-b", "expected_sha256": "a" * 64, "confirmed": True}
                    if case == "existing-failed":
                        expect(dialog.get_by_text("Ingestion failed; check the task record", exact=True)).to_be_visible()
                        expect(dialog.get_by_text("Ingestion task submitted", exact=True)).to_have_count(0)
                else:
                    dialog.get_by_role("button", name="Cancel", exact=True).click()
                    expect(dialog).not_to_be_visible()
                    assert not save_calls()
                assert len([c for c in calls if c["path"] in ("/v1/search", "/v1/ask")]) == search_count
                if case == "cited-answer":
                    expect(page.get_by_text("Captured claim", exact=False).first).to_contain_text("Captured claim")
            assert not unexpected, unexpected
            assert not errors, errors
            page.screenshot(path=str(args.output / f"{case}.png"), full_page=True)
            records.append({"case": case, "passed": True, "calls": calls, "page_errors": errors})
            context.close()
            print(f"PASS {case}", flush=True)
        browser.close()
    (args.output / "RESULTS.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
