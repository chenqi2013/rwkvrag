"""Presentation checks only; APIs intercepted with saved answer + labelled UI fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--response", type=Path)
args = parser.parse_args()
OUT = ROOT / "artifacts/task-matrix-20260920"
if args.response:
    OUT = OUT / "browser-actual"
OUT.mkdir(parents=True, exist_ok=True)
case = json.loads((Path(__file__).parent / "data-v3/validation.cases.jsonl").read_text().splitlines()[0])
saved = json.loads((ROOT / "data/quality-runs/task-matrix-20260920/writer-baseline/validation-000-13.json").read_text())
fixture = {"answer": saved["answer"], "sources": case["sources"],
    "retrieval": {"task_matrix": case["matrix"]}, "generation": {"pipeline": "rwkv", "status": "completed",
        "answer_span": [0, len(saved["answer"])], "model_calls": [{"stage": "writer", "status": "completed"}],
        "citation_map": {str(i): s["id"] for i, s in enumerate(case["sources"], 1)}}}
if args.response:
    fixture = json.loads(args.response.read_text())
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
            route.fulfill(json=fixture)
        elif path == "/v1/admin/knowledge-bases":
            route.fulfill(json=[])
        else:
            raise AssertionError(path)
    page.route("**/v1/**", respond)
    page.goto("http://127.0.0.1:18443/admin/#/search")
    page.get_by_role("textbox").fill(case["question"])
    page.get_by_role("button", name="检索并生成答案").click()
    region = page.get_by_role("region", name="对比项目与证据")
    expect(region.get_by_role("table")).to_be_visible()
    expect(region.get_by_role("row")).to_have_count(5)
    expect(region.get_by_text("模型找到支持证据", exact=True)).to_have_count(4)
    region.get_by_role("button", name="记录1", exact=True).first.click()
    drawer = page.get_by_role("dialog").last
    expect(drawer.locator(".citation-full-text").first).to_contain_text(fixture["sources"][0]["snippet"])
    checks.append("four object/dimension cells; source button opens exact saved evidence")
    page.screenshot(path=str(OUT / "matrix-source.png"), full_page=True)
    page.keyboard.press("Escape")
    fixture["retrieval"]["task_matrix"][0]["status"] = "missing"
    fixture["retrieval"]["task_matrix"][0]["source_ids"] = []
    fixture["retrieval"]["task_matrix"][1]["status"] = "conflict"
    fixture["retrieval"]["answer_review"] = {"status": "completed", "cells": [
        {"cell_id": row["id"], "valid": i != 1} for i, row in enumerate(fixture["retrieval"]["task_matrix"])], "model_judgment": {
        "valid": False, "issues": ["展示验收：第二个版本的引用需要核对。"]}}
    fixture["generation"]["status"] = "answer_quality_failed"
    page.get_by_role("button", name="检索并生成答案").click()
    expect(region.get_by_text("证据仍不足", exact=True)).to_be_visible()
    expect(region.get_by_text("来源存在冲突", exact=True)).to_be_visible()
    expect(region.get_by_text("模型检查未通过", exact=True)).to_have_count(1)
    expect(region.get_by_text("模型检查通过", exact=True)).to_have_count(3)
    expect(page.get_by_text("答案未通过模型内容检查，请核对原文", exact=True)).to_be_visible()
    expect(region.get_by_text("展示验收：第二个版本的引用需要核对。", exact=True)).to_be_visible()
    checks.append("missing/conflict and failed-review states are visibly distinct; original answer retained")
    page.screenshot(path=str(OUT / "matrix-review-failed.png"), full_page=True)
    assert not errors, errors
    (OUT / "browser-qa.json").write_text(json.dumps({"api_intercepted": True,
        "first_scenario_real_pipeline_response": bool(args.response),
        "response_sha256": hashlib.sha256(args.response.read_bytes()).hexdigest() if args.response else None,
        "synthetic_status_fixtures": True, "checks": checks, "javascript_errors": errors}, ensure_ascii=False, indent=2))
    browser.close()
    print(json.dumps({"passed": len(checks), "javascript_errors": errors}))
