"""Check a real unsupported citation remains visibly unsupported in history."""
import json
from pathlib import Path
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/broad-regression-20260920/restored-browser-missing-source'
RUN = ROOT / 'data/quality-runs/restored-retrieval-v2-20260920/run1/calls'


def main():
    OUT.mkdir(exist_ok=False)
    row = next(r for path in RUN.glob('*.json')
        if (r := json.loads(path.read_text()))['case_id'] == 'diverse-0014')
    response = json.loads(row['raw_response'])
    assert response['sources'] == [] and '[资料 1]' in response['answer']
    errors, writes = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.set_default_timeout(20000)
        page.add_init_script("localStorage.setItem('rwkvrag-language','zh')")
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: writes.append(request.url)
            if request.method not in ('GET', 'HEAD', 'OPTIONS') else None)
        page.goto('http://127.0.0.1:18446/admin/#/search-history')
        target = page.get_by_role('row').filter(has_text=row['payload']['question'])
        target.get_by_role('button', name='展开答案与引用', exact=True).click()
        page.get_by_role('button', name='查看引用 1', exact=True).first.click()
        drawer = page.get_by_role('dialog').last
        expect(drawer.get_by_text('引用没有对应来源', exact=True)).to_be_visible()
        assert drawer.locator('.citation-full-text').count() == 0
        page.screenshot(path=str(OUT / 'unsupported-citation.png'), full_page=False)
        assert not errors and not writes
        (OUT / 'RESULT.json').write_text(json.dumps({
            'case_id': row['case_id'], 'source_count': 0, 'citation': 1,
            'real_saved_model_answer': True, 'api_intercepted': False,
            'extra_model_requests': 0, 'explicit_missing_source_message': True,
            'unrelated_source_substituted': False, 'javascript_errors': errors,
            'write_requests': writes,
        }, ensure_ascii=False, indent=2) + '\n')
        browser.close()
    print('Actual unsupported citation shown as missing, without substituted evidence.')


if __name__ == '__main__':
    main()
