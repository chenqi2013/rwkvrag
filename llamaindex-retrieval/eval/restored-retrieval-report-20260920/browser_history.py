"""Read saved real regression answers in the actual UI, without model requests."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/broad-regression-20260920/restored-browser'
RUN = ROOT / 'data/quality-runs/restored-retrieval-v2-20260920/run1/calls'


def main():
    OUT.mkdir(exist_ok=False)
    row = next(r for p in sorted(RUN.glob('*.json'))
        if (r := json.loads(p.read_text()))['case_id'] == 'wiki_verified-0135')
    response = json.loads(row['raw_response'])
    source = response['sources'][0]
    writes = []
    errors = []
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
        expect(drawer.locator('.citation-full-text').first).to_contain_text(source['snippet'].strip())
        link = drawer.get_by_role('link', name='打开来源网页', exact=False)
        assert link.get_attribute('href') == source['uri']
        page.screenshot(path=str(OUT / 'history-source-desktop.png'), full_page=True)
        page.set_viewport_size({'width': 390, 'height': 844})
        page.wait_for_timeout(400)
        box = drawer.bounding_box()
        assert box and box['width'] <= 391
        expect(drawer.locator('.citation-full-text').first).to_be_visible()
        page.screenshot(path=str(OUT / 'history-source-mobile.png'), full_page=False)
        assert not writes and not errors
        (OUT / 'RESULT.json').write_text(json.dumps({
            'case_id': row['case_id'], 'question': row['payload']['question'],
            'base': 'http://127.0.0.1:18446', 'api_intercepted': False,
            'saved_real_regression_response': True, 'extra_model_requests': 0,
            'exact_saved_source_visible': True, 'correct_revision_link': source['uri'],
            'mobile_drawer_fits_390px': True, 'javascript_errors': errors,
            'write_requests': writes,
            'boundary': 'One saved answer UI/source visibility check; not semantic or all-citation validation.'
        }, ensure_ascii=False, indent=2) + '\n')
        browser.close()
    print('Saved real answer and source visible in desktop/mobile history UI.')


if __name__ == '__main__':
    main()
