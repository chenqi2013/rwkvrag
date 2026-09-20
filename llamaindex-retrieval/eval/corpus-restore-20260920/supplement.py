"""Freeze four missing Wikipedia pages at explicit revisions; keep raw HTML/wikitext."""
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import urllib.parse
import urllib.request

from markdownify import markdownify

ROOT = Path('data/corpora/finewiki-restored-20260920/wikipedia-supplement')
PAGES = {'CPUID': 93517304, 'センラ': 94343040, '成县': 86994369, '蒙古航空': 87726745}


def main():
    ROOT.mkdir(exist_ok=False)
    manifests = []
    for title, revision in PAGES.items():
        url = 'https://zh.wikipedia.org/w/api.php?' + urllib.parse.urlencode({
            'action': 'parse', 'format': 'json', 'oldid': revision,
            'prop': 'text|wikitext|revid|displaytitle'})
        with urllib.request.urlopen(urllib.request.Request(url,
                headers={'User-Agent': 'RWKVRAG-corpus-recovery/1.0 (local research)'}), timeout=60) as response:
            raw = response.read()
        data = json.loads(raw)['parse']
        assert data['revid'] == revision and data['title'] == title
        prefix = str(data['pageid']) + '-' + str(revision)
        (ROOT / (prefix + '.json')).write_bytes(raw)
        html = data['text']['*']
        wikitext = data['wikitext']['*']
        (ROOT / (prefix + '.html')).write_text(html)
        (ROOT / (prefix + '.wikitext')).write_text(wikitext)
        text = '# ' + title + '\n\n' + markdownify(html, heading_style='ATX')
        (ROOT / (prefix + '.md')).write_text(text)
        manifests.append({'title': title, 'page_id': data['pageid'], 'version': revision,
            'id': 'zhwiki/' + str(data['pageid']), 'snapshot_id': 'wikipedia:' + prefix,
            'document_id': sha256(('wikipedia:' + prefix).encode()).hexdigest()[:24],
            'url': 'https://zh.wikipedia.org/w/index.php?oldid=' + str(revision),
            'text_path': prefix + '.md', 'text_sha256': sha256(text.encode()).hexdigest(),
            'html_sha256': sha256(html.encode()).hexdigest(),
            'wikitext_sha256': sha256(wikitext.encode()).hexdigest(),
            'api_response_sha256': sha256(raw).hexdigest(),
            'conversion': {'tool': 'markdownify', 'version': version('markdownify'), 'heading_style': 'ATX',
                'prepended_heading': True, 'raw_html_and_wikitext_retained': True},
            'license': 'CC-BY-SA-4.0', 'attribution': 'Wikipedia contributors',
            'boundary': 'Supplemented source revision, not the deleted original FineWiki artifact.'})
        print(json.dumps({'title': title, 'revision': revision, 'characters': len(text)}, ensure_ascii=False), flush=True)
    (ROOT / 'manifest.json').write_text(json.dumps(manifests, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
