"""Mechanically parse tool result order, then fetch with existing SearchReader."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import re
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path("/home/chase/GitHub/RWKV-SearchReader")
sys.path.insert(0, str(ROOT / "src"))
from rwkv_search_reader.config import get_settings
from rwkv_search_reader.models import SearchHit
from rwkv_search_reader.retrieval import fetch_page, normalize_url

def main():
    request_path = Path(sys.argv[1])
    prefix = str(request_path).removesuffix(".request.json")
    request = json.loads(request_path.read_text())
    raw = json.loads(Path(prefix + ".search.json").read_text())
    if not isinstance(raw, str):
        raise TypeError("expected exact web tool text")
    pattern = re.compile(r"^(.+?) \((https?://[^\n]+)\)$", re.MULTILINE)
    matches = list(pattern.finditer(raw))
    settings = get_settings()
    def fetch(index):
        match = matches[index]
        end = matches[index+1].start() if index+1 < len(matches) else len(raw)
        snippet = raw[match.end():end].strip()
        hit = SearchHit(title=match[1], url=normalize_url(match[2]), snippet=snippet,
                        provider="test_web_search_tool")
        hit = fetch_page(hit, settings)
        snapshot = hit.content_markdown or hit.content or hit.snippet
        text = (hit.content or hit.snippet)[:request["material_characters"]]
        value = asdict(hit)
        value.update(snapshot=snapshot, reader_text=text, material_limited=len(text)<len(snapshot),
                     error="page_fetch_failed" if hit.error else "")
        return value
    with ThreadPoolExecutor(max_workers=3) as pool:
        hits = list(pool.map(fetch, range(min(len(matches), request["max_results"])) ))
    output = {"provider":"test_web_search_tool", "hits":hits,
              "source_hashes":{name:hashlib.sha256((ROOT / "src/rwkv_search_reader" / name).read_bytes()).hexdigest()
                  for name in ("config.py","models.py","retrieval.py")}}
    target = Path(prefix + ".response.json")
    tmp = Path(prefix + ".response.tmp")
    tmp.write_text(json.dumps(output, ensure_ascii=False)); tmp.rename(target)
    print(json.dumps({"request":request_path.name,"hits":len(hits),"states":[h["content_status"] for h in hits]}))

if __name__ == "__main__":main()
