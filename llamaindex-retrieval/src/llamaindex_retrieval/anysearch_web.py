"""AnySearch transport: search snippets and separately fetched page snapshots."""
import asyncio
import base64
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from time import monotonic
from urllib.parse import urlsplit

from . import direct_web
from .direct_web import WebProviderError
from .web_guard import WebGuardBlocked, WebGuardUnavailable, WebProviderGuard


async def _post(client, operation, body, settings, guard, key, captures):
    identity = guard.identity("anysearch", key)
    try:
        slot = await guard.reserve(identity)
    except (WebGuardBlocked, WebGuardUnavailable) as exc:
        raise WebProviderError(str(exc), operation=operation) from None

    async def on_http_error(status, retry_after):
        try:
            await guard.block(identity, status, retry_after)
        except WebGuardUnavailable as exc:
            raise WebProviderError(str(exc)) from None

    raw = []
    try:
        payload = await direct_web._json_response(
            client, "POST", "https://api.anysearch.com/v1/" + operation,
            headers={"Authorization": "Bearer " + key}, body=body,
            timeout=settings.web_search_timeout, on_http_error=on_http_error,
            raw_response=raw)
    except WebProviderError as exc:
        exc.operation = operation
        raise
    finally:
        try:
            await guard.complete(slot)
        except WebGuardUnavailable as exc:
            raise WebProviderError(str(exc)) from None
    if type(payload.get("code")) is not int or payload["code"] != 0:
        # Provider messages may contain credentials; never attach them to traces.
        raise WebProviderError("web_anysearch_business_error")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise WebProviderError("web_anysearch_data_not_object")
    if captures is not None:
        captures.append({"operation": operation, "request": body,
                         "response_base64": base64.b64encode(raw[0]).decode("ascii"),
                         "response_sha256": hashlib.sha256(raw[0]).hexdigest()})
    return data


def _allowed(url, domains):
    host = urlsplit(url).hostname.lower().rstrip(".")
    return not domains or any(host == domain.lower().rstrip(".") or
                             host.endswith("." + domain.lower().rstrip("."))
                             for domain in domains)


ROTATE = ('web_upstream_http_401', 'web_upstream_http_402', 'web_upstream_http_403', 'web_upstream_http_429',
          'web_upstream_circuit_open_401', 'web_upstream_circuit_open_402', 'web_upstream_circuit_open_403',
          'web_upstream_circuit_open_429', 'web_upstream_circuit_open_432', 'web_upstream_circuit_open_433',
          'web_upstream_http_432', 'web_upstream_http_433', 'web_anysearch_business_error')


def _keys(settings):
    pool = getattr(settings, 'web_anysearch_api_key_pool_file', None)
    if not pool:
        key = direct_web._api_key(settings, "anysearch")
        return [key] if key else []
    path = Path(pool)
    if path.stat().st_mode & 0o077:
        raise WebProviderError("web_anysearch_key_file_permissions")
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


async def search(request, settings, guard=None):
    """Search with the first usable key; on auth/quota/rate failure move to the next pooled key."""
    keys = _keys(settings)
    if not keys:
        raise WebProviderError("web_anysearch_key_not_configured")
    tried = []
    for index, key in enumerate(keys):
        try:
            result = await _search(request, settings, guard, key)
            result['key_rotation'] = {'key_index': index, 'skipped': tried}
            return result
        except WebProviderError as exc:
            if exc.safe_code not in ROTATE or index == len(keys) - 1:
                raise
            # Only safe codes are recorded; keys themselves never enter traces.
            tried.append({'key_index': index, 'error': exc.safe_code})


async def _search(request, settings, guard, key):
    deadline = monotonic() + settings.web_search_timeout - 0.1
    guard = guard or WebProviderGuard(
        getattr(settings, "web_guard_path", None),
        min_interval=getattr(settings, "web_min_interval_seconds", 1.0),
        auth_cooldown=getattr(settings, "web_auth_cooldown_seconds", 86400.0),
        rate_cooldown=getattr(settings, "web_rate_cooldown_seconds", 60.0),
        lease_seconds=settings.web_search_timeout + 5,
        parallel=getattr(settings, "web_provider_parallel", 1))
    count = request["max_results"]
    captures = [] if request.get("capture_provider_payload") else None
    hits = []
    url_results = []
    async with direct_web._client() as client:
        if request.get('action') == 'extract':
            results = [{'url': url, 'title': url} for url in request['urls']]
        else:
            data = await _post(client, "search", {"query": request["query"],
                               "max_results": count, "format": "json"}, settings, guard, key, captures)
            results = data.get("results")
        if not isinstance(results, list):
            raise WebProviderError("web_upstream_results_not_array")
        candidates = []
        for item in results:
            if not isinstance(item, dict):
                continue
            url = direct_web._public_url(item.get("url"))
            if not url or not _allowed(url, request.get("include_domains")):
                if request.get('action') == 'extract':
                    url_results.append({'url': item.get('url'), 'status': 'filtered'})
                continue
            # Search content can be a decorated snippet, so only /extract can
            # establish fetched status. Preserve every supplied character.
            snippet = item.get("snippet") or item.get("content") or ""
            candidates.append((url, item.get("title") or url, snippet if isinstance(snippet, str) else ""))

        async def extract(url):
            # Pages are fetched concurrently within the guard's per-key slots.
            try:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise WebProviderError("web_anysearch_extract_budget")
                page = await asyncio.wait_for(
                    _post(client, "extract", {"url": url}, settings, guard, key, captures),
                    min(settings.web_anysearch_extract_timeout, remaining))
                if page.get("url") != url:
                    raise WebProviderError("web_anysearch_extract_url_mismatch")
                content = page.get("content")
                return (content if isinstance(content, str) and content.strip() else ""), ""
            except TimeoutError:
                return "", "web_anysearch_extract_timeout"
            except WebProviderError as exc:
                if exc.safe_code not in {"web_upstream_http_400", "web_upstream_http_422",
                                         "web_upstream_failed", "web_anysearch_extract_url_mismatch",
                                         "web_anysearch_extract_budget"} and not (
                        exc.safe_code.startswith('web_upstream_http_5') or
                        exc.safe_code.startswith('web_upstream_circuit_open_5')):
                    raise
                return "", exc.safe_code

        if guard.parallel == 1:
            # One slot keeps the historical page order of extract requests.
            pages = [await extract(url) for url, _, _ in candidates]
        else:
            pages = await asyncio.gather(*(extract(url) for url, _, _ in candidates))
        for (url, title, snippet), (full, error) in zip(candidates, pages):
            snapshot = full or snippet
            url_results.append({'url': url,
                                'status': 'fetched' if full else 'snippet_only' if snippet else 'failed',
                                'error': error or ('' if full else 'web_anysearch_extract_empty')})
            text = snapshot[:request["material_characters"]]
            if not text.strip():
                continue
            hits.append({"url": url, "title": str(title)[:300],
                         "reader_text": text, "snapshot": snapshot,
                         "content_status": "fetched" if full else "snippet_only",
                         "search_snippet": snippet,
                         "material_limited": len(text) < len(snapshot), "score": 0.0,
                         "published_date": "", "error": error or ("" if full else "web_anysearch_extract_empty"),
                         "retrieved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")})
            if len(hits) >= count:
                break
    result = {"provider": "anysearch", "hits": hits,
              "url_results": url_results,
              "domain_filter": "returned_urls" if request.get("include_domains") else "none",
              "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), Path(direct_web.__file__))}}
    if captures is not None:
        result["provider_responses"] = captures
    return result
