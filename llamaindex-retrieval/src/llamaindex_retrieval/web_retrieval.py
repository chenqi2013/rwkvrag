"""In-repository web transport and query-local rank fusion."""
import asyncio
import hashlib
import json
from urllib.parse import urlsplit

from .direct_web import execute
from .lexical_index import LexicalResult
from .web_guard import WebProviderGuard
from .web_replay import WebReplay


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def interleave(local, web):
    """Give both providers positions without comparing incompatible scores."""
    return [group[i] for i in range(max(len(local), len(web)))
            for group in (local, web) if i < len(group)]


def deduplicate_web_groups(groups):
    """Share exact URL/text identities across query results; preserve observations."""
    canonical = {}
    output = []
    for group in groups:
        unique = []
        seen = set()
        for hit in group:
            key = (hit.metadata.get("uri"), hit.text)
            if key not in canonical:
                canonical[key] = hit
                hit.metadata["retrieval_observations"] = []
            item = canonical[key]
            observation = {k: hit.metadata.get(k) for k in ("retrieval_query", "retrieved_at", "snapshot_sha256")}
            if observation not in item.metadata["retrieval_observations"]:
                item.metadata["retrieval_observations"].append(observation)
            if key not in seen:
                unique.append(item)
                seen.add(key)
        output.append(unique)
    return output


class WebSearchAdapter:
    def __init__(self, settings, *, replay=None):
        self.settings = settings
        self.replay = WebReplay(**(replay or {}))
        # Installed only by the server lifespan, never from request/material metadata.
        self.snapshot_recorder = None
        self.slots = asyncio.Semaphore(settings.web_search_concurrency)
        self.guard = WebProviderGuard(getattr(settings, "web_guard_path", None),
            min_interval=getattr(settings, "web_min_interval_seconds", 1.0),
            auth_cooldown=getattr(settings, "web_auth_cooldown_seconds", 86400.0),
            rate_cooldown=getattr(settings, "web_rate_cooldown_seconds", 60.0),
            lease_seconds=getattr(settings, "web_search_timeout", 45) + 5,
            parallel=getattr(settings, "web_provider_parallel", 1))

    async def _execute(self, request, timeout):
        async with self.slots:
            return await asyncio.wait_for(execute(request, self.settings, self.guard), timeout)

    async def decide(self, messages):
        trace = await self._execute({"action": "route", "messages": messages,
            "router_base_url": self.settings.web_router_base_url,
            "router_model": self.settings.web_router_model}, self.settings.web_router_timeout)
        trace["configured_state_sha256"] = self.settings.web_router_state_sha256
        return trace

    async def fetch_urls(self, urls, *, include_domains=None, case_id=None):
        """Fetch the user's exact URLs; their literal content remains the source of facts."""
        return await self.search('', full_snapshot=True, include_domains=include_domains,
                                 case_id=case_id, source_urls=urls)

    async def search(self, query, *, full_snapshot=False, include_domains=None, case_id=None,
                     source_urls=None):
        # Explicit unified-core acquisition: split/rank the full returned snapshot
        # before reading. Default path remains byte-compatible with the baseline.
        request = {"action": "search", "query": query,
            "max_results": self.settings.web_search_results,
            "material_characters": None if full_snapshot else self.settings.web_search_material_characters}
        if source_urls is not None:
            from .direct_web import _public_url
            if (not source_urls or len(source_urls) > 8
                    or any(_public_url(url) is None for url in source_urls)):
                raise ValueError('invalid_direct_source_urls')
            request.update(action='extract', urls=list(dict.fromkeys(source_urls)),
                           max_results=len(set(source_urls)))
        if include_domains is not None:
            request['include_domains'] = include_domains
        if self.replay.mode == 'record':
            request['capture_provider_payload'] = True
        provider_options = ({'extract_timeout': float(self.settings.web_anysearch_extract_timeout),
                             'search_timeout': float(self.settings.web_search_timeout)}
                            if self.settings.web_search_provider == 'anysearch' else None)
        if source_urls is not None:
            provider_options = {**(provider_options or {}), 'action': 'extract', 'urls': request['urls']}
        key = self.replay.key(self.settings.web_search_provider, query, include_domains,
                              request['max_results'], full_snapshot, provider_options=provider_options)
        payload, replay = await self.replay.execute(key, lambda: self._execute(request, self.settings.web_search_timeout),
                                                  case_id=case_id)
        hits = []
        snapshots = []
        material_selections = []
        for item in payload["hits"]:
            url = item["url"]
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                continue
            text = item["reader_text"]
            snapshot = item["snapshot"]
            # Same URL with changed text remains a different immutable identity.
            identity = "web:" + sha(json.dumps([url, text]))
            metadata = {"source": "web", "title": item["title"], "uri": url,
                "provider": payload["provider"], "retrieval_query": query,
                "retrieved_at": item["retrieved_at"], "published_date": item["published_date"],
                "content_status": item["content_status"], "material_limited": item["material_limited"],
                "search_snippet": item.get("search_snippet", ""),
                "content_sha256": sha(text), "snapshot_sha256": sha(snapshot),
                "fetch_error": item["error"]}
            if self.snapshot_recorder is not None:
                metadata["web_snapshot_id"] = await self.snapshot_recorder({
                    "title": item["title"], "url": url, "text": snapshot,
                    "sha256": sha(snapshot), "retrieved_at": item["retrieved_at"],
                    "content_status": item["content_status"], "provider": payload["provider"],
                    "retrieval_query": query,
                })
            if not full_snapshot and getattr(self.settings, "web_material_policy", "prefix") == "ranked_windows":
                from .web_materials import ranked_windows
                identity = "web:" + sha(json.dumps([url, snapshot]))
                windows, selection = ranked_windows(snapshot, query, "web:" + sha(url),
                    window=self.settings.web_search_material_characters,
                    limit=self.settings.web_windows_per_document)
                material_selections.append({"snapshot_id": identity, **selection})
                for node, rank_score in windows:
                    hits.append(LexicalResult(node_id=node.id_, document_id="web:" + sha(url),
                        text=node.text, score=float(item["score"]), metadata={
                            **node.metadata, **metadata, "web_snapshot_parent_id": identity,
                            "content_sha256": sha(node.text), "material_policy": "ranked_windows",
                            "material_limited": selection["unselected_windows"] > 0,
                            "lexical_window_score": rank_score}))
            else:
                hits.append(LexicalResult(node_id=identity, document_id="web:" + sha(url),
                    text=text, metadata=metadata, score=float(item["score"])))
            snapshots.append({"id": identity, "url": url, "text": snapshot,
                              "sha256": sha(snapshot), "retrieved_at": item["retrieved_at"]})
        return hits, {"status": "completed", "query": query, "provider": payload["provider"],
            **({'requested_urls': request['urls'], 'action': 'extract'} if source_urls is not None else {}),
            "source_hashes": payload["source_hashes"], "snapshots": snapshots,
            "material_selections": material_selections,
            "returned": len(hits), **({'web_replay': replay} if replay else {}),
            **({'url_results': payload['url_results']} if 'url_results' in payload else {}),
            **({'provider_options': provider_options, 'domain_filter': payload['domain_filter']}
               if provider_options is not None else {})}


# Historical frozen launchers import this name; current requests use only WebSearchAdapter.
SearchReaderAdapter = WebSearchAdapter
