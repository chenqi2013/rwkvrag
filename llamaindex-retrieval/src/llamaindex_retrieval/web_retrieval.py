"""In-repository web transport and query-local rank fusion."""
import asyncio
import hashlib
import json
from urllib.parse import urlsplit

from .direct_web import execute
from .lexical_index import LexicalResult
from .web_guard import WebProviderGuard


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
    def __init__(self, settings):
        self.settings = settings
        self.slots = asyncio.Semaphore(settings.web_search_concurrency)
        self.guard = WebProviderGuard(getattr(settings, "web_guard_path", None),
            min_interval=getattr(settings, "web_min_interval_seconds", 1.0),
            auth_cooldown=getattr(settings, "web_auth_cooldown_seconds", 86400.0),
            rate_cooldown=getattr(settings, "web_rate_cooldown_seconds", 60.0),
            lease_seconds=getattr(settings, "web_search_timeout", 45) + 5)

    async def _execute(self, request, timeout):
        async with self.slots:
            return await asyncio.wait_for(execute(request, self.settings, self.guard), timeout)

    async def decide(self, messages):
        trace = await self._execute({"action": "route", "messages": messages,
            "router_base_url": self.settings.web_router_base_url,
            "router_model": self.settings.web_router_model}, self.settings.web_router_timeout)
        trace["configured_state_sha256"] = self.settings.web_router_state_sha256
        return trace

    async def search(self, query):
        payload = await self._execute({"action": "search", "query": query,
            "max_results": self.settings.web_search_results,
            "material_characters": self.settings.web_search_material_characters}, self.settings.web_search_timeout)
        hits = []
        snapshots = []
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
                "content_sha256": sha(text), "snapshot_sha256": sha(snapshot),
                "fetch_error": item["error"]}
            hits.append(LexicalResult(node_id=identity, document_id="web:" + sha(url),
                text=text, metadata=metadata, score=float(item["score"])))
            snapshots.append({"id": identity, "url": url, "text": snapshot,
                              "sha256": sha(snapshot), "retrieved_at": item["retrieved_at"]})
        return hits, {"status": "completed", "query": query, "provider": payload["provider"],
            "source_hashes": payload["source_hashes"], "snapshots": snapshots,
            "returned": len(hits)}


# Historical frozen launchers import this name; current requests use only WebSearchAdapter.
SearchReaderAdapter = WebSearchAdapter
