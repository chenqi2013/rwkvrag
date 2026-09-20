"""Bounded local SearchReader process adapter and query-local rank fusion."""
import asyncio
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from .lexical_index import LexicalResult


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


class SearchReaderAdapter:
    def __init__(self, settings):
        self.settings = settings
        self.slots = asyncio.Semaphore(settings.web_search_concurrency)

    async def _execute(self, request, timeout):
        root = self.settings.searchreader_project_dir
        if root is None:
            raise RuntimeError("web_search_not_configured")
        root = root.resolve()
        python = root / ".venv/bin/python"
        bridge = Path(__file__).with_name("searchreader_bridge.py")
        request = json.dumps(request, ensure_ascii=False).encode()
        async with self.slots:
            process = await asyncio.create_subprocess_exec(str(python), str(bridge), str(root),
                cwd=root, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL)
            try:
                async def communicate():
                    process.stdin.write(request)
                    await process.stdin.drain()
                    process.stdin.close()
                    output = await process.stdout.read(8 * 1024 * 1024 + 1)
                    if len(output) > 8 * 1024 * 1024:
                        raise RuntimeError("web_response_too_large")
                    # read(n) may return before EOF; read remaining bounded bytes.
                    while part := await process.stdout.read(8 * 1024 * 1024 + 1 - len(output)):
                        output += part
                        if len(output) > 8 * 1024 * 1024:
                            raise RuntimeError("web_response_too_large")
                    await process.wait()
                    return output
                output = await asyncio.wait_for(communicate(), timeout)
                if process.returncode:
                    raise RuntimeError("searchreader_provider_failed")
                payload = json.loads(output)
            finally:
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()
        return payload

    async def decide(self, messages):
        trace = await self._execute({"action": "route", "messages": messages,
            "router_base_url": self.settings.searchreader_router_base_url,
            "router_model": self.settings.searchreader_router_model}, self.settings.searchreader_router_timeout)
        trace["configured_state_sha256"] = self.settings.searchreader_router_state_sha256
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
