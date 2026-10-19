"""Source adapters and request-local, coordinate-bound evidence catalog."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from itertools import zip_longest

from .rwkv_pipeline import source_from_hit
from .schemas import SourceItem
from .unified_contracts import digest, dumps
from .web_materials import ranked_windows


class SourceAdapters:
    """No routing, query rewriting or cross-provider fallback in the adapter."""
    def __init__(self, settings, index, web):
        self.settings, self.index, self.web = settings, index, web

    async def execute(self, action, request):
        if action.channel == "web":
            hits, receipt = await self.web.search(action.query, full_snapshot=True)
            return [source_from_hit(hit) for hit in hits], receipt
        if self.index is None:
            raise ValueError("knowledge_base_not_configured")
        hits = await asyncio.to_thread(self.index.search_chunks, action.query,
            candidate_k=request.candidate_k or self.settings.candidate_k,
            knowledge_base_id=request.knowledge_base_id)
        return [source_from_hit(hit) for hit in hits], {
            "provider": "knowledge_base", "status": "completed", "query": action.query,
            "index": self.settings.opensearch_index,
            "knowledge_base_id": request.knowledge_base_id, "returned": len(hits),
            "snapshot_scope": "returned_index_chunks_not_entire_documents",
        }


class EvidenceCatalog:
    def __init__(self, window, *, pack_adjacent=False):
        self.window = window
        self.pack_adjacent = pack_adjacent
        self.snapshots = {}
        self.units = {}
        self._identities = {}
        self.observations = []
        self.read = {}

    def add(self, source, query, action_id):
        # Ignore rank/retrieval time for exact deduplication, never content or context.
        parent_context = source.metadata.get("context_spans", [])
        if not isinstance(parent_context, list) or any(
                not isinstance(span, dict) or not isinstance(span.get("text"), str)
                for span in parent_context):
            raise ValueError("invalid_parent_context")
        for span in parent_context:
            if span.get("sha256") not in (None, digest(span["text"])):
                raise ValueError("parent_context_hash_mismatch")
        identity_parts = [source.source, source.id, source.document_id, source.uri,
                          source.title, source.snippet, parent_context]
        if source.metadata.get("kind") is not None:
            identity_parts.append({"kind": source.metadata["kind"]})
        identity = dumps(identity_parts)
        snapshot_id = "snapshot:" + digest(identity)
        self.snapshots.setdefault(snapshot_id, {
            "id": snapshot_id, "text_sha256": digest(source.snippet),
            "source": source.model_dump(),
            "scope": "provider_snapshot" if source.source == "web" else "supplied_chunk_or_material",
        })
        ranked, selection = ranked_windows(source.snippet, query, snapshot_id,
            window=self.window, limit=len(source.snippet) + 1, kind=source.metadata.get("kind"),
            pack_adjacent=self.pack_adjacent)
        # IDs reflect coordinate order, not whichever query happened to rank first.
        for node, _ in sorted(ranked, key=lambda pair: pair[0].metadata["source_span"]["start"]):
            key = node.id_
            if key in self._identities:
                continue
            evidence_id = f"E{len(self.units) + 1}"
            self._identities[key] = evidence_id
            metadata = deepcopy(source.metadata)
            metadata.update(deepcopy(node.metadata))
            metadata.update(snapshot_id=snapshot_id, evidence_id=evidence_id,
                            citation_label=int(evidence_id[1:]),
                            offset_scope="snapshot_text_unicode_code_points",
                            parent_source_id=source.id,
                            parent_metadata=deepcopy(source.metadata),
                            # These are recorded source-supplied context, not fabricated text.
                            parent_context_spans=deepcopy(parent_context))
            self.units[evidence_id] = SourceItem(id=key, document_id=source.document_id,
                source=source.source, title=source.title, uri=source.uri, score=source.score,
                snippet=node.text, metadata=metadata)
        ordered = [self._identities[node.id_] for node, _ in ranked]
        self.observations.append({"action_id": action_id, "snapshot_id": snapshot_id,
            "source_observation": source.model_dump(exclude={"snippet"}),
            "ordered_unit_ids": ordered, "selection": selection})
        return ordered

    def candidates(self, queues):
        seen = set()
        for row in zip_longest(*queues):
            for evidence_id in row:
                if evidence_id is not None and evidence_id not in seen and evidence_id not in self.read:
                    seen.add(evidence_id)
                    yield evidence_id

    def material(self, evidence_id):
        source = self.units[evidence_id]
        return {"evidence_id": evidence_id, "citation": f"[资料 {evidence_id[1:]}]",
                "source_id": source.id, "document_id": source.document_id,
                "title": source.title, "uri": source.uri, "text": source.snippet,
                "snapshot_id": source.metadata["snapshot_id"],
                "span": source.metadata["source_span"],
                "context_spans": source.metadata["context_spans"],
                "parent_context_spans": source.metadata["parent_context_spans"],
                **({"kind": source.metadata["kind"]} if source.metadata.get("kind") else {}),
                "content_status": source.metadata.get("content_status")}

    def binding(self, evidence_id):
        source = self.units[evidence_id]
        return {"source_id": source.id, "snapshot_id": source.metadata["snapshot_id"],
                "span": source.metadata["source_span"],
                "citation_label": int(evidence_id[1:]),
                "original_text_in_this_prompt": False}

    def trace(self):
        return {"snapshots": list(self.snapshots.values()),
                "units": {key: value.model_dump() for key, value in self.units.items()},
                "observations": self.observations, "read": deepcopy(self.read),
                "unread_unit_ids": [key for key in self.units if key not in self.read],
                "deduplication": "exact_snapshot_coordinates_and_context_only",
                "semantic_support_verified": False}
