import json
from collections.abc import AsyncIterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any, BinaryIO, Iterator
from urllib.parse import quote

import httpx
import qdrant_client
from qdrant_client import models
from qdrant_client.http.exceptions import ResponseHandlingException

from .config import Settings


class QdrantUnavailableError(ConnectionError):
    pass


class QdrantAdmin:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @contextmanager
    def _client(self) -> Iterator[qdrant_client.QdrantClient]:
        client = qdrant_client.QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
        )
        try:
            yield client
        except ResponseHandlingException as error:
            raise QdrantUnavailableError("Qdrant 暂时不可用，请稍后重试") from error
        finally:
            client.close()

    def _headers(self) -> dict[str, str]:
        if not self.settings.qdrant_api_key:
            return {}
        return {"api-key": self.settings.qdrant_api_key}

    def health(self) -> dict[str, Any]:
        with self._client() as client:
            collections = client.get_collections().collections
        return {
            "ok": True,
            "url": self.settings.qdrant_url,
            "collections": len(collections),
            "active_collection": self.settings.qdrant_collection,
        }

    def list_collections(self) -> list[dict[str, Any]]:
        with self._client() as client:
            aliases = client.get_aliases().aliases
            aliases_by_collection: dict[str, list[str]] = {}
            for alias in aliases:
                aliases_by_collection.setdefault(alias.collection_name, []).append(alias.alias_name)
            items: list[dict[str, Any]] = []
            for collection in client.get_collections().collections:
                info = client.get_collection(collection.name)
                vectors = info.config.params.vectors
                dense_dimensions = None
                if isinstance(vectors, dict):
                    dense = vectors.get("text-dense") or next(iter(vectors.values()), None)
                    dense_dimensions = dense.size if dense is not None else None
                elif vectors is not None:
                    dense_dimensions = vectors.size
                items.append(
                    {
                        "name": collection.name,
                        "status": str(info.status),
                        "points_count": int(info.points_count or 0),
                        "indexed_vectors_count": int(info.indexed_vectors_count or 0),
                        "dense_dimensions": dense_dimensions,
                        "aliases": sorted(aliases_by_collection.get(collection.name, [])),
                    }
                )
        return sorted(items, key=lambda item: item["name"])

    def delete_points(self, key: str, value: str) -> None:
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))]
            )
        )
        with self._client() as client:
            client.delete(
                collection_name=self.settings.qdrant_collection,
                points_selector=selector,
                wait=True,
            )

    def is_alias(self, name: str) -> bool:
        with self._client() as client:
            aliases = client.get_aliases().aliases
        return any(alias.alias_name == name for alias in aliases)

    def list_chunks(
        self,
        *,
        knowledge_base_id: str | None,
        file_id: str | None,
        limit: int,
        offset: str | None,
    ) -> dict[str, Any]:
        conditions = []
        if knowledge_base_id:
            conditions.append(
                models.FieldCondition(
                    key="knowledge_base_id",
                    match=models.MatchValue(value=knowledge_base_id),
                )
            )
        if file_id:
            conditions.append(
                models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id))
            )
        with self._client() as client:
            records, next_offset = client.scroll(
                collection_name=self.settings.qdrant_collection,
                scroll_filter=models.Filter(must=conditions) if conditions else None,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
        items = []
        for record in records:
            payload = dict(record.payload or {})
            node_content = payload.pop("_node_content", "")
            payload.pop("_node_type", None)
            text = ""
            if node_content:
                try:
                    text = str(json.loads(node_content).get("text") or "")
                except (TypeError, ValueError):
                    text = ""
            items.append(
                {
                    "id": str(record.id),
                    "document_id": str(payload.get("document_id") or payload.get("doc_id") or ""),
                    "text": text,
                    "metadata": payload,
                }
            )
        return {
            "items": items,
            "next_offset": str(next_offset) if next_offset is not None else None,
        }

    def create_snapshot(self, collection_name: str) -> dict[str, Any]:
        with self._client() as client:
            snapshot = client.create_snapshot(collection_name=collection_name, wait=True)
        if snapshot is None:
            raise RuntimeError("Qdrant 未返回 snapshot 信息")
        return self._snapshot_item(snapshot)

    def list_snapshots(self, collection_name: str) -> list[dict[str, Any]]:
        with self._client() as client:
            snapshots = client.list_snapshots(collection_name)
        return [self._snapshot_item(item) for item in snapshots]

    def delete_snapshot(self, collection_name: str, snapshot_name: str) -> None:
        with self._client() as client:
            client.delete_snapshot(collection_name, snapshot_name, wait=True)

    @staticmethod
    def _snapshot_item(snapshot: Any) -> dict[str, Any]:
        created_at = snapshot.creation_time
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        return {
            "name": snapshot.name,
            "size": int(snapshot.size or 0),
            "created_at": created_at,
            "checksum": snapshot.checksum,
        }

    def switch_alias(self, alias_name: str, collection_name: str) -> None:
        with self._client() as client:
            if not client.collection_exists(collection_name):
                raise ValueError(f"collection 不存在：{collection_name}")
            aliases = client.get_aliases().aliases
            operations: list[Any] = []
            if any(alias.alias_name == alias_name for alias in aliases):
                operations.append(
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(alias_name=alias_name)
                    )
                )
            operations.append(
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=collection_name,
                        alias_name=alias_name,
                    )
                )
            )
            client.update_collection_aliases(operations)

    async def stream_snapshot(
        self,
        collection_name: str,
        snapshot_name: str,
    ) -> AsyncIterator[bytes]:
        collection = quote(collection_name, safe="")
        snapshot = quote(snapshot_name, safe="")
        url = f"{self.settings.qdrant_url}/collections/{collection}/snapshots/{snapshot}"
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("GET", url, headers=self._headers()) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        yield chunk
            except httpx.RequestError as error:
                raise QdrantUnavailableError("Qdrant 暂时不可用，请稍后重试") from error

    async def restore_snapshot(
        self,
        collection_name: str,
        filename: str,
        file_object: BinaryIO,
    ) -> None:
        collection = quote(collection_name, safe="")
        url = f"{self.settings.qdrant_url}/collections/{collection}/snapshots/upload"
        params = {"wait": "true", "priority": "snapshot"}
        files = {"snapshot": (filename, file_object, "application/octet-stream")}
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                response = await client.post(
                    url,
                    params=params,
                    files=files,
                    headers=self._headers(),
                )
            except httpx.RequestError as error:
                raise QdrantUnavailableError("Qdrant 暂时不可用，请稍后重试") from error
            response.raise_for_status()
