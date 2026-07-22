import json
from collections.abc import Callable

import qdrant_client
from llama_index.core.schema import TextNode

from .config import Settings
from .lexical_index import LexicalIndex


def rebuild_from_qdrant(
    settings: Settings,
    *,
    collection: str | None = None,
    batch_size: int = 256,
    progress_callback: Callable[[int], None] | None = None,
) -> dict[str, int]:
    """Read legacy Qdrant nodes and build the local BM25 index.

    This function only uses Qdrant scroll (read) operations. It never creates,
    updates, deletes, or aliases a Qdrant collection.
    """
    index = LexicalIndex(settings.lexical_index_path)
    index.recreate()
    client = qdrant_client.QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    processed = 0
    indexed = 0
    skipped = 0
    offset = None
    try:
        while True:
            points, offset = client.scroll(
                collection_name=collection or settings.qdrant_collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            nodes: list[TextNode] = []
            for point in points:
                payload = dict(point.payload or {})
                content = payload.get("_node_content")
                if not content:
                    skipped += 1
                    continue
                try:
                    node = TextNode.from_dict(json.loads(str(content)))
                except (TypeError, ValueError, KeyError):
                    skipped += 1
                    continue
                for key, value in payload.items():
                    if key not in {"_node_content", "_node_type"}:
                        node.metadata[key] = value
                nodes.append(node)
            indexed += index.upsert_nodes(nodes)
            processed += len(points)
            if progress_callback is not None:
                progress_callback(processed)
            if offset is None:
                break
    finally:
        client.close()
    return {"points": processed, "indexed": indexed, "skipped": skipped}
