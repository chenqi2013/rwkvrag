import asyncio
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQueryMode,
)

from .config import Settings
from .schemas import SearchRequest, SearchResponse, SourceItem


class SearchService:
    def __init__(self, settings: Settings, index: VectorStoreIndex, reranker: Any = None) -> None:
        self.settings = settings
        self.index = index
        self.reranker = reranker

    async def search(self, request: SearchRequest) -> SearchResponse:
        top_k = min(request.top_k or self.settings.default_top_k, self.settings.max_top_k)
        candidate_k = max(request.candidate_k or self.settings.candidate_k, top_k)
        filters = None
        if request.knowledge_base_id:
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="knowledge_base_id",
                        value=request.knowledge_base_id,
                    )
                ]
            )
        retriever = self.index.as_retriever(
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            similarity_top_k=candidate_k,
            sparse_top_k=candidate_k,
            hybrid_top_k=candidate_k,
            alpha=self.settings.dense_weight,
            filters=filters,
        )
        nodes = await retriever.aretrieve(request.question)
        if self.reranker is not None and nodes:
            nodes = await asyncio.to_thread(
                self.reranker.postprocess_nodes,
                nodes,
                QueryBundle(request.question),
            )
        min_score = (
            request.min_score
            if request.min_score is not None
            else self.settings.min_relevance_score
        )
        filtered = self._select_results(nodes, top_k, min_score)
        return SearchResponse(
            results=[self._source_item(node) for node in filtered],
            retrieval={
                "embedding_model": self.settings.embedding_model,
                "embedding_dimensions": self.settings.embedding_dimensions,
                "mode": "hybrid",
                "reranked": self.reranker is not None,
                "candidate_k": candidate_k,
                "top_k": top_k,
                "returned": len(filtered),
                "max_chunks_per_document": self.settings.max_chunks_per_document,
                "knowledge_base_id": request.knowledge_base_id,
            },
        )

    def _select_results(
        self,
        nodes: list[NodeWithScore],
        top_k: int,
        min_score: float,
    ) -> list[NodeWithScore]:
        if not nodes:
            return []
        top_score = max(float(node.score or 0) for node in nodes)
        relative_floor = max(0, top_score) * self.settings.relative_score_threshold
        score_floor = max(min_score, relative_floor)
        document_counts: dict[str, int] = {}
        selected: list[NodeWithScore] = []
        for node in nodes:
            if float(node.score or 0) < score_floor:
                continue
            document_id = str(
                node.node.metadata.get("document_id") or node.node.ref_doc_id or node.node.node_id
            )
            if document_counts.get(document_id, 0) >= self.settings.max_chunks_per_document:
                continue
            document_counts[document_id] = document_counts.get(document_id, 0) + 1
            selected.append(node)
            if len(selected) >= top_k:
                break
        return selected

    @staticmethod
    def _source_item(result: NodeWithScore) -> SourceItem:
        node = result.node
        metadata = dict(node.metadata)
        full_answer = str(metadata.pop("full_answer", "")).strip()
        title = str(metadata.pop("title", ""))
        source = str(metadata.pop("source", ""))
        uri = metadata.pop("uri", None)
        document_id = str(metadata.pop("document_id", node.ref_doc_id or node.node_id))
        content = node.get_content().strip()
        snippet = full_answer or (
            content if len(content) <= 900 else content[:900].rstrip() + "..."
        )
        return SourceItem(
            id=node.node_id,
            document_id=document_id,
            source=source,
            title=title,
            uri=str(uri) if uri else None,
            score=float(result.score or 0),
            snippet=snippet,
            metadata=metadata,
        )


def create_reranker(settings: Settings):
    if not settings.enable_reranker:
        return None
    try:
        from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
    except ImportError as error:
        raise RuntimeError(
            "reranker dependencies are missing; install with `uv sync --extra rerank`"
        ) from error
    return SentenceTransformerRerank(
        model=settings.reranker_model,
        device=settings.reranker_device,
        top_n=settings.reranker_top_n,
        keep_retrieval_score=True,
    )
