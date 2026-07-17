import hashlib

import qdrant_client
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import models

from .config import Settings
from .sparse import sparse_encoder


def stable_node_id(index: int, document) -> str:
    value = f"{document.id_}:{index}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:32]


class InstructedOpenAIEmbedding(OpenAIEmbedding):
    query_instruction: str = ""

    def _query_text(self, query: str) -> str:
        if not self.query_instruction:
            return query
        return f"Instruct: {self.query_instruction}\nQuery: {query}"

    def _get_query_embedding(self, query: str) -> list[float]:
        return super()._get_query_embedding(self._query_text(query))

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await super()._aget_query_embedding(self._query_text(query))


def create_embedding_model(settings: Settings) -> InstructedOpenAIEmbedding:
    return InstructedOpenAIEmbedding(
        model_name=settings.embedding_model,
        api_base=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        dimensions=settings.embedding_dimensions,
        embed_batch_size=settings.embed_batch_size,
        query_instruction=settings.query_instruction,
        timeout=settings.embedding_timeout_seconds,
    )


def create_qdrant_clients(
    settings: Settings,
) -> tuple[qdrant_client.QdrantClient, qdrant_client.AsyncQdrantClient]:
    kwargs = {"url": settings.qdrant_url, "api_key": settings.qdrant_api_key}
    return qdrant_client.QdrantClient(**kwargs), qdrant_client.AsyncQdrantClient(**kwargs)


def create_vector_store(
    settings: Settings,
    client: qdrant_client.QdrantClient,
    async_client: qdrant_client.AsyncQdrantClient,
) -> QdrantVectorStore:
    encoder = sparse_encoder()
    return QdrantVectorStore(
        collection_name=settings.qdrant_collection,
        client=client,
        aclient=async_client,
        batch_size=64,
        parallel=1,
        enable_hybrid=True,
        sparse_doc_fn=encoder,
        sparse_query_fn=encoder,
        dense_config=models.VectorParams(
            size=settings.embedding_dimensions,
            distance=models.Distance.COSINE,
            on_disk=True,
        ),
        sparse_config=models.SparseVectorParams(
            index=models.SparseIndexParams(on_disk=True),
            modifier=models.Modifier.IDF,
        ),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=False,
            )
        ),
        payload_indexes=[
            {"field_name": "source", "field_schema": models.PayloadSchemaType.KEYWORD},
            {"field_name": "title", "field_schema": models.PayloadSchemaType.KEYWORD},
            {
                "field_name": "file_id",
                "field_schema": models.PayloadSchemaType.KEYWORD,
            },
            {
                "field_name": "knowledge_base_id",
                "field_schema": models.PayloadSchemaType.KEYWORD,
            },
            {
                "field_name": "document_id",
                "field_schema": models.PayloadSchemaType.KEYWORD,
            },
        ],
    )


def create_index(settings: Settings) -> VectorStoreIndex:
    client, async_client = create_qdrant_clients(settings)
    vector_store = create_vector_store(settings, client, async_client)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=create_embedding_model(settings),
    )


def create_splitter(settings: Settings) -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        id_func=stable_node_id,
    )
