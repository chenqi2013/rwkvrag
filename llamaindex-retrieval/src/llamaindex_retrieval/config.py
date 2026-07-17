from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RWKVRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "rwkvrag-knowledge-current"
    embedding_base_url: str = "http://127.0.0.1:11434/v1"
    embedding_api_key: str = "ollama"
    embedding_model: str = "qwen3-embedding:8b"
    embedding_dimensions: int = Field(default=4096, ge=1)
    query_instruction: str = "为这个问题检索能够直接回答的百科资料。"
    embed_batch_size: int = Field(default=16, ge=1, le=256)
    embedding_timeout_seconds: float = Field(default=300, gt=0)

    chunk_size: int = Field(default=512, ge=128, le=8192)
    chunk_overlap: int = Field(default=64, ge=0, le=1024)
    dense_weight: float = Field(default=0.8, ge=0, le=1)
    candidate_k: int = Field(default=40, ge=5, le=200)
    default_top_k: int = Field(default=5, ge=1, le=50)
    max_top_k: int = Field(default=20, ge=1, le=100)
    max_chunks_per_document: int = Field(default=1, ge=1, le=10)
    relative_score_threshold: float = Field(default=0.55, ge=0, le=1)
    min_relevance_score: float = 0

    enable_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str | None = "mps"
    reranker_top_n: int = Field(default=20, ge=1, le=100)

    mongo_url: str = "mongodb://127.0.0.1:27017"
    mongo_database: str = "rwkvrag_admin"
    upload_dir: Path = Path("/Volumes/mark/rwkvrag/data/admin-uploads")
    max_upload_bytes: int = Field(default=100 * 1024 * 1024, ge=1024)
    task_workers: int = Field(default=2, ge=1, le=16)
    cors_origins: str = "http://127.0.0.1:5173,http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
