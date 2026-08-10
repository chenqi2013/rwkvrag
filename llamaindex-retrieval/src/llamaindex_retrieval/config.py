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

    opensearch_url: str = "http://127.0.0.1:9200"
    opensearch_username: str | None = None
    opensearch_password: str | None = None
    opensearch_verify_certs: bool = False
    opensearch_index: str = "rwkvrag-knowledge-v1"
    opensearch_shards: int = Field(default=1, ge=1, le=32)
    opensearch_replicas: int = Field(default=0, ge=0, le=8)
    opensearch_refresh_interval: str = "1s"
    opensearch_timeout: int = Field(default=30, ge=1, le=300)
    opensearch_bulk_timeout: int = Field(default=120, ge=1, le=1800)
    opensearch_bulk_size: int = Field(default=500, ge=50, le=5000)
    chunk_size: int = Field(default=512, ge=128, le=8192)
    chunk_overlap: int = Field(default=64, ge=0, le=1024)
    candidate_k: int = Field(default=40, ge=5, le=200)
    default_top_k: int = Field(default=5, ge=1, le=50)
    max_top_k: int = Field(default=20, ge=1, le=100)
    max_chunks_per_document: int = Field(default=1, ge=1, le=10)
    list_query_max_chunks_per_document: int = Field(default=10, ge=1, le=20)
    relative_score_threshold: float = Field(default=0.55, ge=0, le=1)
    min_relevance_score: float = 0

    generation_base_url: str = "http://192.168.0.125:8001/v1"
    generation_models_url: str = "http://192.168.0.125:8001/v1/models"
    generation_password: str = ""
    generation_timeout: int = Field(default=90, ge=5, le=300)
    generation_total_timeout: int = Field(default=45, ge=5, le=120)
    generation_max_tokens: int = Field(default=384, ge=32, le=4096)
    generation_max_evidence_characters: int = Field(default=12_000, ge=1_000, le=48_000)

    mongo_url: str = "mongodb://127.0.0.1:27017"
    mongo_database: str = "rwkvrag_admin"
    sqlite_migration_path: Path = Path("/Volumes/mark/rwkvrag/data/lexical/bm25.sqlite3")
    upload_dir: Path = Path("/Volumes/mark/rwkvrag/data/admin-uploads")
    finewiki_import_roots: str = "/Volumes/mark/rwkvrag/data/deploy-demo/finewiki-sample"
    max_upload_bytes: int = Field(default=100 * 1024 * 1024, ge=1024)
    task_workers: int = Field(default=2, ge=1, le=16)
    cors_origins: str = "http://127.0.0.1:5173,http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def finewiki_root_paths(self) -> list[Path]:
        return [
            Path(value.strip()).expanduser()
            for value in self.finewiki_import_roots.split(",")
            if value.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
