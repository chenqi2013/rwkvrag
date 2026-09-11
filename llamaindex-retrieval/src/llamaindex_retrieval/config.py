from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, StrictInt, model_validator
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
    chunk_overlap: int = Field(default=64, ge=1, le=1024)
    candidate_k: int = Field(default=40, ge=5, le=200)
    default_top_k: int = Field(default=5, ge=1, le=50)
    max_top_k: int = Field(default=20, ge=1, le=100)
    max_chunks_per_document: int = Field(default=1, ge=1, le=10)
    list_query_max_chunks_per_document: int = Field(default=10, ge=1, le=20)
    relative_score_threshold: float = Field(default=0.55, ge=0, le=1)
    min_relevance_score: float = 0

    # The existing pipeline remains available for regression comparisons.
    rag_pipeline: str = Field(default="rwkv", pattern="^(existing|rwkv)$")
    native_base_url: str = "http://127.0.0.1:18421/v1"
    native_model: str = "rwkv7-g1j-2.9b-20260831-ctx16384"
    native_api_key: str = ""
    native_transport: Literal["native", "rwkvos_batch"] = "native"
    native_writer_prefill: Literal["<think", "<think></think"] = "<think"
    native_writer_prompt_protocol: Literal["task_first", "evidence_first"] = "task_first"
    rwkvos_cf_access_client_id: SecretStr = SecretStr("")
    rwkvos_cf_access_client_secret: SecretStr = SecretStr("")
    rwkvos_prefill_mode: Literal["complete", "continuation"] = "complete"
    rwkvos_state_id: str | None = None
    rwkvos_reader_state_id: str | None = None
    rwkvos_reader_prompt_protocol: Literal["legacy", "rwkv_g1j_no_think_v1"] = "legacy"
    rwkvos_writer_prompt_protocol: Literal["legacy", "rwkv_g1j_no_think_v1"] = "legacy"
    rwkvos_reader_input_layout: Literal["original", "task_last"] = "original"
    rwkvos_stop_tokens: list[StrictInt] | None = None
    rwkvos_count_input_tokens: bool = False
    rwkvos_input_token_limit: StrictInt | None = Field(default=None, ge=1)
    rwkvos_batch_size: int = Field(default=8, ge=1, le=399)
    rwkvos_batch_wait_ms: float = Field(default=5, ge=0, le=1000)
    native_timeout_seconds: int = Field(default=180, ge=5, le=1800)
    native_context_window_tokens: int = Field(default=16384, ge=1024)
    native_max_concurrency: int = Field(default=32, ge=1, le=256)
    native_planner_prefill: Literal["<think", "<think></think"] = "<think"
    native_plan_protocol: Literal["queries_fields", "shared_tasks"] = "queries_fields"
    native_resolver_prefill: Literal["<think", "<think></think"] = "<think"
    native_resolver_protocol: Literal["fields", "task_units", "binary_query"] = "fields"
    native_resolver_task_grouping: Literal["joint", "individual"] = "joint"
    native_resolver_format_repair: bool = False
    native_task_source: Literal["fields", "queries"] = "fields"
    native_candidate_order: Literal["rrf", "query_round_robin"] = "rrf"
    native_planner_max_tokens: int = Field(default=1024, ge=64, le=4096)
    native_resolver_max_tokens: int = Field(default=1024, ge=32, le=4096)
    native_resolver_sources: int = Field(default=24, ge=1, le=200)
    native_resolver_budget_scope: Literal["global", "per_query"] = "global"
    native_retrieval_scope: Literal["chunks", "documents"] = "chunks"
    native_document_limit: int = Field(default=5, ge=1, le=20)
    native_resolver_window_characters: int = Field(default=1200, ge=256, le=8000)
    native_resolver_overlap_characters: int = Field(default=180, ge=1, le=2000)
    native_resolver_batch_characters: int = Field(default=6000, ge=1000, le=48000)
    native_max_queries: int = Field(default=6, ge=1, le=16)
    native_max_fields: int = Field(default=12, ge=1, le=32)
    native_ingest_chunk_characters: int = Field(default=2400, ge=256, le=24000)
    native_ingest_overlap_characters: int = Field(default=180, ge=1, le=2000)

    generation_base_url: str = "http://192.168.0.125:8002/v1"
    generation_models_url: str = "http://192.168.0.125:8002/v1/models"
    generation_password: str = ""
    generation_timeout: int = Field(default=90, ge=5, le=300)
    generation_total_timeout: int = Field(default=20, ge=5, le=120)
    generation_max_tokens: int = Field(default=2_048, ge=32, le=4096)
    generation_max_evidence_characters: int = Field(default=12_000, ge=1_000, le=48_000)
    generation_output_mode: str = Field(default="legacy", pattern="^(immutable|legacy)$")
    answer_point_fanout_enabled: bool = False
    answer_point_fanout_concurrency: int = Field(default=3, ge=1, le=8)

    @model_validator(mode="after")
    def validate_reader_state_protocol(self) -> "Settings":
        if self.native_resolver_budget_scope == "per_query" and (
                self.native_task_source != "queries"
                or self.native_resolver_task_grouping != "individual"
                or self.native_resolver_protocol not in {"task_units", "binary_query"}):
            raise ValueError("Per-query Reader budget requires individual task selection and query tasks")
        if self.native_resolver_protocol == "binary_query" and (
                self.native_task_source != "queries"
                or self.native_resolver_task_grouping != "individual"
                or self.rwkvos_reader_input_layout != "original"
                or self.rwkvos_reader_state_id is not None
                or self.rwkvos_state_id is not None):
            raise ValueError("Binary Reader requires individual query tasks, original layout and no legacy state")
        if self.native_resolver_format_repair and self.native_resolver_protocol != "task_units":
            raise ValueError("Reader format repair requires task_units")
        if self.rwkvos_writer_prompt_protocol == "rwkv_g1j_no_think_v1":
            if (self.native_transport != "rwkvos_batch" or self.rwkvos_prefill_mode != "complete"
                    or self.native_writer_prefill != "<think></think"):
                raise ValueError("Canonical Writer protocol requires batch and complete no-think")
        if (self.rwkvos_reader_input_layout != "original"
                and self.rwkvos_reader_prompt_protocol != "rwkv_g1j_no_think_v1"):
            raise ValueError("Reader task_last layout requires canonical prompt protocol")
        if self.rwkvos_reader_state_id is not None:
            if not self.rwkvos_reader_state_id.strip():
                raise ValueError("rwkvos_reader_state_id must be nonempty")
            if self.rwkvos_reader_prompt_protocol != "rwkv_g1j_no_think_v1":
                raise ValueError("Reader state requires its canonical prompt protocol")
        if self.rwkvos_reader_prompt_protocol == "rwkv_g1j_no_think_v1":
            if (self.native_transport != "rwkvos_batch" or self.rwkvos_prefill_mode != "complete"
                    or self.native_resolver_prefill != "<think></think"
                    or self.native_resolver_protocol not in {"task_units", "binary_query"}):
                raise ValueError("Canonical Reader protocol requires batch, complete no-think and supported selection protocol")
        return self

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.native_resolver_overlap_characters >= self.native_resolver_window_characters:
            raise ValueError("native resolver overlap must be smaller than window")
        if self.native_ingest_overlap_characters >= self.native_ingest_chunk_characters:
            raise ValueError("native ingest overlap must be smaller than chunk")
        if self.rwkvos_input_token_limit is not None and not self.rwkvos_count_input_tokens:
            raise ValueError("rwkvos_input_token_limit requires rwkvos_count_input_tokens")
        return self
    ask_total_timeout: int = Field(default=45, ge=10, le=180)
    ask_generation_reserve: int = Field(default=12, ge=3, le=60)
    answer_verification_min_budget: int = Field(default=15, ge=3, le=60)
    semantic_pipeline_enabled: bool = False
    model_query_planning_enabled: bool = True
    model_query_planning_timeout: int = Field(default=10, ge=3, le=120)
    model_query_planning_max_tokens: int = Field(default=256, ge=64, le=1024)
    model_query_planning_max_queries: int = Field(default=6, ge=3, le=10)
    model_query_planning_cache_ttl: int = Field(default=600, ge=0, le=86_400)
    document_reranking_enabled: bool = True
    document_reranking_timeout: int = Field(default=10, ge=3, le=120)
    document_reranking_max_tokens: int = Field(default=96, ge=32, le=512)
    document_reranking_max_documents: int = Field(default=5, ge=2, le=20)
    document_reranking_max_characters: int = Field(default=1_600, ge=500, le=8_000)
    document_reranking_concurrency: int = Field(default=2, ge=1, le=10)
    evidence_extraction_enabled: bool = True
    evidence_extraction_timeout: int = Field(default=15, ge=3, le=120)
    evidence_extraction_max_tokens: int = Field(default=384, ge=64, le=2048)
    evidence_extraction_max_sources: int = Field(default=6, ge=1, le=20)
    evidence_extraction_max_source_characters: int = Field(default=2_500, ge=500, le=24_000)
    evidence_extraction_concurrency: int = Field(default=2, ge=1, le=10)
    active_retrieval_enabled: bool = True
    active_retrieval_max_rounds: int = Field(default=1, ge=1, le=4)
    active_retrieval_max_queries: int = Field(default=3, ge=1, le=5)
    active_retrieval_timeout: int = Field(default=10, ge=3, le=120)
    active_retrieval_max_tokens: int = Field(default=192, ge=64, le=1024)
    active_retrieval_max_evidence_characters: int = Field(default=1_500, ge=1_000, le=24_000)
    active_retrieval_max_results: int = Field(default=20, ge=5, le=50)

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
    if filename := os.environ.get("RWKVRAG_SETTINGS_FILE"):
        return Settings(_env_file=None, **json.loads(Path(filename).read_text()))
    return Settings()
