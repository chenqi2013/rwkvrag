from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, ge=1, le=100)
    candidate_k: int | None = Field(default=None, ge=5, le=200)
    min_score: float | None = None
    knowledge_base_id: str | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty")
        return value


class SourceItem(BaseModel):
    id: str
    document_id: str
    source: str
    title: str
    uri: str | None = None
    score: float
    snippet: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SourceItem]
    retrieval: dict[str, Any]


class MarkdownImportRequest(BaseModel):
    path: str
    source: str = "local-markdown"
    limit: int = Field(default=0, ge=0)
    batch_size: int = Field(default=16, ge=1, le=256)
    recreate: bool = False


class ImportResponse(BaseModel):
    documents: int
    nodes: int


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)


class KnowledgeBaseUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)


class KnowledgeBaseItem(BaseModel):
    id: str
    name: str
    description: str = ""
    created_at: datetime
    updated_at: datetime
    file_count: int = 0


FileStatus = Literal["pending", "processing", "ready", "failed", "deleting"]
JobStatus = Literal["pending", "running", "completed", "failed"]
JobKind = Literal["file_ingest", "file_reindex", "finewiki_import"]


class FileItem(BaseModel):
    id: str
    knowledge_base_id: str
    filename: str
    content_type: str
    extension: str
    size: int
    sha256: str
    source: str
    status: FileStatus
    node_count: int = 0
    error: str | None = None
    last_job_id: str | None = None
    created_at: datetime
    updated_at: datetime


class JobItem(BaseModel):
    id: str
    kind: JobKind
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    stage: str
    message: str = ""
    documents_processed: int = 0
    nodes_processed: int = 0
    error: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class JobAccepted(BaseModel):
    job_id: str
    status: JobStatus = "pending"


class FileUploadAccepted(JobAccepted):
    file_id: str


class FineWikiImportRequest(BaseModel):
    path: str
    knowledge_base_id: str = "default"
    source: str = "finewiki-zh"
    titles: list[str] = Field(default_factory=list)
    limit: int = Field(default=0, ge=0)
    batch_size: int = Field(default=8, ge=1, le=128)
    recreate: bool = False


class FineWikiPathEntry(BaseModel):
    name: str
    path: str
    type: Literal["directory", "parquet"]
    size: int | None = None


class FineWikiPathPage(BaseModel):
    current: str
    parent: str | None = None
    roots: list[str]
    entries: list[FineWikiPathEntry]


class ChunkItem(BaseModel):
    id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkPage(BaseModel):
    items: list[ChunkItem]
    next_offset: str | None = None


class AdminHealth(BaseModel):
    status: Literal["ok", "degraded"]
    mongodb: dict[str, Any]
    lexical: dict[str, Any]
