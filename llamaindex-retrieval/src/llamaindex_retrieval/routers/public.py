import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends

from ..config import get_settings
from ..dependencies import search_service
from ..ingest import ingest_markdown
from ..schemas import ImportResponse, MarkdownImportRequest, SearchRequest, SearchResponse
from ..service import SearchService

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/v1/search", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    service: SearchService = Depends(search_service),
) -> SearchResponse:
    return await service.search(payload)


@router.post("/v1/import/markdown", response_model=ImportResponse)
async def import_markdown(payload: MarkdownImportRequest) -> ImportResponse:
    settings = get_settings()
    stats = await asyncio.to_thread(
        ingest_markdown,
        settings,
        Path(payload.path).expanduser(),
        payload.source,
        payload.limit,
        payload.batch_size,
        payload.recreate,
    )
    return ImportResponse(**stats)
