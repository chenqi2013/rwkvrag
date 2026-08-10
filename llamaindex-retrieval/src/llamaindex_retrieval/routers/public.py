import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from ..config import get_settings
from ..dependencies import repository, search_service
from ..ingest import ingest_markdown
from ..generation import AnswerGenerationError
from ..schemas import AskResponse, ImportResponse, MarkdownImportRequest, SearchRequest, SearchResponse
from ..repository import MongoRepository
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


@router.post("/v1/ask", response_model=AskResponse)
async def ask(
    payload: SearchRequest,
    service: SearchService = Depends(search_service),
    repo: MongoRepository = Depends(repository),
) -> AskResponse:
    try:
        response = await service.ask(payload)
    except AnswerGenerationError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    await repo.record_search_test_run(
        payload.model_dump(mode="json"),
        response.model_dump(mode="json"),
    )
    return response


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
