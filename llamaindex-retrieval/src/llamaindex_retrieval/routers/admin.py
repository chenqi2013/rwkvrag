import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from ..admin_service import AdminNotFoundError, AdminService, AdminValidationError
from ..config import get_settings
from ..dependencies import admin_service, lexical_index, qdrant_admin, repository
from ..finewiki_browser import FineWikiPathError, browse_finewiki_paths
from ..qdrant_admin import QdrantAdmin
from ..lexical_index import LexicalIndex
from ..repository import MongoRepository
from ..schemas import (
    AdminHealth,
    AliasSwitchRequest,
    ChunkPage,
    CollectionItem,
    FileItem,
    FileUploadAccepted,
    FineWikiImportRequest,
    FineWikiPathPage,
    JobAccepted,
    JobItem,
    KnowledgeBaseCreate,
    KnowledgeBaseItem,
    KnowledgeBaseUpdate,
    SnapshotItem,
)

router = APIRouter(prefix="/v1/admin", tags=["admin"])


@router.get("/health", response_model=AdminHealth)
async def admin_health(service: AdminService = Depends(admin_service)) -> dict:
    return await service.health()


@router.get("/knowledge-bases", response_model=list[KnowledgeBaseItem])
async def list_knowledge_bases(
    repo: MongoRepository = Depends(repository),
) -> list[dict]:
    return await repo.list_knowledge_bases()


@router.post("/knowledge-bases", response_model=KnowledgeBaseItem, status_code=201)
async def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    repo: MongoRepository = Depends(repository),
) -> dict:
    return await repo.create_knowledge_base(payload.name, payload.description)


@router.patch("/knowledge-bases/{knowledge_base_id}", response_model=KnowledgeBaseItem)
async def update_knowledge_base(
    knowledge_base_id: str,
    payload: KnowledgeBaseUpdate,
    repo: MongoRepository = Depends(repository),
) -> dict:
    item = await repo.update_knowledge_base(
        knowledge_base_id,
        payload.model_dump(exclude_none=True),
    )
    if item is None:
        raise AdminNotFoundError(f"知识库不存在：{knowledge_base_id}")
    return item


@router.delete("/knowledge-bases/{knowledge_base_id}", status_code=204)
async def delete_knowledge_base(
    knowledge_base_id: str,
    service: AdminService = Depends(admin_service),
) -> None:
    await service.delete_knowledge_base(knowledge_base_id)


@router.get("/files", response_model=list[FileItem])
async def list_files(
    knowledge_base_id: str | None = None,
    limit: int = Query(default=200, ge=1, le=1000),
    repo: MongoRepository = Depends(repository),
) -> list[dict]:
    return await repo.list_files(knowledge_base_id, limit)


@router.post("/files", response_model=FileUploadAccepted, status_code=202)
async def upload_file(
    file: UploadFile = File(...),
    knowledge_base_id: str = Form(default="default"),
    service: AdminService = Depends(admin_service),
) -> dict:
    file_item, job = await service.upload_file(file, knowledge_base_id)
    return {"file_id": file_item["id"], "job_id": job["id"], "status": job["status"]}


@router.get("/files/{file_id}", response_model=FileItem)
async def get_file(
    file_id: str,
    repo: MongoRepository = Depends(repository),
) -> dict:
    item = await repo.get_file(file_id)
    if item is None:
        raise AdminNotFoundError(f"文件不存在：{file_id}")
    return item


@router.get("/files/{file_id}/download")
async def download_file(
    file_id: str,
    repo: MongoRepository = Depends(repository),
) -> FileResponse:
    item = await repo.get_file(file_id)
    if item is None:
        raise AdminNotFoundError(f"文件不存在：{file_id}")
    return FileResponse(
        path=Path(item["path"]),
        filename=item["filename"],
        media_type=item["content_type"],
    )


@router.delete("/files/{file_id}", status_code=204)
async def delete_file(
    file_id: str,
    service: AdminService = Depends(admin_service),
) -> None:
    await service.delete_file(file_id)


@router.post("/files/{file_id}/reindex", response_model=JobAccepted, status_code=202)
async def reindex_file(
    file_id: str,
    service: AdminService = Depends(admin_service),
) -> dict:
    job = await service.reindex_file(file_id)
    return {"job_id": job["id"], "status": job["status"]}


@router.get("/files/{file_id}/chunks", response_model=ChunkPage)
async def file_chunks(
    file_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: str | None = None,
    index: LexicalIndex = Depends(lexical_index),
) -> dict:
    return await asyncio.to_thread(
        index.list_chunks,
        knowledge_base_id=None,
        file_id=file_id,
        limit=limit,
        offset=offset,
    )


@router.get("/chunks", response_model=ChunkPage)
async def list_chunks(
    knowledge_base_id: str | None = None,
    file_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: str | None = None,
    index: LexicalIndex = Depends(lexical_index),
) -> dict:
    return await asyncio.to_thread(
        index.list_chunks,
        knowledge_base_id=knowledge_base_id,
        file_id=file_id,
        limit=limit,
        offset=offset,
    )


@router.get("/jobs", response_model=list[JobItem])
async def list_jobs(
    status: str | None = None,
    limit: int = Query(default=200, ge=1, le=1000),
    repo: MongoRepository = Depends(repository),
) -> list[dict]:
    return await repo.list_jobs(status, limit)


@router.get("/jobs/{job_id}", response_model=JobItem)
async def get_job(
    job_id: str,
    repo: MongoRepository = Depends(repository),
) -> dict:
    item = await repo.get_job(job_id)
    if item is None:
        raise AdminNotFoundError(f"任务不存在：{job_id}")
    return item


@router.post("/imports/finewiki", response_model=JobAccepted, status_code=202)
async def import_finewiki(
    payload: FineWikiImportRequest,
    service: AdminService = Depends(admin_service),
) -> dict:
    job = await service.create_finewiki_job(payload)
    return {"job_id": job["id"], "status": job["status"]}


@router.get("/imports/finewiki/paths", response_model=FineWikiPathPage)
async def list_finewiki_paths(path: str | None = None) -> dict:
    settings = get_settings()
    try:
        return await asyncio.to_thread(
            browse_finewiki_paths,
            settings.finewiki_root_paths,
            path,
        )
    except FineWikiPathError as error:
        raise AdminValidationError(str(error)) from error


@router.get("/collections", response_model=list[CollectionItem])
async def list_collections(
    qdrant: QdrantAdmin = Depends(qdrant_admin),
) -> list[dict]:
    return await asyncio.to_thread(qdrant.list_collections)


@router.get("/collections/{collection_name}/snapshots", response_model=list[SnapshotItem])
async def list_snapshots(
    collection_name: str,
    qdrant: QdrantAdmin = Depends(qdrant_admin),
) -> list[dict]:
    return await asyncio.to_thread(qdrant.list_snapshots, collection_name)


@router.post(
    "/collections/{collection_name}/snapshots",
    response_model=SnapshotItem,
    status_code=201,
)
async def create_snapshot(
    collection_name: str,
    qdrant: QdrantAdmin = Depends(qdrant_admin),
) -> dict:
    return await asyncio.to_thread(qdrant.create_snapshot, collection_name)


@router.delete(
    "/collections/{collection_name}/snapshots/{snapshot_name}",
    status_code=204,
)
async def delete_snapshot(
    collection_name: str,
    snapshot_name: str,
    qdrant: QdrantAdmin = Depends(qdrant_admin),
) -> None:
    await asyncio.to_thread(qdrant.delete_snapshot, collection_name, snapshot_name)


@router.get("/collections/{collection_name}/snapshots/{snapshot_name}/download")
async def download_snapshot(
    collection_name: str,
    snapshot_name: str,
    qdrant: QdrantAdmin = Depends(qdrant_admin),
) -> StreamingResponse:
    return StreamingResponse(
        qdrant.stream_snapshot(collection_name, snapshot_name),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{snapshot_name}"'},
    )


@router.post("/collections/{collection_name}/snapshots/restore", status_code=202)
async def restore_snapshot(
    collection_name: str,
    snapshot: UploadFile = File(...),
    qdrant: QdrantAdmin = Depends(qdrant_admin),
) -> dict[str, str]:
    await qdrant.restore_snapshot(
        collection_name,
        snapshot.filename or "upload.snapshot",
        snapshot.file,
    )
    await snapshot.close()
    return {"status": "restored", "collection_name": collection_name}


@router.post("/aliases/switch", status_code=200)
async def switch_alias(
    payload: AliasSwitchRequest,
    qdrant: QdrantAdmin = Depends(qdrant_admin),
) -> dict[str, str]:
    await asyncio.to_thread(
        qdrant.switch_alias,
        payload.alias_name,
        payload.collection_name,
    )
    return payload.model_dump()
