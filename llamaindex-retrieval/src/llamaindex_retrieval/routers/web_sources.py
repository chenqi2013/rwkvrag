"""No route can create a server web receipt from user-supplied content."""
from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, StrictBool

router = APIRouter(prefix="/v1/admin/web-snapshots", tags=["admin"])


class SaveWebSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    knowledge_base_id: str = Field(min_length=1, max_length=200)
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmed: StrictBool


@router.get("/{snapshot_id}")
async def preview_web_snapshot(snapshot_id: str, request: Request):
    return await request.app.state.web_snapshot_service.preview(snapshot_id)


@router.post("/{snapshot_id}/save", status_code=202)
async def save_web_snapshot(snapshot_id: str, payload: SaveWebSnapshot, request: Request):
    return await request.app.state.web_snapshot_service.save(snapshot_id,
        payload.knowledge_base_id, payload.expected_sha256, payload.confirmed)
