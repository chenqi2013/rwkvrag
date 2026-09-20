"""Real Mongo/OpenSearch integration with isolated fixture-owned database/index."""
from copy import deepcopy
from types import SimpleNamespace

from fastapi import FastAPI
import httpx
from llama_index.core.schema import TextNode
import pytest

import test_file_revisions as revisions
from test_atomic_evidence import Model
from llamaindex_retrieval.atomic_evidence import AtomicEvidenceService, AtomicRequest
from llamaindex_retrieval.repository import RepositoryConflictError
from llamaindex_retrieval.routers.admin import router
from llamaindex_retrieval.admin_service import AdminNotFoundError
from fastapi.responses import JSONResponse

index = revisions.index
setup = revisions.setup


@pytest.mark.asyncio
async def test_real_retrieval_snapshot_persistence_immutability_and_scope(setup, index):
    _, repo, _, _ = setup
    index.upsert_nodes([TextNode(id_="atom-source", text="设备甲功率 18 W。", metadata={
        "knowledge_base_id": "default", "title": "设备甲", "source_sha256": "old-sha", "source": "synthetic"})])
    index.client.indices.refresh(index=index.index_name)
    svc = AtomicEvidenceService(index.settings, repo, index, Model(['[1]']), reader=Model(['{"answer":"YES"}']))
    run = await svc.inspect("default", AtomicRequest(object="设备甲", attribute="功率"))
    assert run["status"] == "completed" and run["claims"][0]["statement_quote"] == "设备甲功率 18 W。"
    restored = await repo.get_atomic_run("default", run["id"])
    assert restored == run
    assert await repo.get_atomic_run("other-kb", run["id"]) is None
    assert len(await repo.list_atomic_runs("default")) == 1
    changed = deepcopy(run)
    changed["claims"][0]["statement_quote"] = "999 W"
    with pytest.raises(RepositoryConflictError):
        await repo.finish_atomic_run(changed)
    assert (await repo.get_atomic_run("default", run["id"]))["claims"][0]["statement_quote"] == "设备甲功率 18 W。"
    # Source update never changes the historical evidence snapshot.
    index.upsert_nodes([TextNode(id_="atom-source", text="设备甲功率 25 W。", metadata={"knowledge_base_id": "default"})])
    assert (await repo.get_atomic_run("default", run["id"]))["sources"][0]["snippet"] == "设备甲功率 18 W。"


@pytest.mark.asyncio
async def test_scoped_history_route_and_request_validation(setup, index):
    _, repo, _, _ = setup
    svc = AtomicEvidenceService(index.settings, repo, index, Model([]), reader=Model([]))
    app = FastAPI()
    app.state.atomic_service = svc
    app.include_router(router)
    @app.exception_handler(AdminNotFoundError)
    async def missing(request, error):
        return JSONResponse(status_code=404, content={"detail": str(error)})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/v1/admin/atomic-evidence/capabilities")).json() == {"available": True}
        assert (await client.get("/v1/admin/knowledge-bases/default/atomic-evidence")).json() == []
        assert (await client.get("/v1/admin/knowledge-bases/other/atomic-evidence")).status_code == 404
        assert (await client.get("/v1/admin/knowledge-bases/default/atomic-evidence/unknown")).status_code == 404
        bad = await client.post("/v1/admin/knowledge-bases/default/atomic-evidence", json={"object": " ", "attribute": "功率", "base_url": "http://example.com"})
        assert bad.status_code == 422
