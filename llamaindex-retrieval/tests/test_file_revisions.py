import asyncio
import hashlib
import io
import os
from unittest.mock import Mock
from uuid import uuid4

from fastapi import UploadFile
import pytest
import pytest_asyncio

import test_index_versions as integration

from llamaindex_retrieval.admin_service import AdminService, AdminConflictError
from llamaindex_retrieval.repository import MongoRepository
from llamaindex_retrieval.tasks import TaskManager

index = integration.index
texts = integration.texts


@pytest_asyncio.fixture
async def setup(index):
    url = os.environ.get("RWKVRAG_TEST_MONGO_URL")
    if not url:
        pytest.skip("set RWKVRAG_TEST_MONGO_URL for isolated database tests")
    name = "rwkvrag_revision_test_" + uuid4().hex
    repo = MongoRepository(url, name)
    try:
        await repo.connect()
        tasks = TaskManager(index.settings, repo, index)
        tasks.submit = Mock()
        service = AdminService(index.settings, repo, tasks, index)
        item, job = await service.upload_file(UploadFile(filename="original.md", file=io.BytesIO(b"# Original\nOld content")), "default")
        await tasks._run_job(job["id"])
        item = await repo.get_file(item["id"])
        assert item["status"] == "ready"
        yield service, repo, tasks, item
    finally:
        await repo.client.drop_database(name)
        await repo.close()


def upload(content=b"# Revised\nNew content"):
    return UploadFile(filename="new.md", file=io.BytesIO(content))


@pytest.mark.asyncio
async def test_revision_publishes_before_updating_download_and_preserves_original(setup, index):
    service, repo, tasks, before = setup
    old_index = index.versions.current()
    job = await service.revise_file(before["id"], upload(), before["sha256"])
    pending = await repo.get_file(before["id"])
    assert pending["path"] == before["path"] and pending["sha256"] == before["sha256"]
    await tasks._run_job(job["id"])
    after = await repo.get_file(before["id"])
    assert after["status"] == "ready" and after["filename"] == "new.md"
    assert after["sha256"] == hashlib.sha256(b"# Revised\nNew content").hexdigest()
    from pathlib import Path
    assert Path(before["path"]).read_bytes() == b"# Original\nOld content"
    assert after["last_indexed_index_version"] == index.versions.current() != old_index
    assert any("Old content" in s for s in texts(index, old_index))
    assert any("New content" in s for s in texts(index))


@pytest.mark.asyncio
async def test_stale_and_parallel_revisions_rejected(setup):
    service, repo, tasks, before = setup
    with pytest.raises(AdminConflictError):
        await service.revise_file(before["id"], upload(), "wrong")
    results = await asyncio.gather(*[service.revise_file(before["id"], upload(f"revision {i}".encode()), before["sha256"])
                                     for i in range(2)], return_exceptions=True)
    assert sum(isinstance(r, dict) for r in results) == 1
    assert sum(isinstance(r, AdminConflictError) for r in results) == 1
    with pytest.raises(AdminConflictError):
        await service.reindex_file(before["id"])
    with pytest.raises(AdminConflictError):
        await service.delete_file(before["id"])


@pytest.mark.asyncio
async def test_parse_failure_keeps_original_ready(setup, index):
    service, repo, tasks, before = setup
    active = index.versions.current()
    job = await service.revise_file(before["id"], upload(b"   "), before["sha256"])
    await tasks._run_job(job["id"])
    after = await repo.get_file(before["id"])
    assert after["status"] == "ready" and after["path"] == before["path"]
    assert after["sha256"] == before["sha256"] and after["error"]
    assert index.versions.current() == active
    assert (await repo.get_job(job["id"]))["status"] == "failed"


@pytest.mark.asyncio
async def test_publication_failure_can_resume_without_overwriting_original_early(setup, index, monkeypatch):
    service, repo, tasks, before = setup
    job = await service.revise_file(before["id"], upload(), before["sha256"])
    update = index.client.indices.update_aliases
    def uncertain(**kwargs):
        update(**kwargs)
        raise TimeoutError("lost publish response")
    with monkeypatch.context() as patch:
        patch.setattr(index.client.indices, "update_aliases", uncertain)
        await tasks._run_job(job["id"])
    failed = await repo.get_file(before["id"])
    assert failed["status"] == "processing" and failed["path"] == before["path"]
    assert any("New content" in s for s in texts(index))
    await service.retry_revision(before["id"])
    with pytest.raises(AdminConflictError):
        await service.retry_revision(before["id"])
    await tasks._run_job(job["id"])
    after = await repo.get_file(before["id"])
    assert after["status"] == "ready" and after["filename"] == "new.md"


@pytest.mark.asyncio
async def test_duplicate_content_reserved_before_publication(setup):
    service, repo, tasks, before = setup
    other, job = await service.upload_file(upload(), "default")
    await tasks._run_job(job["id"])
    with pytest.raises(AdminConflictError):
        await service.revise_file(before["id"], upload(), before["sha256"])
    assert (await repo.get_file(before["id"]))["status"] == "ready"


@pytest.mark.asyncio
async def test_database_writeback_failure_recoverable_after_index_published(setup, index, monkeypatch):
    service, repo, tasks, before = setup
    job = await service.revise_file(before["id"], upload(), before["sha256"])
    update = repo.update_claimed_file
    async def broken(file_id, token, values):
        if values.get("status") == "ready":
            raise RuntimeError("database writeback unavailable")
        await update(file_id, token, values)
    with monkeypatch.context() as patch:
        patch.setattr(repo, "update_claimed_file", broken)
        await tasks._run_job(job["id"])
    assert (await repo.get_file(before["id"]))["path"] == before["path"]
    assert any("New content" in s for s in texts(index))
    await service.retry_revision(before["id"])
    await tasks._run_job(job["id"])
    assert (await repo.get_file(before["id"]))["filename"] == "new.md"


@pytest.mark.asyncio
async def test_restart_recovers_draft_if_job_creation_failed(setup, monkeypatch):
    service, repo, tasks, before = setup
    async def fail(*args, **kwargs):
        raise RuntimeError("connection lost before job creation")
    with monkeypatch.context() as patch:
        patch.setattr(repo, "create_job", fail)
        with pytest.raises(RuntimeError):
            await service.revise_file(before["id"], upload(), before["sha256"])
    pending = await repo.get_file(before["id"])
    assert pending["pending_revision"]["revision_upload"]["filename"] == "new.md"
    assert await repo.get_job(pending["last_job_id"]) is None
    await tasks.start()
    job = await repo.get_job(pending["last_job_id"])
    assert job and job["status"] == "pending"
    await tasks._run_job(job["id"])
    assert (await repo.get_file(before["id"]))["status"] == "ready"


@pytest.mark.asyncio
async def test_http_revision_endpoint_requires_baseline_and_keeps_old_download(setup):
    from fastapi import FastAPI
    import httpx
    from llamaindex_retrieval.routers import admin
    service, repo, tasks, before = setup
    app = FastAPI()
    app.include_router(admin.router)
    app.dependency_overrides[admin.admin_service] = lambda: service
    app.dependency_overrides[admin.repository] = lambda: repo
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        url = f"/v1/admin/files/{before['id']}"
        missing = await client.post(url + "/revisions", files={"file": ("new.md", b"new")})
        assert missing.status_code == 422
        response = await client.post(url + "/revisions", data={"expected_sha256": before["sha256"]},
                                     files={"file": ("new.md", b"new")})
        assert response.status_code == 202
        assert (await client.get(url + "/download")).content == b"# Original\nOld content"
        await tasks._run_job(response.json()["job_id"])
        assert (await client.get(url + "/download")).content == b"new"
