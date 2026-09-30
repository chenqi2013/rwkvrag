import asyncio
import copy
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient

from llamaindex_retrieval.admin_service import (
    AdminNotFoundError,
    AdminService,
    AdminConflictError,
    AdminValidationError,
)
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.repository import RepositoryConflictError
from llamaindex_retrieval.routers.web_sources import router
from llamaindex_retrieval.source_revisions import prepare_revision
from llamaindex_retrieval.tasks import TaskManager
from llamaindex_retrieval.web_retrieval import WebSearchAdapter
from llamaindex_retrieval.web_sources import WebSnapshotService, text_sha


class Repo:
    def __init__(self):
        self.snapshots, self.files, self.jobs = {}, {}, {}
        self.fail_job = False

    async def claim_index_job(self, key):
        if self.jobs[key]["status"] != "pending":
            return False
        self.jobs[key]["status"] = "running"
        return True

    async def get_knowledge_base(self, key):
        return {"id": key} if key == "kb" else None

    async def record_web_snapshot(self, item):
        identity = str(len(self.snapshots))
        self.snapshots[identity] = {**copy.deepcopy(item), "id": identity}
        return identity

    async def get_web_snapshot(self, key):
        return copy.deepcopy(self.snapshots.get(key))

    async def get_file(self, key):
        return self.files.get(key)

    async def create_file(self, item):
        if any(f["sha256"] == item["sha256"] for f in self.files.values()):
            raise RepositoryConflictError("duplicate bytes")
        self.files[item["id"]] = {**item, "status": "pending", "last_job_id": None}
        return self.files[item["id"]]

    async def update_file(self, key, patch):
        self.files[key].update(patch)
        return self.files[key]

    async def create_job(self, kind, payload, job_id=None):
        if self.fail_job:
            raise RuntimeError("database unavailable")
        key = job_id or str(len(self.jobs))
        self.jobs.setdefault(key, dict(id=key, kind=kind, payload=payload, status="pending"))
        return self.jobs[key]

    async def get_job(self, key):
        return self.jobs.get(key)

    async def update_job(self, key, patch):
        self.jobs[key].update(patch)


def snapshot(
    text=" \n# 原始页面\n\n完整内容，而不是回答。\n ",
    url="https://example.com/doc?a=1",
    status="fetched",
):
    return dict(
        title="原始页面",
        url=url,
        text=text,
        sha256=text_sha(text),
        retrieved_at="2026-09-24T00:00:00Z",
        content_status=status,
        provider="tavily",
    )


@pytest.fixture
def setup(tmp_path):
    repo = Repo()
    tasks = SimpleNamespace(submit=Mock())
    settings = Settings(_env_file=None, upload_dir=tmp_path / "uploads")
    admin = AdminService(settings, repo, tasks, None)
    return WebSnapshotService(repo, admin), repo, admin


@pytest.mark.asyncio
async def test_confirm_save_exact_snapshot_and_provenance_then_idempotent(setup):
    service, repo, admin = setup
    original = snapshot()
    identity = await service.record(original)
    view = await service.preview(identity)
    assert view["save_allowed"] and view["text"] == original["text"]
    first = await service.save(identity, "kb", view["sha256"], True)
    item = repo.files[first["file_id"]]
    assert Path(item["path"]).read_bytes() == original["text"].encode()
    docs, revision = prepare_revision(
        admin.settings, Path(item["path"]), item["id"], "kb", item["sha256"], item["web_provenance"]
    )
    assert docs[0].text == original["text"]
    assert docs[0].metadata["uri"] == original["url"]
    assert docs[0].metadata["web_provenance"]["sha256"] == original["sha256"]
    parsed = __import__("json").loads(
        Path(revision["parsed_snapshot_uri"].removeprefix("file://")).read_text()
    )
    assert parsed["documents"][0]["metadata"]["uri"] == original["url"]
    second_receipt = await service.record({**original, "retrieved_at": "2026-09-25T00:00:00Z"})
    second = await service.save(second_receipt, "kb", view["sha256"], True)
    assert second["existing"] and second["file_id"] == first["file_id"]
    assert len(repo.files) == len(repo.jobs) == 1
    assert admin.tasks.submit.call_count == 2


@pytest.mark.asyncio
async def test_snippet_bad_hash_and_confirmation_never_create_files(setup):
    service, repo, _ = setup
    s = snapshot(status="snippet_only")
    identity = await service.record(s)
    assert not (await service.preview(identity))["save_allowed"]
    with pytest.raises(AdminValidationError):
        await service.save(identity, "kb", s["sha256"], True)
    s = snapshot()
    identity = await service.record(s)
    with pytest.raises(AdminValidationError):
        await service.save(identity, "kb", s["sha256"], False)
    with pytest.raises(AdminConflictError):
        await service.save(identity, "kb", "0" * 64, True)
    repo.snapshots[identity]["text"] = "tampered"
    with pytest.raises(AdminConflictError):
        await service.preview(identity)
    assert not repo.files and not repo.jobs


@pytest.mark.asyncio
async def test_changed_snapshot_new_file_and_failed_job_never_reported_completed(setup):
    service, repo, _ = setup
    first = snapshot()
    identity = await service.record(first)
    saved = await service.save(identity, "kb", first["sha256"], True)
    repo.jobs[saved["job_id"]]["status"] = "failed"
    again = await service.save(identity, "kb", first["sha256"], True)
    assert again["existing"] and again["status"] == "failed"
    newer = snapshot(text="new version")
    newid = await service.record(newer)
    saved2 = await service.save(newid, "kb", newer["sha256"], True)
    assert saved2["file_id"] != saved["file_id"] and len(repo.files) == 2
    assert Path(repo.files[saved["file_id"]]["path"]).read_text() == first["text"]


@pytest.mark.asyncio
async def test_database_failure_before_job_retry_resumes_without_duplicate(setup):
    service, repo, admin = setup
    s = snapshot()
    identity = await service.record(s)
    repo.fail_job = True
    with pytest.raises(RuntimeError, match="database unavailable"):
        await service.save(identity, "kb", s["sha256"], True)
    assert len(repo.files) == 1 and not repo.jobs
    admin.tasks.submit.assert_not_called()
    repo.fail_job = False
    result = await service.save(identity, "kb", s["sha256"], True)
    assert result["existing"] and result["status"] == "pending"
    assert len(repo.files) == len(repo.jobs) == 1
    admin.tasks.submit.assert_called_once()


@pytest.mark.asyncio
async def test_same_bytes_other_url_cannot_relabel_existing_source(setup):
    service, repo, _ = setup
    s = snapshot()
    identity = await service.record(s)
    await service.save(identity, "kb", s["sha256"], True)
    other = await service.record(snapshot(url="https://example.net/different"))
    with pytest.raises(AdminConflictError):
        await service.save(other, "kb", s["sha256"], True)
    assert len(repo.files) == 1
    assert next(iter(repo.files.values()))["web_provenance"]["url"] == s["url"]


@pytest.mark.asyncio
async def test_adapter_records_full_provider_snapshot_not_reader_and_fails_closed():
    settings = Settings(_env_file=None)
    adapter = WebSearchAdapter(settings)
    s = snapshot(text="Long original provider body")
    adapter._execute = AsyncMock(
        return_value={
            "provider": "tavily",
            "source_hashes": {},
            "hits": [
                {
                    **s,
                    "snapshot": s["text"],
                    "reader_text": "Long",
                    "published_date": "",
                    "material_limited": True,
                    "error": "",
                    "score": 1.0,
                }
            ],
        }
    )
    adapter.snapshot_recorder = AsyncMock(return_value="server-receipt")
    hits, trace = await adapter.search("question")
    assert hits[0].text == "Long" and hits[0].metadata["web_snapshot_id"] == "server-receipt"
    assert adapter.snapshot_recorder.call_args.args[0]["text"] == s["text"]
    assert trace["snapshots"][0]["text"] == s["text"]
    adapter.snapshot_recorder = AsyncMock(side_effect=RuntimeError("storage failure"))
    with pytest.raises(RuntimeError):
        await adapter.search("question")


def test_http_contract_forbids_client_snapshot_or_truthy_confirmation(setup):
    service, _, _ = setup
    app = FastAPI()
    app.include_router(router)
    app.state.web_snapshot_service = service
    with TestClient(app) as client:
        for changes in [{"text": "answer"}, {"url": "https://fake.test"}, {"confirmed": "true"}]:
            result = client.post(
                "/v1/admin/web-snapshots/arbitrary/save",
                json={
                    "knowledge_base_id": "kb",
                    "expected_sha256": "a" * 64,
                    "confirmed": True,
                    **changes,
                },
            )
            assert result.status_code == 422
        assert client.post("/v1/admin/web-snapshots", json=snapshot()).status_code == 404


@pytest.mark.asyncio
async def test_manual_revision_clears_web_binding_only_on_success(setup):
    service, repo, admin = setup
    s = snapshot()
    identity = await service.record(s)
    saved = await service.save(identity, "kb", s["sha256"], True)
    item = repo.files[saved["file_id"]]
    item["status"] = "ready"
    repo.claim_file_operation = AsyncMock(return_value=True)
    repo.update_claimed_file = AsyncMock()
    job = await admin.revise_file(
        item["id"],
        UploadFile(filename="edit.md", file=io.BytesIO(b"edited manually")),
        item["sha256"],
    )
    assert job["payload"]["revision_upload"]["web_provenance"] is None
    assert item["web_provenance"]["url"] == s["url"]  # Existing published revision unchanged.
    draft = job["payload"]["revision_upload"]
    docs, _ = prepare_revision(
        admin.settings,
        Path(draft["path"]),
        item["id"],
        "kb",
        draft["sha256"],
        draft["web_provenance"],
    )
    assert "web_provenance" not in docs[0].metadata
    assert docs[0].metadata["uri"].startswith("file://")


@pytest.mark.asyncio
async def test_index_jobs_wait_and_cancel_queue_without_failing(tmp_path):
    repo = Repo()
    settings = Settings(_env_file=None, upload_dir=tmp_path, task_workers=3)
    manager = TaskManager(settings, repo, None)
    entered, release = asyncio.Event(), asyncio.Event()
    active = 0
    seen = []

    async def run(key, payload):
        nonlocal active
        active += 1
        assert active == 1
        seen.append(key)
        if key == "one":
            entered.set()
            await release.wait()
        await repo.update_job(key, {"status": "completed"})
        active -= 1

    manager._run_file_job = run
    manager._run_finewiki_job = run
    for key, kind in [
        ("one", "file_ingest"),
        ("two", "file_reindex"),
        ("three", "finewiki_import"),
    ]:
        await repo.create_job(kind, {}, job_id=key)
    one = asyncio.create_task(manager._run_job("one"))
    await entered.wait()
    two = asyncio.create_task(manager._run_job("two"))
    three = asyncio.create_task(manager._run_job("three"))
    await asyncio.sleep(0)
    assert seen == ["one"] and repo.jobs["two"]["status"] == "pending"
    two.cancel()
    with pytest.raises(asyncio.CancelledError):
        await two
    assert repo.jobs["two"]["status"] == "pending"
    release.set()
    await asyncio.gather(one, three)
    await manager._run_job("two")
    assert seen == ["one", "three", "two"]
    assert all(j["status"] == "completed" for j in repo.jobs.values())


@pytest.mark.asyncio
async def test_retry_after_job_created_file_update_failed(setup):
    service, repo, admin = setup
    s = snapshot()
    identity = await service.record(s)
    original = repo.update_file
    repo.update_file = AsyncMock(side_effect=RuntimeError("update unavailable"))
    with pytest.raises(RuntimeError):
        await service.save(identity, "kb", s["sha256"], True)
    assert len(repo.jobs) == 1
    admin.tasks.submit.assert_not_called()
    repo.update_file = original
    result = await service.save(identity, "kb", s["sha256"], True)
    assert result["existing"]
    admin.tasks.submit.assert_called_once_with(result["job_id"])


@pytest.mark.asyncio
async def test_existing_uses_latest_reindex_status(setup):
    service, repo, _ = setup
    s = snapshot()
    identity = await service.record(s)
    first = await service.save(identity, "kb", s["sha256"], True)
    repo.jobs[first["job_id"]]["status"] = "completed"
    await repo.create_job("file_reindex", {}, job_id="latest")
    repo.jobs["latest"]["status"] = "failed"
    repo.files[first["file_id"]]["last_job_id"] = "latest"
    result = await service.save(identity, "kb", s["sha256"], True)
    assert result["job_id"] == "latest" and result["status"] == "failed"


@pytest.mark.asyncio
async def test_scheduler_deduplicates_pending():
    manager = TaskManager.__new__(TaskManager)
    manager.tasks = set()
    manager._run_job = AsyncMock()
    manager.submit("same")
    manager.submit("same")
    await asyncio.gather(*manager.tasks)
    manager._run_job.assert_awaited_once_with("same")


@pytest.mark.asyncio
async def test_two_managers_claim_same_pending_job_once():
    repo = Repo()
    await repo.create_job("file_ingest", {}, job_id="shared")
    release = asyncio.Event()
    entered = asyncio.Event()
    calls = []

    async def execute(job_id, payload):
        calls.append(job_id)
        entered.set()
        await release.wait()
        await repo.update_job(job_id, {"status": "completed"})

    managers = [TaskManager(Settings(_env_file=None), repo, None) for _ in range(2)]
    for manager in managers:
        manager._run_file_job = execute
    first = asyncio.create_task(managers[0]._run_job("shared"))
    await entered.wait()
    await managers[1]._run_job("shared")
    release.set()
    await first
    await managers[1]._run_job("shared")
    assert calls == ["shared"] and repo.jobs["shared"]["status"] == "completed"


@pytest.mark.asyncio
async def test_arbitrary_material_metadata_is_not_server_receipt(setup):
    service, repo, _ = setup
    repo.search_history = {"fake": {"source": "web", "metadata": {"web_snapshot_id": "fake"}}}
    with pytest.raises(AdminNotFoundError):
        await service.preview("fake")
    assert not repo.files and not repo.jobs


@pytest.mark.asyncio
async def test_wiki_does_not_wait_for_index_lock():
    repo = Repo()
    await repo.create_job("wiki_generate", {}, job_id="wiki")
    manager = TaskManager(Settings(_env_file=None), repo, None)
    manager.wiki = SimpleNamespace(generate=AsyncMock())
    async with manager.index_jobs_lock:
        await asyncio.wait_for(manager._run_job("wiki"), timeout=1)
    manager.wiki.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_claim_database_failure_cannot_overwrite_other_worker():
    repo = Repo()
    await repo.create_job("file_ingest", {}, job_id="shared")
    repo.jobs["shared"]["status"] = "running"
    repo.claim_index_job = AsyncMock(side_effect=RuntimeError("claim unavailable"))
    manager = TaskManager(Settings(_env_file=None), repo, None)
    await manager._run_job("shared")
    assert repo.jobs["shared"]["status"] == "running"
