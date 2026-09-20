import asyncio
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

import test_file_revisions as revisions
from llamaindex_retrieval.schemas import AskResponse
from llamaindex_retrieval.wiki import WikiService

index = revisions.index
setup = revisions.setup


@pytest_asyncio.fixture
async def wiki(setup, index):
    service, repo, tasks, item = setup
    async def answer(question, sources):
        raw = "原文内容[资料 1]"
        return AskResponse(answer=raw, sources=sources, retrieval={}, generation={
            "status": "completed", "answer_span": [0, len(raw)], "raw_model_answer": raw,
            "model_calls": [{"prompt": question, "raw_text": raw}]})
    pipeline = AsyncMock()
    pipeline.ask_materials.side_effect = answer
    wiki = WikiService(index.settings, repo, index, pipeline, tasks)
    tasks.wiki = wiki
    return wiki, pipeline


@pytest.mark.asyncio
async def test_generation_preserves_output_sources_and_deduplicates_automatic_jobs(setup, wiki):
    service, repo, tasks, item = setup
    wiki, model = wiki
    one = await wiki.enqueue(item["id"], automatic=True)
    two = await wiki.enqueue(item["id"], automatic=True)
    assert one["id"] == two["id"]
    await asyncio.gather(tasks._run_job(one["id"]), tasks._run_job(two["id"]))
    model.ask_materials.assert_awaited_once()
    page = await wiki.detail(one["id"])
    assert page["status"] == "draft" and page["freshness"] == "current"
    assert page["semantic_reviewed"] is False
    assert page["response"]["answer"] == page["body"] == "原文内容[资料 1]"
    assert page["response"]["sources"][0]["snippet"].endswith("Old content")
    await repo.update_job(one["id"], {"status": "pending"})
    await tasks._run_job(one["id"])
    model.ask_materials.assert_awaited_once()


@pytest.mark.asyncio
async def test_revision_auto_generates_new_version_and_marks_old_stale(setup, wiki):
    service, repo, tasks, item = setup
    wiki, model = wiki
    old = await wiki.enqueue(item["id"])
    await tasks._run_job(old["id"])
    job = await service.revise_file(item["id"], revisions.upload(), item["sha256"])
    await tasks._run_job(job["id"])
    latest_job = await repo.get_job(tasks.submit.call_args.args[0])
    assert latest_job["kind"] == "wiki_generate"
    assert (await wiki.detail(old["id"]))["freshness"] == "source_changed"
    await tasks._run_job(latest_job["id"])
    assert (await wiki.detail(latest_job["id"]))["freshness"] == "current"
    assert len(await repo.wiki_versions.find({}).to_list(length=None)) == 2
    assert (await wiki.list_pages())[0]["id"] == latest_job["id"]


@pytest.mark.asyncio
async def test_failed_generation_does_not_fail_file_and_keeps_raw_response(setup, wiki):
    service, repo, tasks, item = setup
    wiki, model = wiki
    async def invalid(question, sources):
        raw = "杜撰格式[资料 N]"
        return AskResponse(answer=raw, sources=sources, retrieval={}, generation={"status": "completed", "answer_span": [0, len(raw)]})
    model.ask_materials.side_effect = invalid
    job = await wiki.enqueue(item["id"])
    await tasks._run_job(job["id"])
    page = await wiki.detail(job["id"])
    assert page["status"] == "generation_failed"
    assert page["response"]["answer"] == "杜撰格式[资料 N]"
    assert (await repo.get_file(item["id"]))["status"] == "ready"
    assert (await repo.get_job(job["id"]))["status"] == "failed"


@pytest.mark.asyncio
async def test_source_deletion_marks_existing_wiki_without_erasing_history(setup, wiki):
    service, repo, tasks, item = setup
    wiki, _ = wiki
    job = await wiki.enqueue(item["id"])
    await tasks._run_job(job["id"])
    await service.delete_file(item["id"])
    assert (await wiki.detail(job["id"]))["freshness"] == "source_deleted"


@pytest.mark.asyncio
async def test_oversized_snapshot_stops_before_model(setup, wiki, monkeypatch):
    service, repo, tasks, item = setup
    wiki, model = wiki
    monkeypatch.setattr(wiki.settings, "wiki_max_source_characters", 1)
    job = await wiki.enqueue(item["id"])
    await tasks._run_job(job["id"])
    model.ask_materials.assert_not_awaited()
    assert (await repo.get_job(job["id"]))["status"] == "failed"
    assert await repo.get_wiki_version(job["id"]) is None


@pytest.mark.asyncio
async def test_index_rollback_marks_page_stale_even_if_file_metadata_unchanged(setup, wiki, index):
    service, repo, tasks, item = setup
    wiki, _ = wiki
    old_index = index.versions.current()
    revision = await service.revise_file(item["id"], revisions.upload(), item["sha256"])
    await tasks._run_job(revision["id"])
    wiki_job = tasks.submit.call_args.args[0]
    await tasks._run_job(wiki_job)
    index.versions.rollback(old_index, expected_current=index.versions.current())
    assert (await wiki.detail(wiki_job))["freshness"] == "source_changed"
    tasks.submit.reset_mock()
    await wiki.backfill()
    tasks.submit.assert_not_called()
