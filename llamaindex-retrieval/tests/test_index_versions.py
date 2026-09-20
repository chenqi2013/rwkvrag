"""Real OpenSearch fault tests, isolated from application indexes.

Run with RWKVRAG_TEST_OPENSEARCH_URL; every test owns a random prefix.
"""
import asyncio
import os
import threading
from uuid import uuid4

from llama_index.core import Document
from llama_index.core.schema import TextNode
from opensearchpy import OpenSearch
import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.ingest import ingest_documents, replace_uploaded_documents
from llamaindex_retrieval.lexical_index import LexicalIndex
from llamaindex_retrieval.tasks import finish_thread_on_cancel


@pytest.fixture
def index(tmp_path):
    url = os.environ.get("RWKVRAG_TEST_OPENSEARCH_URL")
    if not url:
        pytest.skip("set RWKVRAG_TEST_OPENSEARCH_URL for isolated OpenSearch integration tests")
    prefix = "rwkvrag-test-versions-" + uuid4().hex
    client = OpenSearch(url)
    settings = Settings(_env_file=None, opensearch_url=url, opensearch_index=prefix,
                        opensearch_replicas=0, rag_pipeline="rwkv", upload_dir=tmp_path / "uploads")
    # Exercise adoption of a pre-existing concrete index, with no deletion.
    prototype = object.__new__(LexicalIndex)
    prototype.settings = settings
    client.indices.create(index=prefix, body=prototype.index_definition())
    client.index(index=prefix, id="old", body={"node_id": "old", "text": "old",
                                              "file_id": "file-a"}, refresh=True)
    try:
        idx = LexicalIndex(settings, client)
        yield idx
    finally:
        for name in client.indices.get(index=prefix + "*"):
            client.indices.delete(index=name)
        client.close()


def texts(index, name=None):
    index.client.indices.refresh(index=name or index.index_name)
    return {h["_source"]["text"] for h in index.client.search(
        index=name or index.index_name, body={"query": {"match_all": {}}, "size": 100})["hits"]["hits"]}


def test_adoption_and_publication_retain_old_generation(index):
    old = index.versions.current()
    with index.versions.replacement() as staged:
        staged.upsert_nodes([TextNode(id_="new", text="new")])
        assert texts(index) == {"old"}
    assert texts(index) == {"new"}
    assert texts(index, old) == {"old"}


def test_mid_batch_failure_never_replaces_active_knowledge(index):
    def documents():
        yield Document(text="first new document", id_="new")
        raise ValueError("broken input")
    with pytest.raises(ValueError, match="broken input"):
        ingest_documents(index.settings, documents(), 1, True, lexical_index=index)
    assert texts(index) == {"old"}
    assert not index.client.indices.exists(index=index.versions.lock_name)


def test_empty_rebuild_rejected(index):
    with pytest.raises(ValueError):
        ingest_documents(index.settings, iter([]), 1, True, lexical_index=index)
    assert texts(index) == {"old"}


def test_file_replacement_keeps_other_file_and_removes_obsolete_chunks(index):
    index.upsert_nodes([TextNode(id_="other", text="other file", metadata={"file_id": "file-b"})])
    old = index.versions.current()
    stats = replace_uploaded_documents(index.settings, [Document(
        text="replacement text", id_="replacement", metadata={"file_id": "file-a"})],
        "file-a", lexical_index=index)
    assert stats["nodes"] == 1
    assert texts(index) == {"other file", "replacement text"}
    assert texts(index, old) == {"old", "other file"}


def test_incomplete_copy_rejected(index, monkeypatch):
    original = index.client.reindex
    def incomplete(**kwargs):
        result = original(**kwargs)
        result["failures"] = [{"reason": "simulated shard failure"}]
        return result
    monkeypatch.setattr(index.client, "reindex", incomplete)
    with pytest.raises(RuntimeError, match="复制"):
        replace_uploaded_documents(index.settings, [], "file-a", lexical_index=index)
    assert texts(index) == {"old"}


@pytest.mark.parametrize("committed", [False, True])
def test_ambiguous_publication_never_deletes_possibly_active_index(index, monkeypatch, committed):
    original = index.client.indices.update_aliases
    old = index.versions.current()
    def timeout(**kwargs):
        if committed:
            original(**kwargs)
        raise TimeoutError("lost acknowledgement")
    monkeypatch.setattr(index.client.indices, "update_aliases", timeout)
    with pytest.raises(TimeoutError):
        with index.versions.replacement() as staged:
            staged.upsert_nodes([TextNode(id_="new", text="new")])
    assert texts(index) == ({"new"} if committed else {"old"})
    assert texts(index, old) == {"old"}


def test_second_instance_cannot_write_or_rebuild_during_snapshot(index):
    other = LexicalIndex(index.settings, index.client)
    with index.versions.replacement() as staged:
        with pytest.raises(RuntimeError, match="写锁"):
            other.upsert_nodes([TextNode(id_="raced", text="raced")])
        with pytest.raises(RuntimeError, match="写锁"):
            other.delete_by_field("file_id", "file-a")
        with pytest.raises(RuntimeError, match="写锁"):
            with other.versions.replacement():
                pytest.fail("second writer admitted")
        staged.upsert_nodes([TextNode(id_="new", text="new")])
    assert texts(other) == {"new"}


def test_existing_lock_is_not_expired_or_stolen(index):
    index.client.indices.create(index=index.versions.lock_name)
    with pytest.raises(RuntimeError, match="写锁"):
        index.delete_by_field("file_id", "file-a")
    assert texts(index) == {"old"}
    assert index.client.indices.exists(index=index.versions.lock_name)


@pytest.mark.asyncio
async def test_cancel_waits_for_writer_before_releasing_job():
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    def write():
        started.set()
        release.wait(timeout=5)
        finished.set()
    task = asyncio.create_task(finish_thread_on_cancel(write))
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_file_job_uses_staging_without_deleting_live_chunks(index, tmp_path, monkeypatch, fail):
    from unittest.mock import AsyncMock
    from llamaindex_retrieval import tasks
    path = tmp_path / "document.md"
    path.write_text("# New document\n\nReplacement body.")
    payload = {"file_id": "file-a", "path": str(path)}
    repository = AsyncMock()
    repository.get_job.return_value = {"kind": "file_reindex", "payload": payload}
    repository.get_file.return_value = {"knowledge_base_id": "default"}
    if fail:
        original = tasks.replace_uploaded_documents
        def broken(settings, documents, file_id, batch_size, callback, lexical_index, revision):
            def fail_progress(*args):
                assert texts(lexical_index) == {"old"}
                raise ValueError("fault after new chunks written")
            return original(settings, documents, file_id, batch_size, fail_progress, lexical_index, revision)
        monkeypatch.setattr(tasks, "replace_uploaded_documents", broken)
    manager = tasks.TaskManager(index.settings, repository, index)
    await manager._run_job("job-1")
    updates = [call.args[1] for call in repository.update_job.call_args_list]
    assert updates[-1]["status"] == ("failed" if fail else "completed")
    assert any(row.get("stage") == "staging" for row in updates)
    assert (texts(index) == {"old"}) is fail
    if fail:
        assert all("last_indexed_revision" not in call.args[1] for call in repository.update_file.call_args_list)
    assert texts(index, index.settings.opensearch_index) == {"old"}


def publish(index, text="new"):
    with index.versions.replacement() as staged:
        staged.upsert_nodes([TextNode(id_=text, text=text)])
    return index.versions.current()


def test_history_and_round_trip_rollback(index):
    original = index.versions.current()
    new = publish(index)
    history = index.versions.history()
    rows = {r["index"]: r for r in history["items"]}
    assert rows[original]["rollback_eligible"] is True
    assert rows[new]["active"] is True
    result = index.versions.rollback(original, expected_current=new)
    assert result["original_files_restored"] is result["database_records_restored"] is False
    assert texts(index) == {"old"}
    index.versions.rollback(new, expected_current=original)
    assert texts(index) == {"new"}
    assert texts(index, original) == {"old"}


def test_stale_rollback_does_not_replace_newer_version(index):
    original = index.versions.current()
    one = publish(index, "one")
    two = publish(index, "two")
    with pytest.raises(RuntimeError, match="已改变"):
        index.versions.rollback(original, expected_current=one)
    assert index.versions.current() == two
    assert texts(index) == {"two"}


def test_failed_staging_is_visible_but_not_rollback_eligible(index):
    with pytest.raises(ValueError):
        with index.versions.replacement() as staged:
            staged.upsert_nodes([TextNode(id_="partial", text="partial")])
            target = staged.index_name
            raise ValueError("input failed")
    row = next(r for r in index.versions.history()["items"] if r["index"] == target)
    assert row["status"] == "unverified" and row["rollback_eligible"] is False
    with pytest.raises(ValueError, match="未验证"):
        index.versions.rollback(target, expected_current=index.versions.current())
    assert texts(index) == {"old"}


def test_foreign_and_wildcard_rollback_targets_rejected_before_lock(index, monkeypatch):
    def fail(**kwargs):
        pytest.fail("invalid target must not attempt a mutation")
    monkeypatch.setattr(index.client.indices, "create", fail)
    for target in ("foreign-index", index.settings.opensearch_index + "*", index.index_name):
        with pytest.raises(ValueError, match="不属于"):
            index.versions.rollback(target, expected_current=index.versions.current())


@pytest.mark.parametrize("committed", [False, True])
def test_rollback_timeout_preserves_both_versions(index, monkeypatch, committed):
    old = index.versions.current()
    new = publish(index)
    update = index.client.indices.update_aliases
    def timeout(**kwargs):
        if committed:
            update(**kwargs)
        raise TimeoutError("uncertain response")
    monkeypatch.setattr(index.client.indices, "update_aliases", timeout)
    with pytest.raises(TimeoutError):
        index.versions.rollback(old, expected_current=new)
    assert texts(index) == ({"old"} if committed else {"new"})
    assert texts(index, old) == {"old"}
    assert texts(index, new) == {"new"}


def test_read_only_maintenance_constructor_does_not_create_or_map(index, monkeypatch):
    def fail(**kwargs):
        pytest.fail("history must not mutate indexes")
    monkeypatch.setattr(index.client.indices, "create", fail)
    monkeypatch.setattr(index.client.indices, "put_mapping", fail)
    other = LexicalIndex(index.settings, index.client, initialize=False)
    assert other.versions.history()["active"] == index.settings.opensearch_index


def test_cli_rollback_requires_explicit_current_version():
    from llamaindex_retrieval.cli import parser
    with pytest.raises(SystemExit):
        parser().parse_args(["rollback-index", "--target", "old"])
    args = parser().parse_args(["rollback-index", "--target", "old", "--expected-current", "new"])
    assert args.expected_current == "new" and args.target == "old"


def test_rollback_rejects_partial_shard_count(index, monkeypatch):
    old = index.versions.current()
    current = publish(index)
    monkeypatch.setattr(index.client, "count", lambda **kwargs: {"count": 1, "_shards": {"failed": 1}})
    with pytest.raises(RuntimeError, match="分片"):
        index.versions.rollback(old, expected_current=current)
    assert index.versions.current() == current


def test_maintenance_cli_lists_and_rolls_back(index, monkeypatch, capsys):
    import json
    import sys
    from llamaindex_retrieval import cli
    old = index.versions.current()
    current = publish(index)
    monkeypatch.setattr(cli, "get_settings", lambda: index.settings)
    monkeypatch.setattr(sys, "argv", ["rwkvrag-retrieval", "index-versions"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["active"] == current
    monkeypatch.setattr(sys, "argv", ["rwkvrag-retrieval", "rollback-index", "--target", old,
                                      "--expected-current", current])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["active"] == old and result["scope"] == "entire_search_index_only"
    assert texts(index) == {"old"}


@pytest.mark.asyncio
async def test_uploaded_revision_is_bound_to_nodes_index_and_successful_file_record(index, tmp_path):
    from unittest.mock import AsyncMock
    from llamaindex_retrieval.tasks import TaskManager
    import hashlib
    path = tmp_path / "revision.md"
    path.write_text("# Source revision\nVersioned content.")
    repo = AsyncMock()
    repo.get_job.return_value = {"kind": "file_reindex", "payload": {"file_id": "file-a", "path": str(path)}}
    repo.get_file.return_value = {"knowledge_base_id": "default", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    await TaskManager(index.settings, repo, index)._run_job("revision-job")
    ready = next(call.args[1] for call in repo.update_file.call_args_list if call.args[1].get("status") == "ready")
    version = index.versions.current()
    assert ready["last_indexed_index_version"] == version
    revision = ready["last_indexed_revision"]
    assert index.versions._metadata(version)["rwkvrag_build"]["source_revision"] == revision
    assert revision["ingest_job_id"] == "revision-job"
    sources = index.versions.source_revisions(version, "file-a")
    assert sources["untracked_chunks"] == 0 and sources["chunks"] > 0
    assert sources["revisions"][0]["source_sha256"] == repo.get_file.return_value["sha256"]
    # Rebuilding another file retains this file's source-version binding.
    publish_before = version
    replace_uploaded_documents(index.settings, [Document(text="another", metadata={"file_id": "file-b"})],
                               "file-b", lexical_index=index)
    assert index.versions.source_revisions(index.versions.current(), "file-a")["revisions"] == sources["revisions"]
    index.versions.rollback(publish_before, expected_current=index.versions.current())
    assert index.versions.source_revisions(publish_before, "file-a")["revisions"] == sources["revisions"]
