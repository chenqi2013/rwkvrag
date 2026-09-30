"""Temporary original files and fake catalog only; no running data-store repair."""
import copy
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
import pytest
from fastapi import FastAPI

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.dependencies import repository
from llamaindex_retrieval.routers import admin
from llamaindex_retrieval.schemas import SourceItem
from llamaindex_retrieval.source_catalog import SourceCatalogRequest, audit_catalog, check_binding
from llamaindex_retrieval.source_revisions import original_snapshot, prepare_revision, sha256
from llamaindex_retrieval.verbatim_chunking import verbatim_nodes


@pytest.fixture
def fixture(tmp_path):
    settings = Settings(_env_file=None, upload_dir=tmp_path / "uploads")
    path = tmp_path / "原件.md"
    path.write_bytes("# 原文😀\r\n\r\n乙设备未记录温度；容量0 L。\r\n".encode())
    documents, revision = prepare_revision(settings, path, "file-a", "kb-a")
    node = verbatim_nodes(documents[0])[-1]
    source = SourceItem(id=node.node_id, document_id=documents[0].id_, source="upload",
                        title="原件", score=1, snippet=node.text, metadata=node.metadata)
    item = {"id": "file-a", "knowledge_base_id": "kb-a", "sha256": revision["source_sha256"]}
    return settings, source, item, path


def test_complete_retained_original_parsed_document_and_unicode_span_binding(fixture):
    settings, source, item, path = fixture
    before = source.model_dump()
    row = check_binding(settings, source, item)
    assert row["status"] == "bound" and row["chunk_verified"]
    assert row["original_verified"] and row["parsed_verified"]
    assert row["original_url"].endswith("/source/" + sha256(path.read_bytes()))
    assert source.model_dump() == before
    # Supplied paths/URIs are never read; catalog + namespace + hashes determine paths.
    source.uri = "file:///must-not-read"
    source.metadata["source_snapshot_uri"] = "file:///must-not-read"
    source.metadata["parsed_snapshot_uri"] = "https://must-not-fetch.invalid"
    assert check_binding(settings, source, item)["status"] == "bound"


@pytest.mark.parametrize("change,status", [
    ({"file_id": None}, "missing_file_id"),
    ({"file_id": []}, "missing_file_id"),
    ({"knowledge_base_id": None}, "missing_knowledge_base_id"),
    ({"knowledge_base_id": "kb-other"}, "catalog_knowledge_base_mismatch"),
    ({"document_id": "other"}, "document_id_mismatch"),
    ({"source_sha256": "../../outside"}, "missing_or_invalid_source_sha256"),
    ({"source_sha256": None}, "missing_or_invalid_source_sha256"),
    ({"source_revision_id": "0" * 64}, "source_revision_id_mismatch"),
    ({"parsed_snapshot_sha256": None}, "missing_or_invalid_parsed_sha256"),
    ({"source_text_sha256": "0" * 64}, "document_text_hash_mismatch"),
    ({"source_span": None}, "missing_source_span"),
    ({"chunk_text_sha256": "0" * 64}, "chunk_hash_mismatch"),
])
def test_does_not_guess_missing_or_conflicting_identities(fixture, change, status):
    settings, source, item, _ = fixture
    source.metadata.update(change)
    row = check_binding(settings, source, item)
    assert row["status"] == status and row["original_url"] is None
    assert row["chunk_verified"] is False


def test_missing_catalog_file_never_falls_back_to_document_id_or_uri(fixture):
    settings, source, item, _ = fixture
    assert check_binding(settings, source, None)["status"] == "missing_catalog_file"
    item["id"] = "different-file"
    assert check_binding(settings, source, item)["status"] == "catalog_file_id_mismatch"


@pytest.mark.parametrize("change,status", [
    ({"unit": "utf8_bytes"}, "invalid_source_span"),
    ({"start": True}, "invalid_source_span"),
    ({"end": 10**9}, "invalid_source_span"),
    ({"sha256": "0" * 64}, "chunk_hash_mismatch"),
])
def test_span_units_and_hashes_are_not_coerced(fixture, change, status):
    settings, source, item, _ = fixture
    source.metadata["source_span"].update(change)
    assert check_binding(settings, source, item)["status"] == status


def test_changed_snippet_and_missing_document_are_not_relocalized(fixture):
    settings, source, item, _ = fixture
    source.snippet += " invented"
    assert check_binding(settings, source, item)["status"] == "snippet_span_mismatch"
    source.document_id = "other"
    source.metadata["document_id"] = "other"
    assert check_binding(settings, source, item)["status"] == "parsed_document_missing_or_ambiguous"


@pytest.mark.parametrize("part", ["source", "parsed"])
def test_missing_or_corrupt_revision_stays_an_error(fixture, part):
    settings, source, item, _ = fixture
    uri = source.metadata[part + "_snapshot_uri"]
    path = Path(unquote(urlparse(uri).path))
    original = path.read_bytes()
    path.write_bytes(b"corrupt")
    expected = "original_integrity_error" if part == "source" else "parsed_integrity_error"
    assert check_binding(settings, source, item)["status"] == expected
    path.unlink()
    expected = "original_integrity_error" if part == "source" else "missing_parsed_revision"
    assert check_binding(settings, source, item)["status"] == expected
    path.write_bytes(original)
    assert check_binding(settings, source, item)["status"] == "bound"


def test_old_original_remains_bound_after_file_revision_and_current_file_deletion(fixture):
    settings, source, item, path = fixture
    path.write_text("new revision")
    _, new = prepare_revision(settings, path, item["id"], item["knowledge_base_id"])
    item["sha256"] = new["source_sha256"]
    path.unlink()
    assert check_binding(settings, source, item)["status"] == "bound"
    assert source.metadata["source_sha256"] != item["sha256"]


def test_namespace_directory_symlink_cannot_escape_revision_root(fixture, tmp_path):
    settings, source, item, _ = fixture
    original = original_snapshot(settings, item, source.metadata["source_sha256"])
    directory = original.parent
    moved = tmp_path / "outside"
    directory.rename(moved)
    directory.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="超出快照目录"):
        original_snapshot(settings, item, source.metadata["source_sha256"])
    assert check_binding(settings, source, item)["status"] == "original_integrity_error"


@pytest.mark.asyncio
async def test_read_only_route_and_original_file_bytes(fixture, monkeypatch):
    settings, source, item, path = fixture
    class Repo:
        def __init__(self):
            self.calls = []

        async def get_file(self, identity):
            self.calls.append(identity)
            return copy.deepcopy(item) if identity == item["id"] else None

    repo = Repo()
    from llamaindex_retrieval.admin_service import AdminNotFoundError, AdminValidationError
    from llamaindex_retrieval.api import not_found_handler, validation_handler
    app = FastAPI()
    app.add_exception_handler(AdminNotFoundError, not_found_handler)
    app.add_exception_handler(AdminValidationError, validation_handler)
    app.include_router(admin.router)
    app.dependency_overrides[repository] = lambda: repo
    monkeypatch.setattr(admin, "get_settings", lambda: settings)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/admin/source-catalog/audit", json={
            "sources": [source.model_dump(), source.model_dump()], "index_version": "saved-index"})
        assert response.status_code == 200
        audit = response.json()
        assert audit["all_bound"] and repo.calls == ["file-a"]
        assert not audit["semantic_verified"] and not audit["index_membership_verified"]
        assert not audit["context_verified"] and audit["index_version"] == "saved-index"
        download = await client.get(audit["results"][0]["original_url"])
        assert download.status_code == 200 and download.content == path.read_bytes()
        assert download.headers["x-content-type-options"] == "nosniff"
        source_hash = source.metadata["source_sha256"]
        assert (await client.get(f"/v1/admin/files/unknown/source/{source_hash}")).status_code == 404
        assert (await client.get("/v1/admin/files/file-a/source/" + "0" * 64)).status_code == 404
        assert (await client.get("/v1/admin/files/file-a/source/invalid-hash")).status_code == 422
        missing = source.model_copy(deep=True)
        missing.metadata["file_id"] = "unknown"
        diagnostic = await client.post("/v1/admin/source-catalog/audit", json={"sources": [missing.model_dump()]})
        assert diagnostic.json()["results"][0]["status"] == "missing_catalog_file"
        assert diagnostic.json()["results"][0]["original_url"] is None
        for payload in ({"sources": []}, {"sources": [source.model_dump()] * 33},
                        {"sources": [source.model_dump()], "repair": True}):
            assert (await client.post("/v1/admin/source-catalog/audit", json=payload)).status_code == 422


@pytest.mark.parametrize("key,value", [("file_id", "wrong-file"), ("knowledge_base_id", "other-kb"),
                                      ("source_sha256", "0" * 64)])
def test_parsed_snapshot_cannot_claim_another_original_or_catalog_identity(fixture, key, value):
    settings, source, item, _ = fixture
    path = Path(unquote(urlparse(source.metadata["parsed_snapshot_uri"]).path))
    parsed = json.loads(path.read_bytes())
    parsed[key] = value
    encoded = json.dumps(parsed, ensure_ascii=False).encode()
    changed = sha256(encoded)
    (path.parent / ("parsed-" + changed + ".json")).write_bytes(encoded)
    source.metadata["parsed_snapshot_sha256"] = changed
    assert check_binding(settings, source, item)["status"] == "parsed_identity_mismatch"


def test_parsed_snapshot_symlink_is_not_followed_outside_its_revision(fixture, tmp_path):
    settings, source, item, _ = fixture
    path = Path(unquote(urlparse(source.metadata["parsed_snapshot_uri"]).path))
    outside = tmp_path / "outside-parsed.json"
    path.rename(outside)
    path.symlink_to(outside)
    assert check_binding(settings, source, item)["status"] == "parsed_path_escape"


@pytest.mark.asyncio
async def test_database_failure_is_not_reported_as_missing_catalog(fixture):
    settings, source, _, _ = fixture
    class Repo:
        async def get_file(self, identity):
            raise RuntimeError("database unavailable")
    with pytest.raises(RuntimeError, match="database unavailable"):
        await audit_catalog(settings, Repo(), SourceCatalogRequest(sources=[source]))
