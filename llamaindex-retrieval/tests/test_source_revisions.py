import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.source_revisions import prepare_revision


def local(uri):
    return Path(unquote(urlparse(uri).path))


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.md"
    path.write_bytes(" \r\n# 原文\r\n\r\n精确内容😀。\r\n ".encode())
    return Settings(_env_file=None, upload_dir=tmp_path / "uploads"), path


def test_snapshot_preserves_source_bytes_and_exact_parsed_text(source):
    settings, path = source
    raw = path.read_bytes()
    documents, revision = prepare_revision(settings, path, "file", "kb", hashlib.sha256(raw).hexdigest())
    assert local(revision["source_snapshot_uri"]).read_bytes() == raw
    parsed_bytes = local(revision["parsed_snapshot_uri"]).read_bytes()
    assert hashlib.sha256(parsed_bytes).hexdigest() == revision["parsed_snapshot_sha256"]
    parsed = json.loads(parsed_bytes)
    assert parsed["documents"][0]["text"] == documents[0].text
    assert documents[0].metadata["source_text_sha256"] == hashlib.sha256(documents[0].text.encode()).hexdigest()
    assert "source_snapshot_uri" in documents[0].excluded_llm_metadata_keys
    assert prepare_revision(settings, path, "file", "kb")[1] == revision


def test_changed_original_rejected_against_upload_receipt(source):
    settings, path = source
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text("changed")
    with pytest.raises(ValueError, match="上传记录"):
        prepare_revision(settings, path, "file", "kb", expected)


def test_distinct_content_retains_both_snapshots(source):
    settings, path = source
    _, before = prepare_revision(settings, path, "file", "kb")
    old = local(before["source_snapshot_uri"]).read_bytes()
    path.write_text("# new\nnew text")
    _, after = prepare_revision(settings, path, "file", "kb")
    assert before["source_revision_id"] != after["source_revision_id"]
    assert local(before["source_snapshot_uri"]).read_bytes() == old
    path.unlink()
    assert local(after["source_snapshot_uri"]).is_file()


def test_corrupted_snapshot_never_overwritten(source):
    settings, path = source
    _, revision = prepare_revision(settings, path, "file", "kb")
    snapshot = local(revision["source_snapshot_uri"])
    snapshot.write_text("tampered")
    with pytest.raises(ValueError, match="拒绝覆盖"):
        prepare_revision(settings, path, "file", "kb")
    assert snapshot.read_text() == "tampered"


def test_namespace_identifiers_cannot_escape_snapshot_root(source):
    settings, path = source
    _, revision = prepare_revision(settings, path, "../../../outside", "../../kb")
    assert local(revision["source_snapshot_uri"]).is_relative_to(settings.upload_dir / ".revisions")
