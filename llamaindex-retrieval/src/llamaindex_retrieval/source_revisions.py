"""Content-addressed source and parsed snapshots; never overwrite a revision."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from . import parsers


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ValueError("历史快照完整性校验失败，拒绝覆盖")
        finally:
            temporary.unlink(missing_ok=True)


def prepare_revision(settings, path, file_id, knowledge_base_id, expected_sha256=None):
    # Namespace with hashes, so identifiers cannot escape the revision root.
    content = path.read_bytes()
    source_hash = sha256(content)
    if expected_sha256 and source_hash != expected_sha256:
        raise ValueError("原文件与上传记录的 SHA256 不符；请通过正式修订流程更新，不能直接改写原文件")
    directory = (settings.upload_dir / ".revisions" / sha256(knowledge_base_id.encode())
                 / sha256(file_id.encode()) / source_hash)
    snapshot = directory / path.name
    write_once(snapshot, content)
    documents = parsers.parse_uploaded_file(snapshot, file_id, knowledge_base_id)
    if sha256(snapshot.read_bytes()) != source_hash:
        raise ValueError("原文快照在解析期间发生变化")
    parsed = {
        "schema": "rwkvrag-parsed-source-v1", "file_id": file_id,
        "knowledge_base_id": knowledge_base_id, "source_sha256": source_hash,
        "source_snapshot_uri": snapshot.resolve().as_uri(),
        "parser_sha256": sha256(Path(parsers.__file__).read_bytes()),
        "documents": [document.model_dump(mode="json") for document in documents],
    }
    encoded = json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    parsed_hash = sha256(encoded)
    parsed_path = directory / ("parsed-" + parsed_hash + ".json")
    write_once(parsed_path, encoded)
    revision = {"source_revision_id": source_hash, "source_sha256": source_hash,
                "source_snapshot_uri": snapshot.resolve().as_uri(),
                "parsed_snapshot_sha256": parsed_hash,
                "parsed_snapshot_uri": parsed_path.resolve().as_uri(),
                "file_id": file_id, "knowledge_base_id": knowledge_base_id}
    for document in documents:
        fields = {**revision, "source_text_sha256": sha256(document.text.encode())}
        document.metadata.update(fields)
        document.excluded_llm_metadata_keys = list(dict.fromkeys([
            *document.excluded_llm_metadata_keys, *fields]))
        document.excluded_embed_metadata_keys = list(dict.fromkeys([
            *document.excluded_embed_metadata_keys, *fields]))
    return documents, revision
