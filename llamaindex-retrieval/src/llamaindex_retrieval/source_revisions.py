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


def prepare_revision(settings, path, file_id, knowledge_base_id, expected_sha256=None, web_provenance=None):
    # Namespace with hashes, so identifiers cannot escape the revision root.
    content = path.read_bytes()
    source_hash = sha256(content)
    if expected_sha256 and source_hash != expected_sha256:
        raise ValueError("原文件与上传记录的 SHA256 不符；请通过正式修订流程更新，不能直接改写原文件")
    directory = (settings.upload_dir / ".revisions" / sha256(knowledge_base_id.encode())
                 / sha256(file_id.encode()) / source_hash)
    snapshot = directory / path.name
    write_once(snapshot, content)
    if web_provenance is not None:
        from .web_sources import validate_provenance
        validate_provenance(web_provenance, source_hash)
        identity = parsers.parsed_document_id(file_id, "web-snapshot")
        metadata = parsers.document_metadata(file_id=file_id, knowledge_base_id=knowledge_base_id,
            filename=path.name, title=web_provenance["title"], uri=web_provenance["url"],
            kind="web-snapshot", document_id=identity)
        # Preserve provider bytes as text; never apply the special QA-Markdown parser.
        documents = [parsers.llama_document(text=content.decode("utf-8"), metadata=metadata)]
        for document in documents:
            document.metadata.update({"uri": web_provenance["url"],
                "title": web_provenance["title"], "web_provenance": dict(web_provenance),
                "source_origin": "saved-web", "web_snapshot_id": web_provenance["snapshot_id"]})
    else:
        documents = parsers.parse_uploaded_file(snapshot, file_id, knowledge_base_id, settings=settings)
    if sha256(snapshot.read_bytes()) != source_hash:
        raise ValueError("原文快照在解析期间发生变化")
    parsed = {
        "schema": "rwkvrag-parsed-source-v1", "file_id": file_id,
        "knowledge_base_id": knowledge_base_id, "source_sha256": source_hash,
        "source_snapshot_uri": snapshot.resolve().as_uri(),
        "parser_sha256": sha256(Path(parsers.__file__).read_bytes()),
        "ocr_code_sha256": {name: sha256(Path(parsers.__file__).with_name(name).read_bytes())
                            for name in ("ocr.py", "ocr_worker.py")},
        "documents": [document.model_dump(mode="json") for document in documents],
        **({"web_provenance": dict(web_provenance)} if web_provenance else {}),
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
    coverage = documents[0].metadata.get("page_coverage") if documents else None
    if coverage is not None:
        revision["extraction_summary"] = {"pages": len(coverage),
            "ocr_pages": [p["page"] for p in coverage if "ocr" in p["method"]],
            "blank_pages": [p["page"] for p in coverage if p["status"] == "blank"],
            "status": "complete"}
    for document in documents:
        fields = {**revision, "source_text_sha256": sha256(document.text.encode())}
        document.metadata.update(fields)
        document.excluded_llm_metadata_keys = list(dict.fromkeys([
            *document.excluded_llm_metadata_keys, *fields]))
        document.excluded_embed_metadata_keys = list(dict.fromkeys([
            *document.excluded_embed_metadata_keys, *fields]))
    return documents, revision


def original_snapshot(settings, file_item, source_sha256):
    """Resolve a retained revision by server-side file identity, never a client path."""
    import re
    if not re.fullmatch(r'[0-9a-f]{64}', source_sha256):
        raise ValueError('原文版本 SHA256 无效')
    directory = (settings.upload_dir / '.revisions'
                 / sha256(file_item['knowledge_base_id'].encode())
                 / sha256(file_item['id'].encode()) / source_sha256)
    if not directory.resolve().is_relative_to(settings.upload_dir.resolve() / '.revisions'):
        raise ValueError('原文版本路径超出快照目录')
    if not directory.is_dir():
        raise FileNotFoundError('原文版本不存在，请先重建索引以保存原件')
    candidates = [p for p in directory.iterdir()
                  if p.suffix.lower() in parsers.SUPPORTED_EXTENSIONS and p.is_file()]
    valid = [p for p in candidates if p.resolve().is_relative_to(directory.resolve())
             and sha256(p.read_bytes()) == source_sha256]
    if len(valid) != 1:
        raise ValueError('原文版本完整性校验失败')
    return valid[0]
