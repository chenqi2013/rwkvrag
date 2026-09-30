"""Read-only catalog/original/parsed-document/chunk binding diagnostics.

Never infer a file ID from a document ID or URI. A bound result certifies the
submitted snippet's retained byte/character binding, not index membership,
currentness, relevance, citation support, authorization or answer correctness.
"""
import asyncio
import re
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field

from .offline_replay import strict_json
from .schemas import SourceItem
from .source_revisions import original_snapshot, sha256


class SourceCatalogRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sources: list[SourceItem] = Field(min_length=1, max_length=32)
    index_version: str | None = Field(default=None, max_length=256)


class BindingError(ValueError):
    pass


def require(condition, status):
    if not condition:
        raise BindingError(status)


def identifier(value):
    return isinstance(value, str) and bool(value.strip())


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def bind_source_span(source, text):
    """Bind an indexed chunk or its explicitly recorded Reader subspan.

    source_span/chunk_text_sha256 always describe the indexed parent, not a
    Reader-selected snippet. Never relocate a quote by searching the document.
    """
    metadata = source.metadata
    span = metadata.get("source_span")
    require(isinstance(span, dict), "missing_source_span")
    start, end = span.get("start"), span.get("end")
    require(span.get("unit") == "unicode_code_points" and type(start) is int
            and type(end) is int and 0 <= start < end <= len(text), "invalid_source_span")
    chunk = text[start:end]
    snippet_hash = sha256(source.snippet.encode())
    selection_keys = {"parent_source_id", "parent_text_sha256", "span_start", "span_end",
                      "span_sha256", "offset_unit"}
    selected = any(key in metadata for key in selection_keys | {"selection_scope"})
    if selected:
        require(selection_keys <= metadata.keys(), "incomplete_selection_metadata")
        parent_hash = sha256(chunk.encode())
        require(span.get("sha256") == parent_hash
                and metadata.get("chunk_text_sha256") == parent_hash, "chunk_hash_mismatch")
        parent_id = metadata["parent_source_id"]
        require(identifier(parent_id), "invalid_selection_parent_id")
        require(metadata["parent_text_sha256"] == parent_hash, "selection_parent_hash_mismatch")
        local_start, local_end = metadata["span_start"], metadata["span_end"]
        require(metadata["offset_unit"] == "unicode_characters_in_indexed_chunk"
                and type(local_start) is int and type(local_end) is int
                and 0 <= local_start < local_end <= len(chunk), "invalid_selection_span")
        require(chunk[local_start:local_end] == source.snippet, "selection_snippet_mismatch")
        require(metadata["span_sha256"] == snippet_hash, "selection_hash_mismatch")
        require(source.id == f"{parent_id}@{local_start}:{local_end}:{snippet_hash[:12]}",
                "selection_identity_mismatch")
        start, end = start + local_start, start + local_end
    else:
        require(chunk == source.snippet, "snippet_span_mismatch")
        require(span.get("sha256") == snippet_hash
                and metadata.get("chunk_text_sha256") == snippet_hash, "chunk_hash_mismatch")
    return ("resolver_selection" if selected else "indexed_chunk"), {
        "start": start, "end": end, "unit": "unicode_code_points", "sha256": snippet_hash}


def check_binding(settings, source, file_item):
    """Resolve only server-owned revision paths; ignore submitted snapshot URIs."""
    metadata = source.metadata
    result = {
        "source_id": source.id, "document_id": source.document_id,
        "file_id": metadata.get("file_id"),
        "knowledge_base_id": metadata.get("knowledge_base_id"),
        "source_sha256": metadata.get("source_sha256"),
        "snippet_sha256": sha256(source.snippet.encode()),
        "original_verified": False, "parsed_verified": False,
        "chunk_verified": False, "original_url": None,
        "binding_scope": None, "bound_document_span": None,
    }
    try:
        file_id, kb = result["file_id"], result["knowledge_base_id"]
        require(identifier(file_id), "missing_file_id")
        require(identifier(kb), "missing_knowledge_base_id")
        require("/" not in file_id and "\\" not in file_id, "unsupported_file_id")
        require(file_item is not None, "missing_catalog_file")
        require(file_item.get("id") == file_id, "catalog_file_id_mismatch")
        require(file_item.get("knowledge_base_id") == kb, "catalog_knowledge_base_mismatch")
        require(metadata.get("document_id", source.document_id) == source.document_id,
                "document_id_mismatch")
        source_hash = result["source_sha256"]
        require(digest(source_hash), "missing_or_invalid_source_sha256")
        require(metadata.get("source_revision_id", source_hash) == source_hash,
                "source_revision_id_mismatch")
        try:
            original = original_snapshot(settings, file_item, source_hash)
        except FileNotFoundError:
            raise BindingError("missing_original_revision") from None
        except ValueError:
            raise BindingError("original_integrity_error") from None
        result["original_verified"] = True
        parsed_hash = metadata.get("parsed_snapshot_sha256")
        require(digest(parsed_hash), "missing_or_invalid_parsed_sha256")
        parsed_path = original.parent / ("parsed-" + parsed_hash + ".json")
        require(parsed_path.resolve().is_relative_to(original.parent.resolve()),
                "parsed_path_escape")
        require(parsed_path.is_file(), "missing_parsed_revision")
        encoded = parsed_path.read_bytes()
        require(sha256(encoded) == parsed_hash, "parsed_integrity_error")
        try:
            parsed = strict_json(encoded.decode("utf-8"))
        except (ValueError, UnicodeError):
            raise BindingError("invalid_parsed_revision") from None
        require(isinstance(parsed, dict), "invalid_parsed_revision")
        require(parsed.get("schema") == "rwkvrag-parsed-source-v1", "invalid_parsed_revision")
        require(all(parsed.get(key) == value for key, value in (
            ("file_id", file_id), ("knowledge_base_id", kb), ("source_sha256", source_hash))),
            "parsed_identity_mismatch")
        documents = parsed.get("documents")
        require(isinstance(documents, list) and all(isinstance(d, dict) for d in documents),
                "invalid_parsed_revision")
        matches = [d for d in documents if d.get("id_") == source.document_id]
        require(len(matches) == 1, "parsed_document_missing_or_ambiguous")
        text = matches[0].get("text")
        require(isinstance(text, str), "invalid_parsed_document_text")
        require(metadata.get("source_text_sha256") == sha256(text.encode()),
                "document_text_hash_mismatch")
        result["parsed_verified"] = True
        scope, bound_span = bind_source_span(source, text)
        result.update(status="bound", chunk_verified=True,
                      binding_scope=scope, bound_document_span=bound_span,
                      original_url=f"/v1/admin/files/{quote(file_id, safe='')}/source/{source_hash}")
    except BindingError as error:
        result["status"] = str(error)
    except OSError:
        result["status"] = "snapshot_io_error"
    return result


async def audit_catalog(settings, repo, payload):
    """No database writes, index queries or model calls; retain all input rows."""
    files = {}
    rows = []
    for source in payload.sources:
        file_id = source.metadata.get("file_id")
        if identifier(file_id) and file_id not in files:
            # Database failures propagate as failures, never as a missing record.
            files[file_id] = await repo.get_file(file_id)
        item = files.get(file_id) if identifier(file_id) else None
        rows.append(await asyncio.to_thread(check_binding, settings, source, item))
    return {"schema": "source-catalog-audit-v1", "results": rows,
            "index_version": payload.index_version, "index_membership_verified": False,
            "context_verified": False, "semantic_verified": False,
            "all_bound": all(row["status"] == "bound" for row in rows)}
