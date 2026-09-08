"""Conservative Markdown boundaries over unmodified source text.

The character window is soft: Markdown tables, lists, fenced code, headings,
and QA documents are never cut to meet it. This module does not tokenize or
summarize an oversized atom; the caller must enforce its model's hard budget.
Only ATX headings, pipe tables, Markdown lists, and fenced code are recognized.
This is a boundary recognizer, not a CommonMark renderer or semantic parser.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass

from llama_index.core import Document
from llama_index.core.schema import NodeRelationship, TextNode

_VERSION = "verbatim-v1"
_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+|$)")
_LIST = re.compile(r"^[ \t]*(?:[-+*]|[0-9]{1,9}[.)])[ \t]+")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_TABLE_SEPARATOR = re.compile(r":?-{3,}:?$")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Heading:
    level: int
    start: int
    end: int


@dataclass(frozen=True)
class _Block:
    start: int
    end: int
    kind: str
    headings: tuple[_Heading, ...]


def _line_body(line: str) -> str:
    # Used for boundary recognition only. Nodes always slice the original text.
    return line.rstrip("\r\n")


def _separator(line: str) -> bool:
    body = _line_body(line).strip()
    if "|" not in body:
        return False
    cells = body.removeprefix("|").removesuffix("|").split("|")
    return bool(cells) and all(_TABLE_SEPARATOR.fullmatch(cell.strip()) for cell in cells)


def _table_start(lines: list[str], index: int) -> bool:
    body = _line_body(lines[index])
    # Leading-pipe runs are conservatively atomic, including malformed tables.
    return body.lstrip(" \t").startswith("|") or (
        "|" in body and index + 1 < len(lines) and _separator(lines[index + 1])
    )


def _list_end(lines: list[str], index: int) -> int:
    """Keep nested/lazy continuations and blank-separated items in one atom."""
    end = index + 1
    while end < len(lines):
        body = _line_body(lines[end])
        if not body.strip() or _LIST.match(body) or body.startswith((" ", "\t")):
            end += 1
            continue
        if (
            not _line_body(lines[end - 1]).strip()
            or _HEADING.match(body)
            or _FENCE.match(body)
            or _table_start(lines, end)
        ):
            break
        # CommonMark permits an unindented lazy paragraph continuation.
        end += 1
    return end


def _blocks(text: str) -> list[_Block]:
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    blocks: list[_Block] = []
    headings: list[_Heading] = []
    index = 0
    prose_start = 0

    def flush_prose(end: int) -> None:
        if prose_start < end:
            blocks.append(_Block(prose_start, end, "prose", tuple(headings)))

    while index < len(lines):
        body = _line_body(lines[index])
        heading = _HEADING.match(body)
        fence = _FENCE.match(body)
        kind: str | None = None
        end = index + 1
        if fence:
            kind = "code"
            marker = fence.group(1)
            close = re.compile(r"^ {0,3}" + re.escape(marker[0]) + rf"{{{len(marker)},}}[ \t]*$")
            while end < len(lines):
                is_close = close.fullmatch(_line_body(lines[end])) is not None
                end += 1
                if is_close:
                    break
        elif heading:
            kind = "heading"
        elif _table_start(lines, index):
            kind = "table"
            while end < len(lines) and "|" in _line_body(lines[end]):
                # A heading/fence after a table starts a new structural block.
                if _HEADING.match(_line_body(lines[end])) or _FENCE.match(lines[end]):
                    break
                end += 1
        elif _LIST.match(body):
            kind = "list"
            end = _list_end(lines, index)

        if kind is None:
            index += 1
            continue
        flush_prose(offsets[index])
        if heading and kind == "heading":
            level = len(heading.group(1))
            headings = [item for item in headings if item.level < level]
            blocks.append(_Block(offsets[index], offsets[end], kind, tuple(headings)))
            headings.append(_Heading(level, offsets[index], offsets[end]))
        else:
            blocks.append(_Block(offsets[index], offsets[end], kind, tuple(headings)))
        index = end
        prose_start = offsets[index]
    flush_prose(len(text))
    return blocks


def verbatim_nodes(
    document: Document,
    chunk_characters: int = 2400,
    overlap_characters: int = 180,
) -> list[TextNode]:
    """Return exact source slices with original-document Unicode offsets.

    All original characters, including headings and whitespace, belong to at
    least one node. Overlap is applied within prose blocks only. Parent heading
    lines are exact ``context_spans`` metadata, never prefixed to node text.
    Recognized table headers plus their delimiter row are also context spans,
    including when inside the node, for callers selecting individual table rows.
    Empty documents return no nodes; whitespace-only documents are retained.
    """
    if type(chunk_characters) is not int or chunk_characters < 2:
        raise ValueError("chunk_characters must be an integer of at least 2")
    if (
        type(overlap_characters) is not int
        or not 0 < overlap_characters < chunk_characters
    ):
        raise ValueError("overlap_characters must be positive and below chunk_characters")
    text = document.text
    source_sha = _sha(text)
    declared_sha = document.metadata.get("source_text_sha256")
    if declared_sha is not None and declared_sha != source_sha:
        raise ValueError("source_text_sha256 does not match the original document text")
    if not text:
        return []

    blocks = (
        [_Block(0, len(text), "qa", ())]
        if document.metadata.get("kind") == "qa-markdown"
        else _blocks(text)
    )
    nodes: list[TextNode] = []
    for block in blocks:
        start = block.start
        while start < block.end:
            end = (
                min(start + chunk_characters, block.end)
                if block.kind == "prose"
                else block.end
            )
            snippet = text[start:end]
            chunk_sha = _sha(snippet)
            context_spans = [
                {
                    "text": text[heading.start:heading.end],
                    "start": heading.start,
                    "end": heading.end,
                    "sha256": _sha(text[heading.start:heading.end]),
                    "level": heading.level,
                    "kind": "heading",
                }
                for heading in block.headings
                if not start <= heading.start < heading.end <= end
            ]
            if block.kind == "table":
                table_lines = text[block.start:block.end].splitlines(keepends=True)
                if len(table_lines) >= 2 and _separator(table_lines[1]):
                    header_end = block.start + len(table_lines[0]) + len(table_lines[1])
                    header_text = text[block.start:header_end]
                    context_spans.append({
                        "text": header_text,
                        "start": block.start,
                        "end": header_end,
                        "sha256": _sha(header_text),
                        "level": None,
                        "kind": "table_header",
                    })
            derived_metadata = {
                "document_id": document.id_,
                "source_text_sha256": source_sha,
                "source_text_encoding": "utf-8",
                "source_span": {
                    "start": start,
                    "end": end,
                    "unit": "unicode_code_points",
                    "sha256": chunk_sha,
                },
                "chunk_text_sha256": chunk_sha,
                "content_type": block.kind,
                "chunk_order": len(nodes),
                "chunker_version": _VERSION,
                "atomic": block.kind != "prose",
                "heading_path": [text[item.start:item.end] for item in block.headings],
                "context_spans": context_spans,
            }
            metadata = {**copy.deepcopy(document.metadata), **derived_metadata}
            identity = json.dumps(
                [_VERSION, document.id_, source_sha, start, end],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            excluded_embed = list(dict.fromkeys([
                *document.excluded_embed_metadata_keys, *derived_metadata,
            ]))
            excluded_llm = list(dict.fromkeys([
                *document.excluded_llm_metadata_keys, *derived_metadata,
            ]))
            nodes.append(TextNode(
                id_=f"{_VERSION}:{_sha(identity)}",
                text=snippet,
                metadata=metadata,
                start_char_idx=start,
                end_char_idx=end,
                relationships={
                    NodeRelationship.SOURCE: copy.deepcopy(document.as_related_node_info()),
                },
                excluded_embed_metadata_keys=excluded_embed,
                excluded_llm_metadata_keys=excluded_llm,
            ))
            if end == block.end:
                break
            start = end - overlap_characters
    return nodes
