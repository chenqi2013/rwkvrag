import hashlib
import re
from dataclasses import dataclass

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode


_HEADING = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-+*]|\d+[.)]|[（(]?[一二三四五六七八九十]+[）).、])\s+\S")
_KEY_VALUE = re.compile(r"^\s*[^|：:\n，。；！？]{1,24}[：:]\s*[^。！？\n]{1,120}$")
_DATE_PREFIX = re.compile(
    r"^\s*(?:\d{4}(?:[-/.年]\d{1,2})?(?:[-/.月]\d{1,2})?日?|\d{1,2}月\d{1,2}日)"
)
_TABLE_SEPARATOR = re.compile(r"^:?-{3,}:?$")
_REFERENCE_MARK = re.compile(r"\[(?:\d{1,4}|來源請求|来源请求|需要来源)\]")
_SPACE = re.compile(r"\s+")
_TABLE_FOOTERS = {"注释", "备注", "说明", "参考资料", "参考文献", "来源", "论 编"}


@dataclass(frozen=True)
class StructureBlock:
    content_type: str
    section: str
    lines: list[str]


def _stable_id(*parts: object, length: int = 32) -> str:
    raw = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


def _clean(value: str) -> str:
    return _SPACE.sub(" ", _REFERENCE_MARK.sub("", value)).strip()


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.count("|") >= 2


def _table_cells(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [_clean(cell) for cell in stripped.split("|")]


def _is_separator_row(row: list[str]) -> bool:
    return bool(row) and all(not cell or _TABLE_SEPARATOR.fullmatch(cell) for cell in row)


def _section_name(headings: list[str], fallback: str) -> str:
    return " > ".join(headings) if headings else fallback


def parse_structure_blocks(text: str, fallback_section: str) -> list[StructureBlock]:
    lines = text.splitlines()
    blocks: list[StructureBlock] = []
    prose: list[str] = []
    headings: list[str] = []

    def flush_prose() -> None:
        content = "\n".join(prose).strip()
        prose.clear()
        if not content:
            return
        content_lines = content.splitlines()
        nonempty = [line for line in content_lines if line.strip()]
        content_type = "timeline" if nonempty and all(_DATE_PREFIX.match(line) for line in nonempty) else "prose"
        blocks.append(
            StructureBlock(content_type, _section_name(headings, fallback_section), content_lines)
        )

    index = 0
    while index < len(lines):
        line = lines[index]
        heading = _HEADING.match(line.strip())
        if heading:
            flush_prose()
            level = len(heading.group("marks"))
            title = _clean(heading.group("title"))
            headings[level - 1 :] = [title]
            index += 1
            continue

        if _is_table_line(line):
            flush_prose()
            table_lines = []
            while index < len(lines) and _is_table_line(lines[index]):
                table_lines.append(lines[index])
                index += 1
            blocks.append(
                StructureBlock("table", _section_name(headings, fallback_section), table_lines)
            )
            continue

        if _LIST_ITEM.match(line):
            flush_prose()
            list_lines = []
            while index < len(lines):
                candidate = lines[index]
                if _LIST_ITEM.match(candidate) or (candidate.startswith(("  ", "\t")) and candidate.strip()):
                    list_lines.append(candidate)
                    index += 1
                    continue
                break
            blocks.append(
                StructureBlock("list", _section_name(headings, fallback_section), list_lines)
            )
            continue

        if _KEY_VALUE.match(line):
            key_values = []
            cursor = index
            while cursor < len(lines) and _KEY_VALUE.match(lines[cursor]):
                key_values.append(lines[cursor])
                cursor += 1
            if len(key_values) >= 2:
                flush_prose()
                blocks.append(
                    StructureBlock(
                        "key_value",
                        _section_name(headings, fallback_section),
                        key_values,
                    )
                )
                index = cursor
                continue

        prose.append(line)
        index += 1

    flush_prose()
    return blocks


def _group_lines(lines: list[str], max_chars: int) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for line in lines:
        size = len(line) + 1
        if current and current_size + size > max_chars:
            groups.append(current)
            current = []
            current_size = 0
        current.append(line)
        current_size += size
    if current:
        groups.append(current)
    return groups


def _table_texts(
    block: StructureBlock,
    max_chars: int,
    *,
    document_context: str = "",
) -> list[tuple[str, str]]:
    rows = [_table_cells(line) for line in block.lines]
    separator = next((index for index, row in enumerate(rows) if _is_separator_row(row)), None)
    if separator is None:
        header_rows = rows[:1]
        data_rows = rows[1:]
    else:
        header_rows = rows[:separator]
        data_rows = rows[separator + 1 :]
    if not header_rows:
        return [("table", "\n".join(block.lines))]

    column_count = max(len(row) for row in header_rows)
    headers: list[str] = []
    for column in range(column_count):
        values = []
        for row in header_rows:
            value = row[column] if column < len(row) else ""
            if value and value not in values:
                values.append(value)
        headers.append("/".join(values) or f"字段{column + 1}")

    normalized_rows: list[list[str]] = []
    for row in data_rows:
        padded = [*(row[:column_count]), *([""] * max(0, column_count - len(row)))]
        if padded and padded[0] in _TABLE_FOOTERS:
            break
        if any(padded):
            normalized_rows.append(padded)
    _repair_table_first_column(headers, normalized_rows, block, document_context)

    first_values = [row[0] for row in normalized_rows if row and row[0]]
    texts: list[tuple[str, str]] = []
    if first_values:
        summary_lines = [
            f"{block.section}",
            f"表格字段：{'、'.join(headers)}",
            f"{headers[0]}列表：{'、'.join(first_values)}",
        ]
        for group in _group_lines(summary_lines, max_chars):
            texts.append(("table_summary", "\n".join(group)))

    row_lines = []
    for row in normalized_rows:
        values = [f"{header}：{value}" for header, value in zip(headers, row) if value]
        if values:
            row_lines.append("；".join(values))
    for group in _group_lines(row_lines, max_chars):
        texts.append(("table", f"{block.section}\n" + "\n".join(group)))
    return texts or [("table", "\n".join(block.lines))]


def _repair_table_first_column(
    headers: list[str],
    rows: list[list[str]],
    block: StructureBlock,
    document_context: str,
) -> None:
    if not rows or not document_context:
        return
    first_header = headers[0] if headers else ""
    table_context = f"{block.section} {first_header}"
    if not any(marker in table_context for marker in ("站名", "车站", "車站", "站点")):
        return
    for row in rows:
        if not row or len(row[0]) != 1:
            continue
        row[0] = _repair_station_name(row[0], document_context)


def _repair_station_name(value: str, document_context: str) -> str:
    pattern = re.compile(rf"([\u3400-\u4dbf\u4e00-\u9fff]{{1,8}}{re.escape(value)})站")
    for match in pattern.finditer(document_context):
        candidate = re.split(r"[、，,；;：:\s]|和|及|与|與|或", _clean(match.group(1)))[-1]
        if candidate != value and len(candidate) <= 4 and not candidate.endswith(("地铁", "铁路")):
            return candidate
    return value


def _structured_texts(
    block: StructureBlock,
    max_chars: int,
    *,
    document_context: str = "",
) -> list[tuple[str, str]]:
    if block.content_type == "table":
        return _table_texts(block, max_chars, document_context=document_context)
    clean_lines = [_clean(line) for line in block.lines if _clean(line)]
    return [
        (block.content_type, f"{block.section}\n" + "\n".join(group))
        for group in _group_lines(clean_lines, max_chars)
    ]


def _keywords(content_type: str, section: str) -> list[str]:
    common = {
        "table": ["表格", "字段", "数据"],
        "table_summary": ["表格", "列表", "全部"],
        "list": ["列表", "条目", "全部"],
        "key_value": ["参数", "属性", "字段"],
        "timeline": ["时间", "日期", "事件"],
        "qa": ["问题", "答案"],
    }
    return list(dict.fromkeys([section, *common.get(content_type, [])]))


def structure_aware_nodes(document: Document, splitter: SentenceSplitter) -> list[TextNode]:
    metadata = dict(document.metadata)
    document_id = str(metadata.get("document_id") or document.id_)
    fallback_section = str(metadata.get("title") or "正文")
    max_chars = max(600, splitter.chunk_size * 3)

    if metadata.get("kind") == "qa-markdown":
        blocks = [StructureBlock("qa", fallback_section, [document.text])]
    else:
        blocks = parse_structure_blocks(document.text, fallback_section)

    nodes: list[TextNode] = []
    for block_index, block in enumerate(blocks):
        parent_id = _stable_id(document_id, block_index, block.content_type, block.section, length=24)
        if block.content_type == "prose":
            content = f"{block.section}\n\n" + "\n".join(block.lines)
            pieces = [("prose", piece) for piece in splitter.split_text(content)]
        elif block.content_type == "qa":
            pieces = [("qa", "\n".join(block.lines))]
        else:
            pieces = _structured_texts(block, max_chars, document_context=document.text)

        structure_size = len(pieces)
        for chunk_order, (content_type, text) in enumerate(pieces):
            node_metadata = {
                **metadata,
                "section": block.section,
                "content_type": content_type,
                "parent_id": parent_id,
                "chunk_order": chunk_order,
                "structure_size": structure_size,
                "keywords": _keywords(content_type, block.section),
            }
            node_id = _stable_id(document_id, parent_id, chunk_order, content_type, text)
            nodes.append(
                TextNode(
                    id_=node_id,
                    text=text.strip(),
                    metadata=node_metadata,
                    excluded_embed_metadata_keys=list(document.excluded_embed_metadata_keys),
                    excluded_llm_metadata_keys=list(document.excluded_llm_metadata_keys),
                )
            )
    return nodes
