import re

from .lexical_index import lexical_tokens, query_tokens
from .schemas import SourceItem


_REFERENCE_MARK = re.compile(r"\[(?:\d{1,4}|來源請求|来源请求|需要来源)\]")
_HTML_TAG = re.compile(r"<[^>]{1,80}>")
_SPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_LIST_LINE = re.compile(
    r"(?P<label>[^：:\n]{1,48}(?:列表|一览|一覽|站点|車站|车站|站名|名称|民族|成员|项目|條目|条目))"
    r"[：:]\s*(?P<items>[^\n]{3,2400})"
)
_TABLE_FIELD = re.compile(
    r"(?:^|[；;\n])\s*"
    r"(?P<label>[^：:\n；;]{0,32}(?:站名|车站|車站|站点|名稱|名称|民族|成员|项目|條目|条目)[^：:\n；;]{0,24})"
    r"[：:]\s*(?P<value>[^；;\n]{1,80})"
)
_BULLET_ITEM = re.compile(r"^\s*[-*•]\s*(?P<name>[^：:\n]{2,32})(?:[：:]\s*(?P<desc>[^\n]{0,500}))?\s*$")
_ITEM_SPLIT = re.compile(r"[、,，;；]")
_TRAILING_PUNCTUATION = "。.!！?？；;，,"
_LIST_QUESTION_MARKERS = (
    "哪些",
    "有哪",
    "列表",
    "全部",
    "所有",
    "分别",
    "几个",
    "列一下",
    "列出",
    "列举",
)
_LIST_LABEL_NOISE = {"列表", "一览", "一覽", "全部"}
_GENERIC_LIST_TOKENS = {"中国", "中华人民共和国", "著名", "哪些", "有哪", "列表", "全部", "所有", "分别", "几个"}
_CAPITAL_QUESTION_MARKERS = ("首都", "国都", "首府")
_CJK_NAME = r"[\u3400-\u4dbf\u4e00-\u9fff]{2,16}"
_CITY_NAME = r"[\u3400-\u4dbf\u4e00-\u9fff]{2,6}"
_CAPITAL_DECISION_SENTENCE = re.compile(rf"(?:国都|首都)定于(?P<city>{_CITY_NAME})")
_CAPITAL_IS_SENTENCE = re.compile(rf"(?:国都|首都)(?:是|为)(?P<city>{_CITY_NAME})(?:[，,。；;\n]|$)")
_CITY_AS_CAPITAL_SENTENCE = re.compile(rf"(?P<city>{_CITY_NAME})(?:是|为)[^。；\n]{{0,20}}(?:国都|首都)")
_CITY_AS_SUBJECT_CAPITAL_SENTENCE = re.compile(
    rf"(?P<city>{_CITY_NAME})作为[^。；\n]{{0,20}}(?:国都|首都)"
)
_RENAMED_CAPITAL_SENTENCE = re.compile(rf"改名(?P<old>{_CITY_NAME})为(?P<city>{_CITY_NAME})")
_INVALID_CITY_FRAGMENTS = {
    "可能",
    "成为",
    "新中国",
    "将来",
    "已经",
    "预料",
    "历史",
    "文化",
    "未来",
    "能成",
}
_SUBJECT_EQUIVALENTS = {
    "中国": {"中华人民共和国"},
    "中华人民共和国": {"中国"},
}


def clean_evidence_text(text: str) -> str:
    """Clean retrieval snippets before returning them to users or generation models."""

    cleaned = _HTML_TAG.sub("", text)
    cleaned = _REFERENCE_MARK.sub("", cleaned)
    cleaned = cleaned.replace("\u200b", "").replace("\ufeff", "")
    cleaned = "\n".join(_SPACE.sub(" ", line).strip() for line in cleaned.splitlines())
    return _BLANK_LINES.sub("\n\n", cleaned).strip()


def structured_list_answer(question: str, sources: list[SourceItem]) -> str | None:
    """Return a deterministic answer when evidence already contains a complete list."""

    if not _looks_like_list_question(question):
        return None
    question_tokens = set(query_tokens(question))
    context = "\n".join(clean_evidence_text(source.snippet) for source in sources)
    for index, source in enumerate(sources, start=1):
        if not _source_can_supply_list(source):
            continue
        match = _best_list_match(question_tokens, source.snippet)
        if match is None:
            continue
        label, items = match
        items = _repair_items_from_context(question, items, context)
        if len(items) < 3:
            continue
        subject = _answer_subject(question, source)
        normalized_label = _normalize_label(label)
        joined = "、".join(items)
        if normalized_label:
            return f"{subject}的{normalized_label}包括：{joined}。[资料 {index}]"
        return f"{subject}包括：{joined}。[资料 {index}]"
    aggregate = _best_table_field_match(question_tokens, sources)
    if aggregate is not None:
        index, subject, label, items = aggregate
        items = _repair_items_from_context(question, items, context)
        joined = "、".join(items)
        return f"{subject}的{_normalize_label(label)}包括：{joined}。[资料 {index}]"
    bullet = _best_bullet_list_match(question_tokens, sources)
    if bullet is not None:
        index, subject, label, items = bullet
        joined = "、".join(items)
        return f"{subject}的{_normalize_label(label)}包括：{joined}。[资料 {index}]"
    return None


def direct_evidence_answer(question: str, sources: list[SourceItem]) -> str | None:
    return structured_list_answer(question, sources) or _capital_answer(question, sources)


def _looks_like_list_question(question: str) -> bool:
    return any(marker in question for marker in _LIST_QUESTION_MARKERS)


def _capital_answer(question: str, sources: list[SourceItem]) -> str | None:
    if not any(marker in question for marker in _CAPITAL_QUESTION_MARKERS):
        return None
    subject = _capital_subject(question)
    for index, source in enumerate(sources, start=1):
        text = clean_evidence_text(source.snippet)
        title_and_first_line = f"{source.title}\n{text.splitlines()[0] if text else ''}"
        subject_terms = {subject, *_SUBJECT_EQUIVALENTS.get(subject, set())} if subject else set()
        if subject_terms and not any(
            term and (term in title_and_first_line or term in text[:200])
            for term in subject_terms
        ):
            continue
        hierarchy_answer = _capital_from_hierarchy(text)
        if hierarchy_answer:
            return f"{subject or source.title}的首都是{hierarchy_answer}。[资料 {index}]"
        sentence_answer = _capital_from_sentence(text)
        if sentence_answer:
            return f"{subject or source.title}的首都是{sentence_answer}。[资料 {index}]"
    return None


def _capital_subject(question: str) -> str:
    for marker in _CAPITAL_QUESTION_MARKERS:
        position = question.find(marker)
        if position > 0:
            return _clean_subject(question[:position])
    return ""


def _clean_subject(value: str) -> str:
    cleaned = value
    for token in ("你知道", "请问", "现在", "当前", "目前", "的", "是", "在"):
        cleaned = cleaned.replace(token, "")
    return cleaned.strip(" ？?，,。；;")


def _capital_from_hierarchy(text: str) -> str | None:
    first_line = text.splitlines()[0] if text else ""
    if ">" not in first_line or "首都" not in first_line:
        return None
    candidate = first_line.split(">")[-1].strip()
    if re.fullmatch(_CJK_NAME, candidate):
        return candidate
    return None


def _capital_from_sentence(text: str) -> str | None:
    for pattern in (
        _RENAMED_CAPITAL_SENTENCE,
        _CAPITAL_DECISION_SENTENCE,
        _CAPITAL_IS_SENTENCE,
        _CITY_AS_CAPITAL_SENTENCE,
        _CITY_AS_SUBJECT_CAPITAL_SENTENCE,
    ):
        for match in pattern.finditer(text):
            candidate = match.group("city").strip()
            if _valid_city_name(candidate):
                return candidate
    return None


def _valid_city_name(candidate: str) -> bool:
    if not re.fullmatch(_CITY_NAME, candidate):
        return False
    return not any(fragment in candidate for fragment in _INVALID_CITY_FRAGMENTS)


def _source_can_supply_list(source: SourceItem) -> bool:
    content_type = str(source.metadata.get("content_type") or "")
    if content_type in {"table_summary", "table", "list", "qa"}:
        return True
    return bool(
        _LIST_LINE.search(source.snippet)
        or _TABLE_FIELD.search(source.snippet)
        or _BULLET_ITEM.search(source.snippet)
    )


def _best_list_match(question_tokens: set[str], text: str) -> tuple[str, list[str]] | None:
    best: tuple[float, str, list[str]] | None = None
    cleaned = clean_evidence_text(text)
    for match in _LIST_LINE.finditer(cleaned):
        label = match.group("label").strip()
        items = _split_items(match.group("items"))
        if len(items) < 3:
            continue
        label_tokens = set(lexical_tokens(label))
        overlap = len(question_tokens & label_tokens)
        score = overlap + min(len(items), 20) / 100
        if best is None or score > best[0]:
            best = (score, label, items)
    if best is None:
        return None
    return best[1], best[2]


def _best_table_field_match(
    question_tokens: set[str],
    sources: list[SourceItem],
) -> tuple[int, str, str, list[str]] | None:
    groups: dict[str, tuple[float, int, str, list[str]]] = {}
    for source_index, source in enumerate(sources, start=1):
        cleaned = clean_evidence_text(source.snippet)
        for match in _TABLE_FIELD.finditer(cleaned):
            label = match.group("label").strip()
            value = _clean_table_value(match.group("value"))
            if not value:
                continue
            label_tokens = set(lexical_tokens(label))
            overlap = len(question_tokens & label_tokens)
            if "站" in question_tokens and any(marker in label for marker in ("站名", "车站", "車站", "站点")):
                overlap += 2
            key = _normalize_label(label)
            score = float(overlap)
            previous = groups.get(key)
            if previous is None:
                groups[key] = (score, source_index, source.title, [value])
                continue
            previous_score, first_source, title, values = previous
            if value not in values:
                values.append(value)
            groups[key] = (max(previous_score, score), first_source, title, values)
    best: tuple[float, int, str, str, list[str]] | None = None
    for label, (score, source_index, title, values) in groups.items():
        if len(values) < 3:
            continue
        candidate = (score + min(len(values), 20) / 100, source_index, title, label, values)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return None
    _score, source_index, title, label, values = best
    return source_index, title or "资料中", label, values


def _best_bullet_list_match(
    question_tokens: set[str],
    sources: list[SourceItem],
) -> tuple[int, str, str, list[str]] | None:
    specific_tokens = question_tokens - _GENERIC_LIST_TOKENS
    best: tuple[float, int, str, str, list[str]] | None = None
    for source_index, source in enumerate(sources, start=1):
        cleaned = clean_evidence_text(source.snippet)
        lines = cleaned.splitlines()
        label = _list_label_from_context(lines)
        context = f"{source.title}\n{lines[0] if lines else ''}\n{label}"
        context_tokens = set(lexical_tokens(context))
        matched_specific = specific_tokens & context_tokens
        if specific_tokens and not matched_specific:
            continue
        names = [_clean_bullet_name(match.group("name")) for line in lines if (match := _BULLET_ITEM.match(line))]
        items = [name for name in names if name]
        items = list(dict.fromkeys(items))
        if len(items) < 3:
            continue
        score = len(matched_specific) * 3 + len(question_tokens & context_tokens)
        candidate = (float(score), source_index, source.title or "资料中", label, items)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return None
    _score, source_index, subject, label, items = best
    return source_index, subject, label, items


def _split_items(raw: str) -> list[str]:
    items: list[str] = []
    for part in _ITEM_SPLIT.split(raw):
        item = clean_evidence_text(part).strip().strip(_TRAILING_PUNCTUATION)
        if not item or item in {"等", "其他"}:
            continue
        if len(item) > 48:
            break
        if item not in items:
            items.append(item)
    return items


def _clean_table_value(raw: str) -> str:
    value = clean_evidence_text(raw).strip().strip(_TRAILING_PUNCTUATION)
    value = re.sub(r"^[\-*•]\s*", "", value).strip()
    if not value or value in {"—", "-", "无", "未知"}:
        return ""
    if len(value) > 48:
        return ""
    return value


def _list_label_from_context(lines: list[str]) -> str:
    first_line = lines[0].strip() if lines else ""
    if ">" in first_line:
        return first_line.split(">")[-1].strip()
    return first_line or "列表"


def _clean_bullet_name(raw: str) -> str:
    name = clean_evidence_text(raw).strip().strip(_TRAILING_PUNCTUATION)
    name = re.sub(r"^[\-*•]\s*", "", name).strip()
    if not name or name in {"参考资料", "外部链接", "官方网站"}:
        return ""
    if len(name) > 24:
        return ""
    return name


def _repair_items_from_context(question: str, items: list[str], context: str) -> list[str]:
    if not any(marker in question for marker in ("地铁", "鐵路", "铁路", "车站", "車站", "站点", "站名")):
        return items
    repaired: list[str] = []
    for item in items:
        replacement = _repair_station_name(item, context)
        repaired.append(replacement if replacement not in repaired else item)
    return repaired


def _repair_station_name(item: str, context: str) -> str:
    if len(item) != 1:
        return item
    pattern = re.compile(rf"([\u3400-\u4dbf\u4e00-\u9fff]{{1,8}}{re.escape(item)})站")
    matches = [match.group(1) for match in pattern.finditer(context)]
    candidates = [
        re.split(r"[、，,；;：:\s]|和|及|与|與|或", value)[-1]
        for value in matches
    ]
    candidates = [
        value
        for value in candidates
        if value != item and len(value) <= 4 and not value.endswith(("地铁", "铁路"))
    ]
    return candidates[0] if candidates else item


def _answer_subject(question: str, source: SourceItem) -> str:
    title = source.title.strip()
    if title and title in question:
        return title
    return title or "资料中"


def _normalize_label(label: str) -> str:
    parts = [part.strip() for part in label.split("/") if part.strip()]
    normalized = next(
        (
            part
            for part in parts
            if any(marker in part for marker in ("站名", "车站", "車站", "站点", "名称", "民族", "关隘", "关城", "关口"))
        ),
        parts[-1] if parts else label,
    )
    for noise in _LIST_LABEL_NOISE:
        normalized = normalized.replace(noise, "")
    return normalized.strip() or "项目"
