import re

from .evidence_quality import is_repetitive_garbage
from .lexical_index import lexical_tokens, normalize_search_text, query_tokens
from .qa_analysis import counted_list_size
from .question_patterns import is_agent_relation_question
from .schemas import SourceItem


_REFERENCE_MARK = re.compile(r"\[(?:\d{1,4}|來源請求|来源请求|需要来源)\]")
_HTML_TAG = re.compile(r"<[^>]{1,80}>")
_SPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_LIST_LINE = re.compile(
    r"(?:^|\n)\s*"
    r"(?P<label>[^：:\n；;]{1,48}(?:列表|一览|一覽|站点|車站|车站|站名|名称|民族|成员|项目|條目|条目|朝代|王朝))"
    r"[：:]\s*(?P<items>[^\n]{3,2400})"
)
_TABLE_FIELD = re.compile(
    r"(?:^|[；;\n])\s*"
    r"(?P<label>[^：:\n；;]{0,32}(?:站名|车站|車站|站点|名稱|名称|民族|成员|项目|條目|条目|朝代|王朝)[^：:\n；;]{0,24})"
    r"[：:]\s*(?P<value>[^；;\n]{1,80})"
)
_BULLET_ITEM = re.compile(r"^\s*[-*•]\s*(?P<name>[^：:\n]{2,32})(?:[：:]\s*(?P<desc>[^\n]{0,500}))?\s*$")
_BULLET_ROW = re.compile(r"^\s*[-*•]\s*(?P<value>\S.{1,500})\s*$")
_ITEM_SPLIT = re.compile(r"[、,，;；]")
_TRAILING_PUNCTUATION = "。.!！?？；;，,"
_LIST_QUESTION_MARKERS = (
    "哪些",
    "有哪",
    "列表",
    "全部",
    "所有",
    "分别",
    "列一下",
    "列出",
    "列举",
)
_LIST_LABEL_NOISE = {"列表", "一览", "一覽", "全部"}
_GENERIC_LIST_TOKENS = {"中国", "中华人民共和国", "著名", "哪些", "有哪", "列表", "全部", "所有", "分别"}
_CAPITAL_QUESTION_MARKERS = ("首都", "国都", "首府")
_CJK_NAME = r"[\u3400-\u4dbf\u4e00-\u9fff]{2,16}"
_CITY_NAME = r"[\u3400-\u4dbf\u4e00-\u9fff]{2,6}"
_CAPITAL_DECISION_SENTENCE = re.compile(rf"(?:国都|首都)定于(?P<city>{_CITY_NAME})")
_CAPITAL_IS_SENTENCE = re.compile(rf"(?:国都|首都)(?:是|为)(?P<city>{_CITY_NAME})(?:[，,。；;\n]|$)")
_CITY_AS_CAPITAL_SENTENCE = re.compile(
    rf"(?:^|[。！？!?；;\n])(?:现时|目前|当前|如今)?"
    rf"(?P<city>{_CITY_NAME})(?:自[^，,。；;\n]{{0,24}})?"
    rf"(?:是|为|定为|设为|確立為|确立为)"
    rf"(?:中华人民共和国|中華人民共和國|中国|中國|该国|該國|其)?(?:的)?"
    rf"(?:国都|國都|首都)"
)
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
_DEFINITION_QUESTION_MARKERS = (
    "是什么",
    "是谁",
    "指的是什么",
    "简要介绍",
)
_PURE_DEFINITION_QUESTION = re.compile(r"(?:是什么|是谁|指的是什么)[。？?]?$")
_SENTENCE = re.compile(r"[^。！？!?\n]{8,500}[。！？!?]?")
_DEFINITION_RELATION_PATTERN = re.compile(
    r"^.{0,80}?(?:是|(?<!作)为|指|属于|位于|成立于|创建于|出生于)"
)
_EXPLICIT_TIME = re.compile(r"(?:公元|西元)?\d{3,4}年(?:\d{1,2}月(?:\d{1,2}日)?)?")
_FULL_DATE = re.compile(r"(?:公元|西元)?\d{3,4}年\d{1,2}月\d{1,2}日")
_YEAR_MONTH = re.compile(r"(?:公元|西元)?\d{3,4}年\d{1,2}月")
_RELATIVE_TIME = re.compile(r"(?:同年|次年|翌年|当年|其后|随后)")
_TIME_ACTION_GROUPS = {
    "成立": ("成立", "创立", "创建", "组建", "建立", "宣告"),
    "创建": ("创建", "创立", "成立", "建立", "组建"),
    "发生": ("发生", "爆发", "开始"),
    "开通": ("开通", "启用", "通车", "运营"),
    "出生": ("出生", "生于"),
    "逝世": ("逝世", "去世", "病逝", "卒于"),
    "上映": ("上映", "首映", "发行"),
    "回归": ("回归", "政权移交", "主权移交", "恢复行使主权", "交接"),
}
_AGENT_ACTION_GROUPS = {
    "发起": ("发起", "发动", "率军", "率领", "南攻", "进攻", "攻打"),
    "提出": ("提出", "提议", "倡议"),
    "创立": ("创立", "创建", "建立", "组建"),
    "创建": ("创建", "创立", "建立", "组建"),
    "发明": ("发明", "研制", "创造"),
    "发现": ("发现", "首次发现"),
    "开启": ("开启", "开辟", "开通", "出使", "凿空"),
    "开辟": ("开辟", "开启", "开通", "出使", "凿空"),
    "开通": ("开通", "开启", "开辟", "通达"),
    "领导": ("领导", "率领", "指挥"),
    "指挥": ("指挥", "统率", "率领"),
    "主演": ("主演", "饰演"),
    "导演": ("导演", "执导"),
    "执导": ("执导", "导演"),
    "撰写": ("撰写", "著", "编写"),
    "创作": ("创作", "作曲", "作词", "撰写"),
    "设计": ("设计", "设计者"),
    "建造": ("建造", "修建", "建设"),
    "开发": ("开发", "研发"),
    "制作": ("制作", "制片"),
    "主持": ("主持", "主导"),
    "组织": ("组织", "筹办"),
    "推动": ("推动", "促成"),
    "负责": ("负责", "负责人"),
    "开国皇帝": ("开国皇帝", "高祖", "太祖", "称帝", "登基"),
    "创始人": ("创始人", "创立", "创建", "创办"),
    "创办人": ("创办人", "创立", "创建", "创办"),
    "创办者": ("创办者", "创办人", "创立", "创建", "创办"),
    "建立者": ("建立者", "建立", "创立", "创建"),
    "创建者": ("创建者", "创建", "创立", "建立"),
    "发明者": ("发明者", "发明", "研制", "创造"),
    "发现者": ("发现者", "发现", "首次发现"),
    "作者": ("作者", "撰写", "创作", "著"),
    "创作者": ("创作者", "作者", "撰写", "创作", "著"),
    "设计者": ("设计者", "设计"),
    "执导者": ("执导者", "执导", "导演"),
    "得主": ("得主", "获得", "获奖", "授予"),
    "获得者": ("获得者", "得主", "获得", "获奖", "授予"),
    "现在": ("现任", "目前", "当前"),
    "目前": ("现任", "目前", "当前"),
    "当前": ("现任", "目前", "当前"),
    "现任": ("现任", "目前", "当前"),
}
_LOCATION_RELATIONS = ("位于", "位於", "坐落于", "坐落於", "地处", "地處")
_BIRTH_RELATIONS = ("出生于", "出生於", "生于", "生於", "出生地")
_FACT_QUERY_NOISE = {
    "中国", "时候", "哪一年", "何时", "谁", "哪里", "什么", "事件", "时期",
}
_QUANTITATIVE_LABELS = ("人口", "面积", "全长", "长度", "高度", "海拔")
_FLATTENED_STATION_NAME = re.compile(
    r"(?P<name>[\u3400-\u4dbf\u4e00-\u9fff]{2,16}站)"
    r"(?=[A-Z][A-Za-z' .&-]{1,48}(?:150x?150(?:像素)?|150px))"
)
_TRANSIT_REGION_PREFIX = re.compile(
    r"(?:罗湖|羅湖|福田|南山|宝安|寶安|龙岗|龍崗|龙华|龍華|光明|盐田|鹽田|坪山)[区區]"
)
_FULL_CHINESE_DATE = re.compile(r"\d{4}年\d{1,2}月\d{1,2}日")


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
    counted_answer = _counted_enumeration_answer(question, sources)
    if counted_answer is not None:
        return counted_answer
    flattened_station_answer = _flattened_station_table_answer(question, sources)
    if flattened_station_answer is not None:
        return flattened_station_answer
    question_tokens = set(query_tokens(question))
    context = "\n".join(clean_evidence_text(source.snippet) for source in sources)
    for index, source in enumerate(sources, start=1):
        if not _source_can_supply_list(source):
            continue
        match = _best_list_match(question_tokens, source.snippet)
        if match is None:
            continue
        label, items = match
        if not _list_label_matches_question(question, label):
            continue
        if any("：" in item or ":" in item for item in items):
            continue
        items = _repair_items_from_context(question, items, context)
        if len(items) < 3:
            continue
        subject = _answer_subject(question, source)
        normalized_label = _normalize_label(label)
        joined = "、".join(items)
        if normalized_label:
            return f"{subject}的{normalized_label}包括：{joined}。[资料 {index}]"
        return f"{subject}包括：{joined}。[资料 {index}]"
    aggregate = _best_table_field_match(question, question_tokens, sources)
    if aggregate is not None:
        index, subject, label, items = aggregate
        items = _repair_items_from_context(question, items, context)
        joined = "、".join(items)
        return f"{subject}的{_normalize_label(label)}包括：{joined}。[资料 {index}]"
    bullet = _best_bullet_list_match(question, question_tokens, sources)
    if bullet is not None:
        index, subject, label, items = bullet
        joined = "、".join(items)
        return f"{subject}的{_normalize_label(label)}包括：{joined}。[资料 {index}]"
    return None


def _counted_enumeration_answer(
    question: str,
    sources: list[SourceItem],
) -> str | None:
    expected_count = counted_list_size(question)
    if expected_count is None:
        return None
    relation_pattern = re.compile(r"(?:是指|包括|包含|分别是|分別是|分别为|分別為|即为|即為|即)")
    for source_index, source in enumerate(sources, start=1):
        text = clean_evidence_text(source.snippet)
        if source.title and source.title not in question:
            title_terms = {
                term for term in lexical_tokens(source.title) if len(term) >= 2
            }
            question_terms = set(query_tokens(question))
            if title_terms and not (title_terms & question_terms):
                continue
        for sentence in re.split(r"[。！？!?；;\n]", text[:4000]):
            if not relation_pattern.search(sentence):
                continue
            quoted_items = [
                f"《{value.strip()}》"
                for value in re.findall(r"《([^《》\n]{1,48})》", sentence)
                if value.strip()
            ]
            quoted_items = list(dict.fromkeys(quoted_items))
            if len(quoted_items) == expected_count:
                subject = _answer_subject(question, source)
                return (
                    f"{subject}包括：{'、'.join(quoted_items)}。"
                    f"[资料 {source_index}]"
                )
            relation_matches = list(relation_pattern.finditer(sentence))
            if not relation_matches:
                continue
            tail = sentence[relation_matches[-1].end():].strip(" ：:，,")
            tail = re.sub(r"(?:共)?\d{1,2}(?:个|部|本|种|位|家|条|项|篇|首|座|名).*$", "", tail)
            items = [
                value.strip(" 《》“”\"'（）()，,：:")
                for value in re.split(r"[、，,]|(?:以及|及|和|与|跟)", tail)
            ]
            items = list(dict.fromkeys(
                value for value in items if 1 < len(value) <= 24
            ))
            if len(items) != expected_count:
                continue
            subject = _answer_subject(question, source)
            return f"{subject}包括：{'、'.join(items)}。[资料 {source_index}]"
    return None


def _flattened_station_table_answer(
    question: str,
    sources: list[SourceItem],
) -> str | None:
    if not any(marker in question for marker in ("车站", "站点", "哪些站", "站名")):
        return None
    expected_count = next(
        (
            int(match.group(1))
            for source in sources
            if (match := re.search(
                r"(?:共|共有|总计|總計|沿途共)\s*(?:设|設|有)?\s*(\d{1,3})\s*(?:个|個|座)?车站",
                normalize_search_text(source.snippet),
            ))
        ),
        None,
    )
    if expected_count is None:
        return None
    station_names: list[str] = []
    citation_indexes: list[int] = []
    for index, source in enumerate(sources, start=1):
        if not normalize_search_text(source.title).replace(" ", "").endswith("车站列表"):
            continue
        text = clean_evidence_text(source.snippet)
        text = re.sub(r"^.*?(?:参考来源|參考來源)", "", text, count=1, flags=re.DOTALL)
        text = re.sub(r"-\{(?P<value>[\u3400-\u9fff]+)\}-", r"\g<value>", text)
        text = _FULL_CHINESE_DATE.sub("", text)
        text = _TRANSIT_REGION_PREFIX.sub("", text)
        found = [
            normalize_search_text(match.group("name")).replace(" ", "")
            for match in _FLATTENED_STATION_NAME.finditer(text)
        ]
        if not found:
            continue
        citation_indexes.append(index)
        for name in found:
            if name not in station_names:
                station_names.append(name)
    if len(station_names) != expected_count:
        return None
    subject = next(
        (
            source.title
            for source in sources
            if re.search(r"地铁\d+号线", normalize_search_text(source.title))
        ),
        _answer_subject(question, sources[0]),
    )
    citations = "".join(f"[资料 {index}]" for index in citation_indexes)
    return f"{subject}共有{expected_count}个车站：{'、'.join(station_names)}。{citations}"


def direct_evidence_answer(question: str, sources: list[SourceItem]) -> str | None:
    return (
        structured_list_answer(question, sources)
        or _capital_answer(question, sources)
        or quantitative_evidence_answer(question, sources)
        or definition_evidence_answer(question, sources)
    )


def list_evidence_answer(question: str, sources: list[SourceItem]) -> str | None:
    """Return grounded list rows when a complete structured list cannot be proven."""

    if not _looks_like_list_question(question):
        return None
    question_tokens = set(query_tokens(question))
    best: tuple[float, int, SourceItem, list[str]] | None = None
    for source_index, source in enumerate(sources, start=1):
        rows = [
            clean_evidence_text(match.group("value")).strip(_TRAILING_PUNCTUATION)
            for line in clean_evidence_text(source.snippet).splitlines()
            if (match := _BULLET_ROW.match(line))
        ]
        rows = [row for row in rows if 2 <= len(row) <= 500][:16]
        if len(rows) < 2:
            continue
        context_tokens = set(query_tokens(
            f"{source.title} {source.metadata.get('section') or ''}"
        ))
        score = len(question_tokens & context_tokens) * 3.0 + min(len(rows), 10)
        candidate = (score, -source_index, source, rows)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        return None
    _score, negative_index, source, rows = best
    source_index = -negative_index
    subject = _answer_subject(question, source)
    return f"根据检索资料，{subject}可确认的内容包括：" + "；".join(rows) + f"。[资料 {source_index}]"


def time_evidence_answer(question: str, sources: list[SourceItem]) -> str | None:
    term_end = _fixed_term_end_answer(question, sources)
    if term_end is not None:
        return term_end
    lifespan = _lifespan_time_answer(question, sources)
    if lifespan is not None:
        return lifespan
    actions = _matched_action_terms(question, _TIME_ACTION_GROUPS)
    origin = _explicit_subject_origin_answer(question, sources, actions)
    if origin is not None:
        return origin
    return _best_fact_sentence(
        question,
        sources,
        actions=actions,
        requires_time=True,
        prefer_explicit_subject=True,
    )


def _fixed_term_end_answer(question: str, sources: list[SourceItem]) -> str | None:
    normalized_question = normalize_search_text(question)
    if "任期" not in normalized_question or "结束" not in normalized_question:
        return None
    for source_index, source in enumerate(sources, start=1):
        text = clean_evidence_text(source.snippet)
        match = re.search(
            r"(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日"
            r"[^。！？\n]{0,80}?(?:获得|当选)?连任[^。！？\n]{0,40}?"
            r"任期(?P<duration>\d{1,2})年",
            text,
        )
        if not match:
            continue
        end_year = int(match.group("year")) + int(match.group("duration"))
        return (
            f"按资料所载连任日期和{match.group('duration')}年任期推算，"
            f"该任期于{end_year}年{match.group('month')}月{match.group('day')}日结束。"
            f"[资料 {source_index}]"
        )
    return None


def _explicit_subject_origin_answer(
    question: str,
    sources: list[SourceItem],
    actions: tuple[str, ...],
) -> str | None:
    normalized_question = normalize_search_text(question)
    if not any(marker in normalized_question for marker in ("成立", "创建", "建立", "创立")):
        return None
    action_pattern = "|".join(
        re.escape(normalize_search_text(action)) for action in actions
    )
    if not action_pattern:
        return None
    for source_index, source in enumerate(sources, start=1):
        normalized_title = normalize_search_text(source.title)
        if not normalized_title or normalized_title not in normalized_question:
            continue
        text = clean_evidence_text(source.snippet)
        match = re.search(
            rf"{re.escape(source.title)}[^。！？\n]{{0,80}}?"
            rf"(?:{action_pattern})(?:时间众说不一[^。！？\n]{{0,30}}?)?(?:是|为)?于"
            rf"[^。！？\n]{{0,30}}?(?:公元前)?\d{{3,4}}年[^。！？\n]{{0,80}}[。！？]?",
            text,
        )
        if match:
            return f"{match.group(0).strip()} [资料 {source_index}]"
        sentence_match = re.search(
            rf"(?P<sentence>[^。！？\n]{{0,100}}{re.escape(source.title)}[^。！？\n]{{0,80}}?"
            rf"(?:{action_pattern})[^。！？\n]{{0,30}}?(?:公元前)?\d{{3,4}}年[^。！？\n]{{0,80}}[。！？]?)",
            text,
        )
        if sentence_match:
            sentence = sentence_match.group("sentence").strip()
            normalized_sentence = normalize_search_text(sentence)
            if not any(
                marker in normalized_sentence
                for marker in ("子公司", "分公司", "总行", "分行", "员工", "雇员", "成员")
            ):
                return f"{sentence} [资料 {source_index}]"
        year_before_action = re.search(
            rf"(?P<sentence>[^。！？\n]{{0,80}}{re.escape(source.title)}[^。！？\n]{{0,100}}?"
            rf"(?:公元前)?\d{{3,4}}年[^。！？\n]{{0,30}}?(?:{action_pattern})[^。！？\n]{{0,80}}[。！？]?)",
            text,
        )
        if year_before_action:
            sentence = year_before_action.group("sentence").strip()
            normalized_sentence = normalize_search_text(sentence)
            if not any(
                marker in normalized_sentence
                for marker in ("子公司", "分公司", "总行", "分行", "员工", "雇员", "成员")
            ):
                return f"{sentence} [资料 {source_index}]"
    return None


def _lifespan_time_answer(question: str, sources: list[SourceItem]) -> str | None:
    normalized_question = normalize_search_text(question)
    if "出生" not in normalized_question and "逝世" not in normalized_question:
        return None
    for source_index, source in enumerate(sources, start=1):
        normalized_title = normalize_search_text(source.title)
        if not normalized_title or normalized_title not in normalized_question:
            continue
        match = re.search(
            rf"{re.escape(source.title)}\s*[（(]"
            r"(?P<birth>(?:公元前)?\d{3,4}年(?:\d{1,2}月(?:\d{1,2}日)?)?)"
            r"[—–-]"
            r"(?P<death>(?:公元前)?\d{3,4}年(?:\d{1,2}月(?:\d{1,2}日)?)?)",
            clean_evidence_text(source.snippet)[:500],
        )
        if not match:
            continue
        value = match.group("birth" if "出生" in normalized_question else "death")
        event = "出生" if "出生" in normalized_question else "逝世"
        return f"{source.title}{event}于{value}。[资料 {source_index}]"
    return None


def coordinated_time_evidence_answer(
    sources: list[SourceItem],
    subjects: tuple[str, ...],
) -> str | None:
    if len(subjects) < 4 or len(subjects) % 2:
        return None
    answers: list[str] = []
    for index in range(0, len(subjects), 2):
        subject = subjects[index]
        event_query = subjects[index + 1]
        event = event_query.removeprefix(subject).strip()
        if not subject or not event:
            return None
        answer = time_evidence_answer(f"{subject}是哪一年{event}的？", sources)
        if answer is None:
            return None
        answers.append(f"{subject}：{answer}")
    return "；".join(answers)


def agent_evidence_answer(question: str, sources: list[SourceItem]) -> str | None:
    actions = _matched_action_terms(question, _AGENT_ACTION_GROUPS)
    if not actions:
        return None
    answer = _best_fact_sentence(
        question,
        sources,
        actions=actions,
        requires_time=False,
        prefer_explicit_agent=True,
    )
    if answer is None:
        return None
    answer = answer.lstrip("（(")
    if any(marker in question for marker in ("从哪里", "哪个地方开始", "从哪开始")):
        location = _origin_location_answer(sources)
        if location and location not in answer:
            answer = f"{answer.rstrip()}；起点为{location}。"
    return answer


def _origin_location_answer(sources: list[SourceItem]) -> str | None:
    patterns = (
        re.compile(r"从(?P<place>[^，,。；;（）()\n]{2,24})出发"),
        re.compile(r"起点(?:是|为|位于)?(?P<place>[^，,。；;（）()\n]{2,24})"),
    )
    for source_index, source in enumerate(sources, start=1):
        text = clean_evidence_text(source.snippet)
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return f"{match.group('place').strip()}[资料 {source_index}]"
    return None


def cause_evidence_answer(sources: list[SourceItem]) -> str | None:
    """Return concise grounded excerpts when a cause synthesis model refuses."""

    event_markers = ("灭亡", "滅亡", "衰落", "解体", "解體", "失败", "失敗", "覆亡")
    event_sources = [
        (index, source)
        for index, source in enumerate(sources, start=1)
        if any(marker in str(source.metadata.get("section") or "") for marker in event_markers)
    ]
    candidates = event_sources or list(enumerate(sources, start=1))
    excerpts: list[tuple[int, str]] = []
    seen: set[str] = set()
    for source_index, source in candidates:
        text = clean_evidence_text(source.snippet)
        lines = text.splitlines()
        if lines and (">" in lines[0] or lines[0].strip() == source.title.strip()):
            text = "\n".join(lines[1:]).strip()
        sentences = [
            match.group(0).strip()
            for match in _SENTENCE.finditer(text[:2500])
            if len(match.group(0).strip()) >= 16
        ]
        if not sentences:
            continue
        cause_markers = (
            "争议", "腐化", "天灾", "严寒", "干旱", "饥荒", "鼠疫", "起义",
            "失败", "攻克", "失守", "财政", "党争", "专权", "导致", "造成",
            "分权", "依附", "失去民心", "衰弱", "矛盾", "殖民", "吞并", "入侵",
        )
        sentence = max(
            sentences[:8],
            key=lambda value: (
                sum(marker in normalize_search_text(value) for marker in cause_markers),
                min(len(value), 220),
            ),
        )
        normalized = normalize_search_text(sentence)
        if normalized in seen:
            continue
        seen.add(normalized)
        excerpts.append((source_index, sentence[:240].rstrip("，,；;。") + "。"))
        if len(excerpts) >= 3:
            break
    if not excerpts:
        return None
    return "资料显示，与该结果相关的因素和过程包括：" + "；".join(
        f"{sentence}[资料 {source_index}]"
        for source_index, sentence in excerpts
    )


def ordinal_evidence_answer(question: str, sources: list[SourceItem]) -> str | None:
    relation_match = re.search(
        r"(?:第一个|第一位|首位|最早的?)(?P<relation>[^，。？?]{1,24}?)"
        r"(?:(?:是|为)?(?:谁|哪位|什么人))?[。？?]?$",
        question,
    )
    if not relation_match:
        relation_match = re.search(
            r"^(?:谁|哪位|什么人)(?:是|为).+?(?:第一个|第一位|首位|最早的?)"
            r"(?P<relation>[^，。？?]{1,24})[。？?]?$",
            question,
        )
    if not relation_match:
        return None
    relation_tokens = set(lexical_tokens(relation_match.group("relation"))) - {
        "的", "是", "为", "谁", "哪位", "什么人",
    }
    ordinal_markers = ("第一个", "第一位", "首位", "最早")
    best: tuple[float, int, str] | None = None
    for source_index, source in enumerate(sources, start=1):
        text = clean_evidence_text(source.snippet)
        if is_repetitive_garbage(text):
            continue
        for order, match in enumerate(_SENTENCE.finditer(text[:4000])):
            sentence = match.group(0).strip()
            normalized_sentence = normalize_search_text(sentence)
            if not any(marker in normalized_sentence for marker in ordinal_markers):
                continue
            sentence_tokens = set(lexical_tokens(sentence))
            relation_overlap = len(relation_tokens & sentence_tokens)
            if relation_tokens and relation_overlap == 0:
                continue
            score = relation_overlap * 4.0
            score += sum(marker in normalized_sentence for marker in ordinal_markers) * 2.0
            score += 1.0 if "中国" in normalized_sentence else 0.0
            score -= order * 0.01
            candidate = (score, source_index, sentence)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        return None
    _score, source_index, sentence = best
    return f"{sentence.rstrip()} [资料 {source_index}]"


def location_evidence_answer(question: str, sources: list[SourceItem]) -> str | None:
    normalized_question = normalize_search_text(question)
    for source_index, source in enumerate(sources, start=1):
        normalized_title = normalize_search_text(source.title)
        if not normalized_title or normalized_title not in normalized_question:
            continue
        text = clean_evidence_text(source.snippet)
        paragraph_match = re.search(
            rf"{re.escape(source.title)}[^。！？\n]{{0,260}}?"
            rf"(?:{'|'.join(map(re.escape, _LOCATION_RELATIONS))})[^。！？\n]{{1,220}}[。！？]?",
            text,
        )
        if paragraph_match:
            return f"{paragraph_match.group(0).strip()} [资料 {source_index}]"
        for match in _SENTENCE.finditer(text[:3000]):
            sentence = match.group(0).strip()
            normalized_sentence = normalize_search_text(sentence)
            if (
                normalized_title in normalized_sentence
                and any(normalize_search_text(relation) in normalized_sentence for relation in _LOCATION_RELATIONS)
            ):
                return f"{sentence.rstrip()} [资料 {source_index}]"
    return _best_fact_sentence(
        question,
        sources,
        actions=_LOCATION_RELATIONS,
        requires_time=False,
        prefer_explicit_subject=True,
    )


def quantitative_evidence_answer(question: str, sources: list[SourceItem]) -> str | None:
    normalized_question = normalize_search_text(question)
    label = next(
        (item for item in _QUANTITATIVE_LABELS if item in normalized_question),
        None,
    )
    if label is None:
        return None
    for source_index, source in enumerate(sources, start=1):
        normalized_title = normalize_search_text(source.title)
        if not normalized_title or normalized_title not in normalized_question:
            continue
        text = clean_evidence_text(source.snippet)
        for sentence in re.split(r"[。！？\n]", text[:3000]):
            normalized_sentence = normalize_search_text(sentence)
            if normalized_title not in normalized_sentence or label not in normalized_sentence:
                continue
            if label == "人口":
                population = re.search(
                    r"(?:(?:19|20)\d{2}年)?人口(?:为|為|是|共|有)?\s*(?P<value>\d[\d,]*(?:\.\d+)?(?:万|萬|亿|億|千)?)(?:人)?",
                    sentence,
                )
                if population:
                    value = population.group("value")
                    return f"{source.title}的人口为{value}人。[资料 {source_index}]"
            elif re.search(rf"{re.escape(label)}[^，,；;]{{0,24}}\d", normalized_sentence):
                return f"{sentence.strip()}。[资料 {source_index}]"
    return None


def birthplace_evidence_answer(question: str, sources: list[SourceItem]) -> str | None:
    return _best_fact_sentence(
        question,
        sources,
        actions=_BIRTH_RELATIONS,
        requires_time=False,
        prefer_explicit_subject=True,
    )


def _matched_action_terms(question: str, groups: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    normalized_question = normalize_search_text(question)
    for marker, terms in sorted(groups.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(
            rf"(?:的|背后的){re.escape(normalize_search_text(marker))}(?:是|为)?"
            r"(?:谁|哪位|什么人)[。？?]?$",
            normalized_question,
        ):
            return terms
    for marker, terms in groups.items():
        if normalize_search_text(marker) in normalized_question:
            return terms
    return ()


def _best_fact_sentence(
    question: str,
    sources: list[SourceItem],
    *,
    actions: tuple[str, ...],
    requires_time: bool,
    prefer_explicit_agent: bool = False,
    prefer_explicit_subject: bool = False,
) -> str | None:
    normalized_question = normalize_search_text(question)
    asks_for_origin = any(
        marker in normalized_question
        for marker in ("开启", "开辟", "成立", "创立", "创建", "建立", "发明", "发现", "最早")
    )
    action_tokens = {
        token
        for action in actions
        for token in lexical_tokens(action)
    }
    anchors = set(query_tokens(question)) - action_tokens - _FACT_QUERY_NOISE
    best: tuple[float, int, str] | None = None
    for source_index, source in enumerate(sources, start=1):
        title_match = bool(source.title and normalize_search_text(source.title) in normalized_question)
        cleaned_source = clean_evidence_text(source.snippet)
        if is_repetitive_garbage(cleaned_source):
            continue
        for order, match in enumerate(_SENTENCE.finditer(cleaned_source[:4000])):
            sentence = match.group(0).strip()
            normalized_sentence = normalize_search_text(sentence)
            candidate_context = normalize_search_text(f"{source.title} {sentence}")
            if anchors and not any(token in candidate_context for token in anchors):
                continue
            explicit_time = bool(_EXPLICIT_TIME.search(sentence))
            relative_time = bool(_RELATIVE_TIME.search(sentence))
            if requires_time and not (explicit_time or relative_time):
                continue
            action_hits = sum(normalize_search_text(term) in normalized_sentence for term in actions)
            if actions and not action_hits:
                continue
            asks_year_only = any(marker in normalized_question for marker in ("哪一年", "年份"))
            time_score = (
                5.0 if asks_year_only and explicit_time
                else 8.0 if _FULL_DATE.search(sentence)
                else 7.0 if _YEAR_MONTH.search(sentence)
                else 5.0 if explicit_time
                else 1.0 if relative_time
                else 0.0
            )
            score = action_hits * 4.0 + time_score
            if prefer_explicit_agent and any(
                re.search(
                    rf"由[^，,。；;（）()\n]{{1,40}}{re.escape(normalize_search_text(action))}",
                    normalized_sentence,
                )
                for action in actions
            ):
                score += 10.0
            sentence_anchor_hits = sum(
                normalize_search_text(anchor) in normalized_sentence
                for anchor in anchors
                if len(anchor.strip()) >= 2
            )
            score += sentence_anchor_hits * 3.0
            if asks_for_origin:
                if any(
                    marker in normalized_sentence
                    for marker in (
                        "再次", "重新", "恢复", "荒废已久", "后来", "其后",
                        "总行", "总部", "分行", "分部", "前身",
                    )
                ):
                    score -= 12.0
                if any(
                    marker in normalized_sentence
                    for marker in ("首次", "第一次", "第一个", "最早", "开创")
                ):
                    score += 6.0
            score += 3.0 if title_match else 0.0
            score += 2.0 if source.title and normalize_search_text(source.title) in normalize_search_text(sentence) else 0.0
            if prefer_explicit_subject and source.title:
                normalized_title = normalize_search_text(source.title)
                score += 12.0 if normalized_title in normalized_sentence else -3.0
            if any(marker in normalized_sentence for marker in ("计划", "拟", "准备", "预计", "预定")):
                score -= 8.0
            if ">" in sentence or "＞" in sentence:
                score -= 8.0
            if len(sentence) < 20:
                score -= 2.0
            score -= order * 0.01
            candidate = (score, source_index, sentence)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        return None
    _score, source_index, sentence = best
    return f"{sentence.rstrip()} [资料 {source_index}]"


def definition_evidence_answer(question: str, sources: list[SourceItem]) -> str | None:
    normalized_question = question.strip()
    is_introduction = any(
        normalized_question.startswith(prefix)
        for prefix in ("请简要介绍", "请介绍", "简要介绍", "介绍")
    )
    if not is_introduction and not _PURE_DEFINITION_QUESTION.search(normalized_question):
        return None
    for index, source in enumerate(sources, start=1):
        title = source.title.strip()
        normalized_title = normalize_search_text(title)
        if not title or normalized_title not in normalize_search_text(question):
            continue
        text = clean_evidence_text(source.snippet)
        if is_repetitive_garbage(text):
            continue
        all_candidates = [
            match.group(0).strip()
            for match in _SENTENCE.finditer(text[:3000])
            if "Category:" not in match.group(0) and "thumb|" not in match.group(0)
        ]
        candidates = []
        for sentence in all_candidates:
            if normalized_title not in normalize_search_text(sentence):
                continue
            candidates.append(sentence)
        if not candidates:
            candidates = all_candidates
        if not candidates:
            continue
        related = [value for value in candidates if _DEFINITION_RELATION_PATTERN.search(value)]
        # Wiki introductions normally state the identity before secondary facts.
        # Preserve document order instead of preferring a shorter later sentence.
        sentence = (related or candidates)[0]
        if "介绍" in question and len(sentence) < 40:
            try:
                position = all_candidates.index(sentence)
            except ValueError:
                position = -1
            if 0 <= position < len(all_candidates) - 1:
                following = all_candidates[position + 1]
                if len(sentence) + len(following) <= 300:
                    sentence = f"{sentence}{following}"
        return f"{sentence.rstrip()} [资料 {index}]"
    subject_match = re.match(
        r"^(?:请简要介绍|请介绍|简要介绍|介绍)?"
        r"(?P<subject>.+?)(?:是什么|是谁|指的是什么)?[。？?]?$",
        normalized_question,
    )
    subject = subject_match.group("subject").strip() if subject_match else ""
    normalized_subject = normalize_search_text(subject)
    if len(normalized_subject) >= 2:
        for index, source in enumerate(sources, start=1):
            text = clean_evidence_text(source.snippet)
            for match in _SENTENCE.finditer(text[:4000]):
                sentence = match.group(0).strip()
                normalized_sentence = normalize_search_text(sentence)
                if normalized_subject not in normalized_sentence:
                    continue
                if not re.search(
                    rf"{re.escape(normalized_subject)}(?:[（(][^）)\n]{{0,40}}[）)])?"
                    r"(?:是|为|指|属于|由[^。！？\n]{1,40}(?:发展|放大|缩小|改造|研制|设计)而来)",
                    normalized_sentence,
                ):
                    continue
                return f"{sentence.rstrip()} [资料 {index}]"
    return None


def _looks_like_list_question(question: str) -> bool:
    return not is_agent_relation_question(question) and (
        counted_list_size(question) is not None
        or any(marker in question for marker in _LIST_QUESTION_MARKERS)
    )


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
        or any(_BULLET_ROW.match(line) for line in source.snippet.splitlines())
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
    question: str,
    question_tokens: set[str],
    sources: list[SourceItem],
) -> tuple[int, str, str, list[str]] | None:
    groups: dict[str, tuple[float, int, str, list[str]]] = {}
    for source_index, source in enumerate(sources, start=1):
        cleaned = clean_evidence_text(source.snippet)
        for match in _TABLE_FIELD.finditer(cleaned):
            label = match.group("label").strip()
            if not _list_label_matches_question(question, label):
                continue
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
    question: str,
    question_tokens: set[str],
    sources: list[SourceItem],
) -> tuple[int, str, str, list[str]] | None:
    specific_tokens = question_tokens - _GENERIC_LIST_TOKENS
    best: tuple[float, int, str, str, list[str]] | None = None
    for source_index, source in enumerate(sources, start=1):
        cleaned = clean_evidence_text(source.snippet)
        lines = cleaned.splitlines()
        label = _list_label_from_context(lines)
        if not _list_label_matches_question(question, label):
            continue
        context = f"{source.title}\n{lines[0] if lines else ''}\n{label}"
        context_tokens = set(lexical_tokens(context))
        matched_specific = specific_tokens & context_tokens
        if specific_tokens and not matched_specific:
            continue
        names = [
            _clean_bullet_value(match.group("value"))
            for line in lines
            if (match := _BULLET_ROW.match(line))
        ]
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


def _list_label_matches_question(question: str, label: str) -> bool:
    """Require the structure field to describe the type of items being asked for."""

    normalized = normalize_search_text(label).replace(" ", "")
    if any(marker in question for marker in ("站点", "站名", "车站", "哪些站", "有哪些站")):
        return any(marker in normalized for marker in ("站点", "站名", "车站"))
    if "民族" in question:
        return "民族" in normalized
    if any(marker in question for marker in ("朝代", "王朝")):
        return any(marker in normalized for marker in ("朝代", "王朝"))
    if any(marker in question for marker in ("关隘", "关城", "关口")):
        return any(marker in normalized for marker in ("关隘", "关城", "关口"))
    return True


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
    if any(marker in name for marker in ("，", ",", "；", ";", "。")):
        return ""
    if len(name) > 24:
        return ""
    return name


def _clean_bullet_value(raw: str) -> str:
    value = clean_evidence_text(raw).strip().strip(_TRAILING_PUNCTUATION)
    value = re.sub(r"^[\-*•]\s*", "", value).strip()
    if not value or value in {"参考资料", "外部链接", "官方网站"}:
        return ""
    if value.startswith(("（", "(", "→")):
        return ""
    labelled_name = re.match(r"(?P<name>[^：:]{2,80})[：:]", value)
    if labelled_name:
        return labelled_name.group("name").strip()
    leading_name = re.match(r"(?P<name>[^（(：:，,；;]{2,80})(?:[（(]|$)", value)
    if leading_name:
        return leading_name.group("name").strip()
    work = re.search(r"《([^》]{1,60})》", value)
    if work:
        return f"《{work.group(1).strip()}》"
    if len(value) > 180:
        value = value[:180].rstrip("，,；; ")
    return value


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
    pattern = re.compile(
        rf"(?:^|[在至到往经經于於和与與及、，,；;：:\s])"
        rf"(?P<name>[\u3400-\u4dbf\u4e00-\u9fff]{{1,6}}{re.escape(item)})站"
    )
    candidates = [
        re.split(r"在|至|到|往|经|經|于|於|和|与|與|及", match.group("name"))[-1]
        for match in pattern.finditer(context)
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
