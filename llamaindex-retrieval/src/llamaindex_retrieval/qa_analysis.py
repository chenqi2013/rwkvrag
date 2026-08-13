import re
from dataclasses import dataclass

from .lexical_index import normalize_search_text
from .question_patterns import is_agent_relation_question
from .schemas import SourceItem


_COMPARISON_PATTERNS = (
    re.compile(r"^比较\s*(?P<left>[^，。？?]{2,40}?)(?:和|与|跟|及)(?P<right>[^，。？?]{2,40})[。？?]?$"),
    re.compile(r"^(?P<left>[^，。？?]{2,40}?)(?:和|与|跟|及)(?P<right>[^，。？?]{2,40}?)(?:有什么区别|有何区别|的区别|区别是什么|相比如何)[。？?]?$"),
)
_NUMBER = re.compile(r"\d+(?:\.\d+)?%?")
_CITATION = re.compile(r"\[资料\s*([1-9]\d*)\]")
_SENTENCE = re.compile(r"[^。！？!?\n]+[。！？!?]?")
_PERSON_HINTS = (
    "谁",
    "哪几个人",
    "哪些人",
    "哪几位",
    "什么人",
    "人物",
    "出生",
    "逝世",
    "生平",
    "职业",
    "作家",
    "演员",
)
_PLACE_HINTS = ("哪里", "位于", "哪个国家", "哪个省", "城市", "地区", "县", "乡")
_WORK_HINTS = ("电影", "电视剧", "歌曲", "小说", "作品", "游戏", "出版", "上映")
_ORG_HINTS = ("公司", "组织", "协会", "基金会", "大学", "机构", "成立")
_PURE_DEFINITION_PATTERN = re.compile(r"(?:是什么|是谁|指的是什么)[。？?]?$")
_PARTIAL_LIST_MARKERS = ("例如", "举例", "一些", "部分", "著名", "主要", "典型")
_TIME_QUESTION_PATTERN = re.compile(
    r"^(?P<subject>.+?)(?:是什么时候|是在什么时候|是哪一年|在哪一年|什么时候|哪一年|何时)"
    r"(?P<event>.*?)(?:的)?[。？?]?$"
)
_LIST_RELATION_MARKER = re.compile(r"有哪些|有哪(?:些|几个|几种)|哪些|哪几个|哪几种")
_LIST_SCOPE_SUFFIXES = (
    "经历了",
    "经历",
    "包括了",
    "包括",
    "包含了",
    "包含",
    "拥有",
    "总共",
    "一共",
    "从古至今",
    "自古至今",
    "迄今为止",
    "至今",
)
_REVERSE_AGENT_QUERY = re.compile(
    r"^(?P<subject>.+?)(?:的|背后的)?(?P<relation>開國皇帝|創始人|創辦人|建立者|創建者|發明者|發現者|創作者|設計者|建造者|執導者|开国皇帝|创始人|创办人|创办者|建立者|创建者|发明者|发现者|创作者|设计者|建造者|执导者|负责人|得主|获得者|作者|导演|主演)"
    r"(?:是|为)?(?:谁|哪几个人|哪些人|哪几位|哪位|什么人)[。？?]?$"
)
_CURRENT_OFFICE_QUERY = re.compile(
    r"^(?:现在|目前|当前|如今|现任)?(?P<office>.+?(?:副总统|总统|总理|首相|主席))"
    r"(?:是|为)?(?:谁|哪位|什么人)[。？?]?$"
)
_ACTION_AGENT_QUERY = re.compile(
    r"^(?P<prefix>.+?)(?P<relation>发起|提出|创立|创建|建立|发明|发现|开启|开辟|开通|"
    r"领导|指挥|主演|导演|执导|撰写|创作|设计|建造|开发|制作|主持|组织|推动|负责)"
    r"(?:了)?(?P<object>[^，。？?]{1,48}?)(?:的是|者是|的人是)?"
    r"(?:谁|哪位|什么人)[。？?]?$"
)
_FORWARD_ACTION_AGENT_QUERY = re.compile(
    r"^(?P<prefix>.*?)(?:谁|哪位|什么人)"
    r"(?P<relation>发起|提出|创立|创建|建立|发明|发现|开启|开辟|开通|"
    r"领导|指挥|主演|导演|执导|撰写|创作|设计|建造|开发|制作|主持|组织|推动|负责)"
    r"(?:了)?(?P<object>[^，。？?]{1,48})"
)
_REVERSE_ACTION_OBJECT_QUERY = re.compile(
    r"^(?P<subject>[^，。？?]{2,40}?)(?P<relation>发起|提出|创立|创建|建立|发明|发现|"
    r"开启|开辟|开通|撰写|创作|设计|建造|开发|制作)"
    r"(?:了|过)?(?:什么|哪些(?:东西|事物|作品)?|哪(?:个|些)(?:东西|事物|作品)?)[。？?]?$"
)
_ORDINAL_RELATION_QUERY = re.compile(
    r"^(?P<scope>.+?)(?:历史上|史上)?(?:的)?"
    r"(?P<ordinal>第一个|第一位|首位|最早的?)"
    r"(?P<relation>[^，。？?]{1,32}?)"
    r"(?:(?:是|为)?(?:谁|哪位|什么人))?[。？?]?$"
)
_REVERSE_ORDINAL_RELATION_QUERY = re.compile(
    r"^(?:谁|哪位|什么人)(?:是|为)"
    r"(?P<scope>.+?)(?:历史上|史上)?(?:的)?"
    r"(?P<ordinal>第一个|第一位|首位|最早的?)"
    r"(?P<relation>[^，。？?]{1,32})[。？?]?$"
)


@dataclass(frozen=True)
class QuestionAnalysis:
    intent: str
    entity_type: str
    subjects: tuple[str, ...]
    expects_list: bool
    expects_complete_list: bool


@dataclass(frozen=True)
class GroundingValidation:
    answer: str
    valid: bool
    issues: tuple[str, ...]
    unsupported_numbers: tuple[str, ...]


@dataclass(frozen=True)
class ListValidation:
    complete: bool | None
    expected_count: int | None
    answer_count: int | None
    issues: tuple[str, ...]


def comparison_subjects(question: str) -> tuple[str, str] | None:
    normalized = question.strip()
    for pattern in _COMPARISON_PATTERNS:
        match = pattern.match(normalized)
        if match:
            left = match.group("left").strip("《》“”\"' ")
            right = match.group("right").strip("《》“”\"' ")
            if left and right and left != right:
                return left, right
    return None


def analyze_question(question: str) -> QuestionAnalysis:
    question = clean_question_shell(question)
    comparison = comparison_subjects(question)
    ordinal_queries = _ordinal_search_queries(question)
    agent_question = bool(_REVERSE_ACTION_OBJECT_QUERY.match(question.strip())) or is_agent_relation_question(question)
    expects_list = not agent_question and any(
        marker in question
        for marker in ("哪些", "有哪些", "列出", "列举", "列一下", "全部", "所有", "分别")
    )
    expects_complete = expects_list and not any(marker in question for marker in _PARTIAL_LIST_MARKERS)
    if comparison:
        intent = "comparison"
        subjects = comparison
    elif ordinal_queries:
        intent = "ordinal"
        subjects = ordinal_queries
    elif agent_question:
        intent = "agent"
        subjects = _agent_search_queries(question)
    elif any(marker in question for marker in ("为什么", "原因", "为何", "哪些因素造成", "怎么一步步")):
        intent = "cause"
        subjects = ()
    elif any(marker in question for marker in ("什么时候", "哪一年", "何时")):
        intent = "time"
        subjects = _time_search_queries(question)
    elif any(marker in question for marker in ("出生于哪里", "出生在哪里", "哪里出生")):
        intent = "birthplace"
        subjects = ()
    elif any(marker in question for marker in (
        "位于哪里", "位于哪", "在哪里", "在哪儿", "在哪个球场", "在哪座球场", "在哪个场馆",
    )):
        intent = "location"
        subjects = ()
    elif expects_list:
        intent = "list"
        subjects = _list_search_queries(question)
    elif any(marker in question for marker in ("如何", "怎么", "过程", "步骤")):
        intent = "procedure"
        subjects = ()
    elif _is_pure_definition_question(question):
        intent = "definition"
        subjects = ()
    else:
        intent = "fact"
        subjects = ()

    if agent_question or any(marker in question for marker in _PERSON_HINTS):
        entity_type = "person"
    elif any(marker in question for marker in _PLACE_HINTS):
        entity_type = "place"
    elif any(marker in question for marker in _WORK_HINTS):
        entity_type = "work"
    elif any(marker in question for marker in _ORG_HINTS):
        entity_type = "organization"
    else:
        entity_type = "unknown"
    return QuestionAnalysis(intent, entity_type, tuple(subjects), expects_list, expects_complete)


def _is_pure_definition_question(question: str) -> bool:
    normalized = question.strip()
    if any(normalized.startswith(prefix) for prefix in ("请简要介绍", "请介绍", "简要介绍", "介绍")):
        return True
    if any(marker in normalized for marker in ("是干什么的", "指的是什么", "用两三句话说明", "通俗说说")):
        return True
    return bool(_PURE_DEFINITION_PATTERN.search(normalized))


def clean_question_shell(question: str) -> str:
    value = question.strip()
    for prefix in ("请问", "你知道", "我想知道", "我不太了解", "想去", "说下", "从资料看，", "从资料看"):
        if value.startswith(prefix):
            value = value[len(prefix):].strip()
    value = re.sub(r"^(?:请)?(?:简要)?介绍一下", "请简要介绍", value)
    value = re.sub(r"^(?:请)?介绍下(?:一下)?", "请简要介绍", value)
    value = re.sub(r"^能通俗说说", "请简要介绍", value)
    value = re.sub(r"^关于(.+?)[，,]用两三句话说明一下[。.]?$", r"请简要介绍\1", value)
    value = re.sub(r"^(.+?)[，,](?:它|他|她)指的是什么[？?]?$", r"\1是什么？", value)
    value = re.sub(r"^(.+?)(?:是干什么的|是做什么的)吗?[？?]?$", r"\1是什么？", value)
    value = re.sub(r"^(.+?)[，,](?:它|他|她)的位置在哪里[？?]?$", r"\1位于哪里？", value)
    value = re.sub(r"^(.+?)具体坐落在哪儿[？?]?$", r"\1位于哪里？", value)
    value = re.sub(r"^(.+?)属于哪里吗?[？?]?$", r"\1位于哪里？", value)
    value = re.sub(r"^(.+?)背后的(.+?者)[。.]?$", r"\1的\2是谁？", value)
    value = re.sub(
        r"^(.+?)由谁(发起|提出|创立|创建|建立|发明|发现|开启|开辟|开通|领导|指挥|主演|导演|撰写|创作|设计|建造|开发|制作|主持|组织|推动|负责|執導|設計|創作)吗?[？?]?$",
        r"谁\2\1？",
        value,
    )
    value = re.sub(
        r"^(.+?)(成立|创建|出生|逝世|开通|上映)的具体时间是什么[？?]?$",
        r"\1是什么时候\2的？",
        value,
    )
    value = re.sub(r"^请列出(.+?)中关于(.+?)的内容[。.]?$", r"\1有哪些\2？", value)
    value = re.sub(r"^(.+?)在(.+?)方面都包括什么[？?]?$", r"\1有哪些\2？", value)
    value = re.sub(r"^能整理一下(.+?)的(.+?)列表吗[？?]?$", r"\1有哪些\2？", value)
    value = re.sub(r"^请概括(.+?)(灭亡|衰落|失败|解体)的主要原因和过程[。.]?$", r"\1\2的原因是什么？", value)
    value = re.sub(r"^(.+?)(成立|创建|出生|逝世|开通|上映)的年份[。.]?$", r"\1是哪一年\2的？", value)
    value = re.sub(
        r"^(.+?)(人口|面积|面積|全长|全長|长度|長度|高度|海拔)的具体数字[。.]?$",
        r"\1的\2是多少？",
        value,
    )
    value = re.sub(r"^(.+?)在什么地方[？?]?$", r"\1位于哪里？", value)
    if value.endswith(("吗？", "吗?")):
        value = value[:-2] + "？"
    return value


def _time_search_queries(question: str) -> tuple[str, ...]:
    match = _TIME_QUESTION_PATTERN.match(clean_question_shell(question).strip())
    if not match:
        return ()
    subject = re.sub(r"(?:分别|各自)$", "", match.group("subject")).strip(" ，,。？?")
    event = match.group("event").strip(" 的，,。？?")
    if not subject or not event:
        return ()
    coordinated = [value.strip(" 的") for value in re.split(r"和|与|跟|及", subject) if value.strip(" 的")]
    if len(coordinated) >= 2:
        return tuple(dict.fromkeys(
            value
            for item in coordinated
            for value in (item, f"{item}{event}")
        ))
    return (subject, f"{subject}{event}")


def _list_search_queries(question: str) -> tuple[str, ...]:
    normalized = clean_question_shell(question).strip(" ？?，,。；;")
    match = _LIST_RELATION_MARKER.search(normalized)
    if not match:
        return ()
    left = normalized[:match.start()].rstrip(" 的")
    right = normalized[match.end():].strip(" 的")
    left = re.sub(r"(?:由|有)$", "", left).rstrip(" 的")
    changed = True
    while changed and left:
        changed = False
        for suffix in _LIST_SCOPE_SUFFIXES:
            if left.endswith(suffix):
                left = left[:-len(suffix)].rstrip(" 的")
                changed = True
                break
    if left and not right and "的" in left:
        left, right = left.rsplit("的", 1)
        left = left.strip(" 的")
        right = right.strip(" 的")
    if not left or not right:
        return ()
    right_core = re.sub(
        r"^(?:比较|较为|最为|最|很|非常)?(?:著名|主要|重要|典型|常见|知名|全部|所有)(?:的)?",
        "",
        right,
    ).strip(" 的")
    if right_core != right and len(right_core) >= 4:
        return tuple(dict.fromkeys((
            f"{right_core}列表",
            right_core,
            f"{left} {right_core}",
            f"{left} {right}",
        )))
    relation = f"{left} {right}"
    return (f"{relation}列表", relation)


def _agent_search_queries(question: str) -> tuple[str, ...]:
    normalized = clean_question_shell(question).strip()
    reverse_object_match = _REVERSE_ACTION_OBJECT_QUERY.match(normalized)
    if reverse_object_match:
        subject = reverse_object_match.group("subject").strip(" 的")
        relation = reverse_object_match.group("relation")
        if subject:
            return (f"{subject} {relation}", subject)
    office_match = _CURRENT_OFFICE_QUERY.match(normalized)
    if office_match:
        office = office_match.group("office").strip(" 的")
        return (f"{office} 现任", office)
    match = _REVERSE_AGENT_QUERY.match(normalized)
    if match:
        subject = match.group("subject").strip(" 的")
        relation = match.group("relation")
        if subject:
            return (f"{subject} {relation}",)
    action_match = _ACTION_AGENT_QUERY.match(normalized)
    if not action_match:
        action_match = _FORWARD_ACTION_AGENT_QUERY.match(normalized)
    if not action_match:
        return ()
    relation = action_match.group("relation")
    object_name = action_match.group("object").strip(" 的了")
    if "的" in object_name:
        nearest_object = object_name.rsplit("的", 1)[-1].strip()
        if 2 <= len(nearest_object) <= 32:
            object_name = nearest_object
    if not object_name:
        return ()
    return (f"{object_name} {relation}", object_name)


def _ordinal_search_queries(question: str) -> tuple[str, ...]:
    match = _REVERSE_ORDINAL_RELATION_QUERY.match(question.strip())
    if not match:
        match = _ORDINAL_RELATION_QUERY.match(question.strip())
    if not match:
        return ()
    scope = re.sub(r"(?:历史上|史上)$", "", match.group("scope")).strip(" 的")
    relation = match.group("relation").strip(" 的")
    if not scope or not relation:
        return ()
    return tuple(dict.fromkeys((
        f"{scope} 第一个 {relation}",
        f"{scope} 第一位 {relation}",
        f"{scope} 首位 {relation}",
    )))


def ambiguity_candidates(question: str, sources: list[SourceItem]) -> list[str]:
    if not sources:
        return []
    subject = _definition_subject(question)
    if not subject:
        return []
    normalized_subject = normalize_search_text(subject)
    candidates = []
    for source in sources:
        title = source.title.strip()
        normalized_title = normalize_search_text(title).replace(" ", "")
        normalized_subject_compact = normalized_subject.replace(" ", "")
        if normalized_title == normalized_subject_compact or normalized_title.startswith(f"{normalized_subject_compact}(") or normalized_title.startswith(f"{normalized_subject_compact}（"):
            if title not in candidates:
                candidates.append(title)
    has_bare = any(normalize_search_text(value).replace(" ", "") == normalized_subject.replace(" ", "") for value in candidates)
    qualified = [value for value in candidates if normalize_search_text(value).replace(" ", "") != normalized_subject.replace(" ", "")]
    if not has_bare and any("消歧义" in value or "消歧義" in value for value in candidates):
        return candidates
    if not has_bare and len(qualified) >= 2:
        return candidates
    return []


def validate_grounding(answer: str, sources: list[SourceItem]) -> GroundingValidation:
    evidence = "\n".join(source.snippet for source in sources)
    evidence_numbers = set(_NUMBER.findall(evidence))
    answer_without_citations = _CITATION.sub("", answer)
    unsupported = tuple(sorted(set(_NUMBER.findall(answer_without_citations)) - evidence_numbers))
    issues = []
    citations = [int(value) for value in _CITATION.findall(answer)]
    if answer and not citations:
        issues.append("missing_citation")
    if any(value > len(sources) for value in citations):
        issues.append("invalid_citation")
    if unsupported:
        issues.append("unsupported_number")
    return GroundingValidation(answer, not issues, tuple(issues), unsupported)


def remove_unsupported_number_sentences(answer: str, unsupported: tuple[str, ...]) -> str:
    if not unsupported:
        return answer
    kept = [
        match.group(0).strip()
        for match in _SENTENCE.finditer(answer)
        if not any(value in _CITATION.sub("", match.group(0)) for value in unsupported)
    ]
    return "".join(kept).strip()


def validate_list_answer(question: str, answer: str, sources: list[SourceItem]) -> ListValidation:
    analysis = analyze_question(question)
    if analysis.intent != "list" or not analysis.expects_complete_list:
        return ListValidation(None, None, None, ())
    evidence = "\n".join(source.snippet for source in sources)
    counts = [
        int(value)
        for value in re.findall(r"(?:共|共有|总计|總計)\s*(?:设|設|有)?\s*(\d{1,3})\s*(?:个|個|名|项|項|座|站)", evidence)
    ]
    expected_count = max(counts) if counts else None
    answer_body = _CITATION.sub("", answer)
    list_body = answer_body.split("：", 1)[-1]
    items = [value.strip(" 。；;，,") for value in re.split(r"[、；;\n]", list_body) if value.strip(" 。；;，,")]
    answer_count = len(items) if len(items) >= 2 else None
    issues = []
    if re.search(r"(?:^|[、，,；;：:\s])等(?:[。；;，,\s]|$)", answer_body) or any(
        marker in answer_body for marker in ("省略", "部分", "其余", "其餘")
    ):
        issues.append("explicitly_incomplete")
    if expected_count is not None and (answer_count is None or answer_count < expected_count):
        issues.append("count_mismatch")
    return ListValidation(not issues, expected_count, answer_count, tuple(issues))


def _definition_subject(question: str) -> str:
    value = question.strip()
    for prefix in ("请简要介绍", "请介绍", "简要介绍"):
        if value.startswith(prefix):
            value = value[len(prefix):]
    for marker in ("指的是什么", "是什么", "是谁"):
        if marker in value:
            value = value.split(marker, 1)[0]
            break
    return value.strip("《》“”\"' ？?，,。；;")
