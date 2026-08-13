from dataclasses import dataclass
import re
from typing import Literal

from .lexical_index import normalize_query_text
from .qa_analysis import QuestionAnalysis, analyze_question, clean_question_shell


MergeStrategy = Literal["rank_fusion", "document_interleave"]
ContextPolicy = Literal["none", "lead", "lead_append", "section", "structure"]


@dataclass(frozen=True)
class QueryPlan:
    original_question: str
    normalized_question: str
    analysis: QuestionAnalysis
    queries: tuple[str, ...]
    subject: str
    relations: tuple[str, ...]
    merge_strategy: MergeStrategy
    context_policy: ContextPolicy

    @property
    def query_rewritten(self) -> bool:
        return self.queries != (self.original_question,)


def build_query_plan(question: str) -> QueryPlan:
    normalized = normalize_query_text(clean_question_shell(question))
    analysis = analyze_question(normalized)
    subject = _subject_for(analysis, normalized)
    relations = _relations_for(analysis.intent, normalized)
    queries = _queries_for(analysis, normalized, subject, relations)
    merge_strategy: MergeStrategy = (
        "document_interleave"
        if analysis.intent in {"comparison", "time"}
        else "rank_fusion"
    )
    return QueryPlan(
        original_question=question,
        normalized_question=normalized,
        analysis=analysis,
        queries=queries,
        subject=subject,
        relations=relations,
        merge_strategy=merge_strategy,
        context_policy=_context_policy(analysis.intent),
    )


def _queries_for(
    analysis: QuestionAnalysis,
    question: str,
    subject: str,
    relations: tuple[str, ...],
) -> tuple[str, ...]:
    if analysis.intent == "comparison" and len(analysis.subjects) == 2:
        candidates = analysis.subjects
    elif analysis.intent == "time" and analysis.subjects:
        candidates = (*analysis.subjects, question)
    elif analysis.intent == "agent" and subject:
        candidates = (
            *(f"{subject} {relation}" for relation in relations),
            *analysis.subjects,
            question,
        )
    elif analysis.intent in {"list", "ordinal"} and analysis.subjects:
        candidates = (*analysis.subjects, question)
    elif analysis.intent == "cause" and subject and relations:
        candidates = (
            f"{subject}{relations[0]}",
            f"{subject} {' '.join(relations)}",
            question,
            subject,
        )
    elif subject and relations:
        candidates = (
            f"{subject} {' '.join(relations)}",
            question,
            subject,
        )
    elif subject:
        candidates = (subject, question)
    else:
        candidates = (question,)
    return tuple(dict.fromkeys(value.strip() for value in candidates if value.strip()))


def _subject_for(analysis: QuestionAnalysis, question: str) -> str:
    if analysis.intent == "comparison":
        return ""
    if analysis.intent == "ordinal" and analysis.subjects:
        return analysis.subjects[0].split(" 第一个 ", 1)[0].strip()
    if analysis.intent == "agent" and analysis.subjects:
        return analysis.subjects[0].rsplit(" ", 1)[0].strip()
    if analysis.intent == "list" and analysis.subjects:
        first = analysis.subjects[0]
        return first.split(" ", 1)[0].strip()
    quantitative = re.match(
        r"^(?P<subject>.+?)(?:的)?(?:人口|面积|面積|全长|全長|长度|長度|高度|海拔)"
        r"(?:数据|數據)?(?:(?:是|为|有)?(?:多少|什么|具体数字)|[。？?]|$)",
        question,
    )
    if quantitative:
        return _clean_subject(quantitative.group("subject"))
    patterns = {
        "definition": r"^(?:请简要介绍|请介绍|简要介绍|介绍)?(?P<subject>.+?)(?:是什么|是谁|指的是什么)[。？?]?$",
        "cause": r"^(?:导致)?(?P<subject>.+?)(?:是)?(?:因为什么原因|为什么|为何|的原因|是哪些因素造成)",
        "procedure": r"^(?P<left>.+?)(?:如何|怎么)(?P<right>.+?)[。？?]?$",
        "time": r"^(?P<subject>.+?)(?:是什么时候|是在什么时候|是哪一年|在哪一年|什么时候|哪一年|何时)",
        "location": r"^(?P<subject>.+?)(?:位于哪里|位于哪|在哪里|在哪儿|在哪个球场|在哪座球场|在哪个场馆)",
        "birthplace": r"^(?P<subject>.+?)(?:出生于哪里|出生在哪里|哪里出生)",
    }
    pattern = patterns.get(analysis.intent)
    if not pattern:
        if analysis.intent == "definition":
            for prefix in ("请简要介绍", "请介绍", "简要介绍", "介绍"):
                if question.startswith(prefix):
                    return question[len(prefix):].strip(" 的是请，,。；;？?")
        return ""
    match = re.search(pattern, question.strip())
    if not match:
        if analysis.intent == "definition":
            for prefix in ("请简要介绍", "请介绍", "简要介绍", "介绍"):
                if question.startswith(prefix):
                    return _clean_subject(question[len(prefix):])
        if analysis.intent == "cause":
            colloquial = re.match(
                r"^(?P<subject>.+?)(?:究竟)?(?:是)?怎么(?:一步步)?(?:走向)?"
                r"(?:灭亡|衰落|失败|解体)",
                question,
            )
            if colloquial:
                return colloquial.group("subject").strip(" 的")
            prefix = question.split("原因", 1)[0].rstrip(" 的主要根本直接")
            event_match = re.match(
                r"^(?P<subject>.+?)(?:走向|走上|发生|出现|产生|形成|爆发|灭亡|失败|成功|衰落|崩溃)",
                prefix,
            )
            return (
                event_match.group("subject").strip(" 的")
                if event_match
                else prefix.strip(" 的")
            )
        if analysis.intent == "procedure":
            event_match = re.match(
                r"^(?P<subject>.+?)(?:究竟)?(?:是)?怎么(?:一步步)?(?:走向)?"
                r"(?:灭亡|衰落|失败|解体)",
                question,
            )
            if event_match:
                return event_match.group("subject").strip(" 的")
        return ""
    if analysis.intent == "procedure":
        value = f"{match.group('left')} {match.group('right')}"
    else:
        value = match.group("subject")
    if analysis.intent == "cause":
        value = re.sub(r"(?:走向|走上)?(?:灭亡|衰落|失败|解体)$", "", value).strip(" 的")
    return _clean_subject(value)


def _relations_for(intent: str, question: str) -> tuple[str, ...]:
    if intent == "cause":
        event_match = re.search(
            r"(灭亡|覆亡|衰亡|衰落|崩溃|失败|成功|爆发|形成|发生|出现|产生)",
            question,
        )
        event = event_match.group(1) if event_match else ""
        return tuple(dict.fromkeys(value for value in (event, "原因", "导致") if value))
    if intent == "procedure":
        return ("方法", "过程", "步骤")
    if intent == "time":
        time_relations = {
            "成立": ("成立", "创立", "创建", "建立", "组建"),
            "创建": ("创建", "创立", "成立", "建立"),
            "出生": ("出生", "生于"),
            "逝世": ("逝世", "去世", "病逝", "卒于"),
            "开通": ("开通", "启用", "通车", "运营"),
            "上映": ("上映", "首映", "发行"),
            "回归": ("回归", "政权移交", "主权移交", "恢复行使主权"),
        }
        for marker, values in time_relations.items():
            if marker in question:
                return values
        return ("时间", "年份")
    if intent == "location":
        return ("位于", "地点")
    if intent == "birthplace":
        return ("出生于", "出生地")
    if intent == "agent":
        return _agent_relations(question)
    if intent == "definition":
        return ("简介", "定义")
    if intent == "list":
        relation_groups = {
            "主办": ("主办", "举办", "承办", "主办国"),
            "举办": ("举办", "主办", "承办", "举办国"),
            "承办": ("承办", "主办", "举办", "承办国"),
            "获得": ("获得", "获奖", "授予", "得主"),
            "参加": ("参加", "参赛", "参与"),
            "组成": ("组成", "包括", "成员"),
        }
        for marker, values in relation_groups.items():
            if marker in question:
                return values
    return ()


def _agent_relations(question: str) -> tuple[str, ...]:
    if any(marker in question for marker in ("现在", "目前", "当前", "如今", "现任")):
        return ("现任", "目前", "当前")
    reverse_match = re.search(
        r"(?:的|背后的)(创作者|创办者|设计者|执导者|作者|导演|主演)(?:是|为)?"
        r"(?:谁|哪位|什么人)[。？?]?$",
        question,
    )
    if reverse_match:
        relation = reverse_match.group(1)
    else:
        match = re.search(
        r"(发起|提出|创立|创建|建立|发明|发现|开启|开辟|开通|领导|指挥|主演|导演|"
        r"撰写|创作者|创作|设计者|设计|建造|开发|制作|主持|组织|推动|负责|得主|获得者|创办者|执导者|执导|"
        r"創作|設計|執導|創辦)",
        question,
        )
        if not match:
            return ()
        relation = match.group(1)
    equivalents = {
        "开启": ("开启", "开辟", "出使", "凿空"),
        "开辟": ("开辟", "开启", "出使", "凿空"),
        "开通": ("开通", "开启", "开辟", "通达"),
        "得主": ("得主", "获得", "获奖", "授予"),
        "获得者": ("获得者", "得主", "获得", "获奖"),
        "創作": ("创作", "創作"),
        "設計": ("设计", "設計"),
        "執導": ("执导", "导演", "執導", "導演"),
        "执导": ("执导", "导演"),
        "創辦": ("创办", "创建", "創辦", "創建"),
        "创作者": ("创作", "作者", "创作者"),
        "设计者": ("设计", "设计者"),
        "创办者": ("创办", "创建", "创办者"),
        "执导者": ("执导", "导演", "执导者"),
    }
    return equivalents.get(relation, (relation,))


def _clean_subject(value: str) -> str:
    cleaned = value.strip(" 的是请，,。；;？?")
    cleaned = re.sub(r"^(?:现在|目前|当前|如今)", "", cleaned)
    cleaned = re.sub(r"(?:究竟|到底)$", "", cleaned)
    cleaned = re.sub(r"^(?:中国历史上|历史上)", "", cleaned)
    cleaned = re.sub(r"^(?:请概括|请说明|概括)", "", cleaned)
    return cleaned.strip(" 的是请，,。；;？?")


def _context_policy(intent: str) -> ContextPolicy:
    if intent == "list":
        return "structure"
    if intent in {"cause", "procedure"}:
        return "section"
    if intent in {"definition", "comparison"}:
        return "lead"
    if intent in {"time", "agent", "location", "birthplace"}:
        return "lead_append"
    return "none"
