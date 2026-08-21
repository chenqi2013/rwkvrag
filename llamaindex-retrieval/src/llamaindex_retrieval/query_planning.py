from dataclasses import dataclass, replace
import re
from typing import Literal

import jieba.posseg as posseg

from .lexical_index import lexical_tokens, normalize_query_text
from .qa_analysis import QuestionAnalysis, analyze_question, clean_question_shell


MergeStrategy = Literal["rank_fusion", "document_interleave"]
ContextPolicy = Literal["none", "lead", "lead_append", "section", "structure"]
AnswerShape = Literal["single_fact", "list", "summary", "narrative"]
SetSemantics = Literal["latest", "all", "partial", "specific"]


@dataclass(frozen=True)
class TaskField:
    field_id: str
    question: str
    relations: tuple[str, ...]


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
    fields: tuple[TaskField, ...] = ()
    answer_shape: AnswerShape = "single_fact"
    set_semantics: SetSemantics = "specific"

    @property
    def query_rewritten(self) -> bool:
        return self.queries != (self.original_question,)


def build_query_plan(question: str) -> QueryPlan:
    normalized = normalize_query_text(clean_question_shell(question))
    analysis = analyze_question(normalized)
    relational = _relational_contract(normalized, analysis)
    if relational is None:
        subject = _subject_for(analysis, normalized)
        relations = _relations_for(analysis.intent, normalized)
    else:
        analysis, subject, relations = relational
    answer_shape = _answer_shape_for(analysis, normalized)
    set_semantics = _set_semantics_for(analysis, normalized)
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
        fields=(TaskField("f1", normalized, relations),),
        answer_shape=answer_shape,
        set_semantics=set_semantics,
    )


def _relational_contract(
    question: str,
    analysis: QuestionAnalysis,
) -> tuple[QuestionAnalysis, str, tuple[str, ...]] | None:
    contextual_agent = re.match(
        r"^(?P<subject>.+?)(?:里|中)(?P<field>[^，。？?]{2,40}?)"
        r"(?:的是|的人是)(?:谁|哪位|什么人)[。？?]?$",
        question,
    )
    if contextual_agent:
        return (
            replace(
                analysis,
                intent="agent",
                entity_type="person",
                subjects=(),
                expects_list=False,
                expects_complete_list=False,
            ),
            _clean_subject(contextual_agent.group("subject")),
            _narrative_relation_variants(contextual_agent.group("field")),
        )
    role = re.match(
        r"^(?P<subject>.+?)的(?P<field>[^的是为，。？?][^的，。？?]{0,19}?)(?:是|为)?"
        r"(?:谁|哪位|什么人)[。？?]?$",
        question,
    )
    if role and analysis.intent not in {"agent", "ordinal"}:
        return (
            replace(
                analysis,
                intent="agent",
                entity_type="person",
                subjects=(),
                expects_list=False,
                expects_complete_list=False,
            ),
            _clean_subject(role.group("subject")),
            (role.group("field").strip(),),
        )
    property_location = re.match(
        r"^(?P<subject>.+?)的(?P<field>[^的，。？?]{1,20}?)"
        r"(?:在哪里|在哪儿|位于哪里|位于哪)[。？?]?$",
        question,
    )
    if property_location:
        return (
            replace(
                analysis,
                intent="location",
                subjects=(),
                expects_list=False,
                expects_complete_list=False,
            ),
            _clean_subject(property_location.group("subject")),
            (property_location.group("field").strip(),),
        )
    property_fact = re.match(
        r"^(?P<subject>.+?)(?:最后)?的(?P<field>[^的，。？?]{1,20}?)"
        r"(?:是什么|是怎样的|如何)[。？?]?$",
        question,
    )
    if property_fact and analysis.intent in {"definition", "fact"}:
        field = property_fact.group("field").strip()
        return (
            replace(
                analysis,
                intent="fact",
                subjects=(),
                expects_list=False,
                expects_complete_list=False,
            ),
            _clean_subject(property_fact.group("subject")),
            _field_relation_variants(field),
        )
    predicate_object = _predicate_object_contract(question, analysis)
    if predicate_object is not None:
        return predicate_object
    return None


def _predicate_object_contract(
    question: str,
    analysis: QuestionAnalysis,
) -> tuple[QuestionAnalysis, str, tuple[str, ...]] | None:
    if analysis.intent != "fact":
        return None
    tagged = [
        (word.strip(), flag)
        for word, flag in posseg.cut(question.strip("。？?！!"))
        if word.strip()
    ]
    interrogative_index = next(
        (
            index
            for index, (word, _) in enumerate(tagged)
            if word in {"哪", "哪些", "什么"} or word.startswith("哪几")
        ),
        None,
    )
    if interrogative_index is None or interrogative_index < 2:
        return None
    prefix = tagged[:interrogative_index]
    while prefix and prefix[-1][0] in {"了", "过", "着", "的"}:
        prefix.pop()
    predicate_index = next(
        (
            index
            for index in range(len(prefix) - 1, 0, -1)
            if (
                prefix[index][1].startswith("v")
                or prefix[index][1] in {"p"}
            )
            and prefix[index][0] not in {"有", "是", "为", "属于"}
        ),
        None,
    )
    if predicate_index is None:
        return None
    subject = _clean_subject("".join(word for word, _ in prefix[:predicate_index]))
    predicate = prefix[predicate_index][0]
    if len(subject) < 2 or not predicate:
        return None
    expanded = _agent_relations(question)
    relations = tuple(dict.fromkeys((predicate, *expanded)))
    expects_list = any(
        marker in question
        for marker in ("哪些", "哪几", "分别", "全部", "所有")
    )
    return (
        replace(
            analysis,
            intent=(
                "list"
                if expects_list
                else analysis.intent if analysis.intent in {"agent", "list"} else "fact"
            ),
            subjects=(),
            expects_list=expects_list,
            expects_complete_list=any(
                marker in question for marker in ("全部", "所有", "完整")
            ),
        ),
        subject,
        relations,
    )


def _field_relation_variants(field: str) -> tuple[str, ...]:
    normalized = field.strip()
    if normalized in {"结局", "結局", "结尾", "結尾", "最终结果", "最終結果"}:
        return (normalized, "终结", "结束", "最终", "统一", "归一统")
    return (normalized,)


def _narrative_relation_variants(field: str) -> tuple[str, ...]:
    normalized = field.strip()
    compact_terms = [
        term
        for term in lexical_tokens(normalized)
        if len(term.replace(" ", "")) >= 3
        and term != normalized
        and term in normalized
    ]
    compact_terms.sort(key=lambda value: (-len(value), -normalized.rfind(value)))
    return tuple(dict.fromkeys((normalized, *compact_terms[:4])))


def _answer_shape_for(analysis: QuestionAnalysis, question: str) -> AnswerShape:
    if analysis.expects_list or re.search(r"哪几(?:家|个|位|种|条|项)", question):
        return "list"
    if any(marker in question for marker in ("故事", "经过", "来龙去脉")):
        return "narrative"
    if analysis.intent in {"cause", "comparison", "procedure"} or any(
        marker in question for marker in ("讲讲", "概括", "总结", "介绍")
    ):
        return "summary"
    return "single_fact"


def _set_semantics_for(analysis: QuestionAnalysis, question: str) -> SetSemantics:
    if any(marker in question for marker in ("现在", "目前", "当前", "现任", "最新")):
        return "latest"
    if any(
        marker in question for marker in ("全部", "所有", "完整", "总共", "一共")
    ):
        return "all"
    if analysis.expects_list or re.search(r"哪几(?:家|个|位|种|条|项)", question):
        return "partial"
    return "specific"


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
        standalone_relations = tuple(
            relation
            for relation in relations[1:]
            if len(relation.replace(" ", "")) >= 4
        )
        candidates = (
            f"{subject} {relations[0]}",
            *standalone_relations,
            *(f"{subject} {relation}" for relation in relations[1:]),
            *analysis.subjects,
            question,
        )
    elif analysis.intent == "list" and analysis.subjects:
        bare_subject = (
            (subject,)
            if len(subject.replace(" ", "")) >= 4 and " " not in subject
            else ()
        )
        candidates = (
            *analysis.subjects,
            *bare_subject,
            *_list_companion_queries(question),
            question,
        )
    elif analysis.intent == "ordinal" and analysis.subjects:
        candidates = (*analysis.subjects, question)
    elif analysis.intent == "cause" and subject and relations:
        candidates = (
            f"{subject}{relations[0]}",
            f"{subject} {' '.join(relations)}",
            question,
            subject,
        )
    elif subject and relations:
        bare_subject = (subject,) if relations == ("简介", "定义") else ()
        candidates = (
            f"{subject} {' '.join(relations)}",
            question,
            *bare_subject,
        )
    elif subject:
        candidates = (subject, question)
    else:
        candidates = (question,)
    return tuple(dict.fromkeys(value.strip() for value in candidates if value.strip()))


def _list_companion_queries(question: str) -> tuple[str, ...]:
    if not any(marker in question for marker in ("哪些", "有哪", "列表", "全部", "所有")):
        return ()
    candidates: list[str] = []
    transit = re.search(
        r"(?P<network>[\u3400-\u9fff]{2,20}?地铁)(?P<line>\d+号线)",
        question,
    )
    if transit:
        network = transit.group("network")
        line = transit.group("line")
        candidates.extend((f"{network}车站列表", f"{network}车站列表 {line}"))
    earliest = re.search(
        r"(?P<place>[\u3400-\u9fff]{2,12})(?:最早开通的地铁线路|地铁首条线路)",
        question,
    )
    if earliest:
        network = f"{earliest.group('place')}地铁"
        candidates.extend((f"{network}车站列表", f"{network} 最早投入服务 车站"))
    endpoints = re.search(
        r"连接(?P<start>[^，。？?和与]{1,20})(?:和|与)"
        r"(?P<end>[^，。？?的]{1,20})的(?P<network>[^，。？?]{2,24}?)(?:线路)?(?:有|经过|途经)",
        question,
    )
    if endpoints:
        network = endpoints.group("network")
        candidates.append(
            f"{network} {endpoints.group('start')} "
            f"{endpoints.group('end')} 车站"
        )
        if network.endswith("地铁"):
            candidates.append(f"{network}车站列表")
    return tuple(dict.fromkeys(candidates))


def _subject_for(analysis: QuestionAnalysis, question: str) -> str:
    if analysis.intent == "comparison":
        return ""
    if analysis.intent == "ordinal" and analysis.subjects:
        return analysis.subjects[0].split(" 第一个 ", 1)[0].strip()
    if analysis.intent == "agent" and analysis.subjects:
        return analysis.subjects[0].rsplit(" ", 1)[0].strip()
    if analysis.intent == "list" and analysis.subjects:
        first = analysis.subjects[0]
        return first.split(" ", 1)[0].removesuffix("列表").strip()
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
        value = re.sub(
            r"(?:走向|走上)?(?:灭亡|衰落|失败|解体)(?:发生|出现|产生)?$",
            "",
            value,
        ).strip(" 的")
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
        relation_match = re.search(
            r"(?:有哪些|有哪(?:些|几个)?|包括哪些|包含哪些|分别是哪些|列出)"
            r"(?P<relation>[^？?。]+)",
            question,
        )
        if relation_match:
            relation = relation_match.group("relation").strip(" 的，,；;")
            relation_core = re.sub(
                r"^(?:伟大|主要|重要|著名|知名|典型|全部|所有|具体|相关)(?:的)?",
                "",
                relation,
            ).strip(" 的")
            return tuple(dict.fromkeys(
                value for value in (relation, relation_core) if len(value) >= 2
            ))
    return ()


def _agent_relations(question: str) -> tuple[str, ...]:
    if any(marker in question for marker in ("现在", "目前", "当前", "如今", "现任")):
        return ("现任", "目前", "当前")
    reverse_match = re.search(
        r"(?:的|背后的)?(创始人|创办人|创办者|建立者|创建者|发明者|发现者|"
        r"创作者|设计者|建造者|执导者|负责人|作者|导演|主演)(?:是|为)?"
        r"(?:谁|哪位|什么人)[。？?]?$",
        question,
    )
    if reverse_match:
        relation = reverse_match.group(1)
    else:
        match = re.search(
        r"(发起|提出|创立|创建|建立|发明|发现|开启|开辟|开通|领导|指挥|主演|导演|"
        r"撰写|创作者|创作|设计者|设计|建造|开发|制作|主持|组织|推动|负责|得主|获得者|创办者|执导者|执导|"
        r"创办|創作|設計|執導|創辦)",
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
        "创办": ("创办", "创立", "创建", "成立", "创始人", "联合创始人"),
        "创作者": ("创作", "作者", "创作者"),
        "创始人": ("创始人", "创办人", "创办者", "创立", "创建", "创办"),
        "创办人": ("创办人", "创始人", "创办者", "创立", "创建", "创办"),
        "设计者": ("设计", "设计者"),
        "创办者": ("创办", "创建", "创办者"),
        "建立者": ("建立者", "建立", "创立", "创建"),
        "创建者": ("创建者", "创建", "创立", "建立"),
        "发明者": ("发明者", "发明", "研制", "创造"),
        "发现者": ("发现者", "发现", "首次发现"),
        "建造者": ("建造者", "建造", "修建", "建设"),
        "负责人": ("负责人", "负责", "领导", "主持"),
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
