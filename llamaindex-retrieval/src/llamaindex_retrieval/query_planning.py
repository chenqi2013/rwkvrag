from dataclasses import dataclass, replace
import re
from typing import Literal

import jieba.posseg as posseg

from .lexical_index import lexical_tokens, normalize_query_text, normalize_search_text
from .qa_analysis import (
    QuestionAnalysis,
    analyze_question,
    clean_subject_scope,
    clean_question_shell,
    counted_list_size,
)


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
    if analysis.intent == "cause":
        event_relation = _cause_event_relation(normalized, subject)
        if event_relation:
            relations = tuple(dict.fromkeys((event_relation, *relations)))
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
    role_attribution = _role_attribution_contract(question, analysis)
    if role_attribution is not None:
        return role_attribution
    terminal_agent = re.match(
        r"^(?P<subject>.+?)(?:的由来[，,]?)?(?:是)?(?:为了|為了)?"
        r"(?P<action>纪念|紀念|杀害|殺害|害死|处死|處死|发起|發起|创立|創立|创建|創建|建立|发明|發明|发现|發現)"
        r"(?:的是|的人是)?(?:谁|誰|哪位|什么人)[。？?]?$",
        question,
    )
    if terminal_agent:
        action = normalize_search_text(terminal_agent.group("action"))
        return (
            replace(
                analysis,
                intent="agent",
                entity_type="person",
                subjects=(),
                expects_list=False,
                expects_complete_list=False,
            ),
            _clean_subject(terminal_agent.group("subject")),
            _agent_relation_equivalents(action),
        )
    passive_agent = re.match(
        r"^(?P<subject>.+?)(?:是)?被(?:谁|誰|哪位|什么人)"
        r"(?P<action>杀害|殺害|害死|处死|處死|发起|發起|创建|創建|建立)的?[。？?]?$",
        question,
    )
    if passive_agent:
        action = normalize_search_text(passive_agent.group("action"))
        return (
            replace(
                analysis,
                intent="agent",
                entity_type="person",
                subjects=(),
                expects_list=False,
                expects_complete_list=False,
            ),
            _clean_subject(passive_agent.group("subject")).removesuffix("最后"),
            _agent_relation_equivalents(action),
        )
    reverse_action_agent = re.match(
        r"^(?P<subject>.+?)(?:是|由)?(?:谁|哪位|什么人)"
        r"(?P<action>发起|發起|提出|创立|創立|创建|創建|建立|导演|導演|负责|負責)的?[。？?]?$",
        question,
    )
    if reverse_action_agent:
        action = normalize_search_text(reverse_action_agent.group("action"))
        return (
            replace(
                analysis,
                intent="agent",
                entity_type="person",
                subjects=(),
                expects_list=False,
                expects_complete_list=False,
            ),
            _clean_subject(reverse_action_agent.group("subject")),
            _agent_relation_equivalents(action),
        )
    action_attribution = re.match(
        r"^(?P<subject>.+?)(?:是|由)?(?:谁|哪位|什么人)"
        r"(?P<action>写|拍|创作|设计|发明|发现|建立|创建|开发|制作)的[。？?]?$",
        question,
    )
    if action_attribution:
        action = action_attribution.group("action")
        relation_variants = {
            "写": ("作者", "撰写", "创作", "著"),
            "拍": ("导演", "执导", "拍摄"),
        }
        return (
            replace(
                analysis,
                intent="agent",
                entity_type="person",
                subjects=(),
                expects_list=False,
                expects_complete_list=False,
            ),
            _clean_subject(action_attribution.group("subject")),
            relation_variants.get(action, (action,)),
        )
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
    compact_role = re.match(
        r"^(?P<subject>.+?)(?P<field>创始人|创办人|创办者|建立者|创建者|负责人|老板|首席执行官|CEO|作者|导演|编剧|设计师)"
        r"(?:是|为)?(?:谁|哪位|什么人)[。？?]?$",
        question,
    )
    if compact_role and analysis.intent not in {"agent", "ordinal"}:
        return (
            replace(
                analysis,
                intent="agent",
                entity_type="person",
                subjects=(),
                expects_list=False,
                expects_complete_list=False,
            ),
            _clean_subject(compact_role.group("subject")),
            (compact_role.group("field"),),
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


def _role_attribution_contract(
    question: str,
    analysis: QuestionAnalysis,
) -> tuple[QuestionAnalysis, str, tuple[str, ...]] | None:
    match = re.match(
        r"^(?P<subject>.+?)(?:是|为)(?P<interrogative>哪个|哪位|什么)"
        r"(?P<predicate>[^，。？?]{2,20})的[。？?]?$",
        question,
    )
    if match is None:
        return None
    tagged = [
        (word.strip(), flag)
        for word, flag in posseg.cut(match.group("predicate"))
        if word.strip() and word.strip() != "所"
    ]
    if len(tagged) < 2 or not tagged[-1][1].startswith("v"):
        return None
    role = "".join(word for word, _ in tagged[:-1]).strip(" 的")
    action = tagged[-1][0].strip(" 的")
    subject = _clean_subject(match.group("subject"))
    if len(subject) < 2 or not role or not action:
        return None
    person_role = match.group("interrogative") == "哪位" or role.endswith(
        ("人", "者", "家", "师", "员", "手", "官", "帝", "王", "长")
    )
    return (
        replace(
            analysis,
            intent="agent",
            entity_type="person" if person_role else analysis.entity_type,
            subjects=(),
            expects_list=False,
            expects_complete_list=False,
        ),
        subject,
        tuple(dict.fromkeys((role, action))),
    )


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
    if analysis.expects_list or counted_list_size(question) is not None:
        return "list"
    if any(marker in question for marker in ("故事", "经过", "来龙去脉")):
        return "narrative"
    if any(marker in question for marker in ("结局", "结尾", "最终结果")):
        return "summary"
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
    ) or counted_list_size(question) is not None or any(
        marker in question
        for marker in ("哪几个", "哪几种", "哪几类", "哪几项", "哪几篇", "哪几部", "哪几本")
    ):
        return "all"
    if analysis.expects_list:
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
        primary_relation = relations[0] if relations else "人物"
        candidates = (
            f"{subject} {primary_relation}",
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
        event_relation = next(
            (
                relation
                for relation in relations
                if relation not in {"原因", "因由", "缘由", "导致", "因素"}
            ),
            relations[0],
        )
        candidates = (
            f"{subject}{event_relation}",
            f"{subject} {event_relation} 原因",
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


def _cause_event_relation(question: str, subject: str) -> str:
    """Keep a concrete event in cause queries, rather than only ``原因``."""

    if not subject:
        return ""
    match = re.search(
        r"(?:因为什么原因|为什么|为何)(?:要|会|会去|而)?(?P<event>[^？?。；;，,]{2,32})",
        question,
    )
    if match is None:
        return ""
    event = match.group("event").strip(" 的了过着")
    if any(
        marker in event
        for marker in ("灭亡", "衰亡", "衰落", "失败", "崩溃", "解体", "成功")
    ):
        return ""
    return event


def _clean_definition_subject(value: str) -> str:
    value = re.split(
        r"(?:这个|這個|的由来|的由來|得由来|得由來|由来|由來|并|並|以及|同时|同時)",
        value,
        maxsplit=1,
    )[0]
    return _clean_subject(value)


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
        subject = analysis.subjects[0].rsplit(" ", 1)[0].strip()
        office = re.match(
            r"^(?P<scope>.+?)的(?:副总统|总统|总理|首相|主席)$",
            subject,
        )
        return office.group("scope").strip() if office else subject
    if analysis.intent == "list" and analysis.subjects:
        first = analysis.subjects[0]
        return _clean_subject(
            first.split(" ", 1)[0].removesuffix("列表").strip()
        )
    if analysis.intent == "list" and counted_list_size(question) is not None:
        nominal = question.strip(" 的是请，,。；;？?")
        if not any(marker in nominal for marker in ("哪", "哪些", "什么", "谁")):
            return _clean_subject(nominal)
    quantitative = re.match(
        r"^(?P<subject>.+?)(?:的)?(?:人口|面积|面積|全长|全長|长度|長度|高度|海拔)"
        r"(?:数据|數據)?(?:(?:是|为|有)?(?:多少|什么|具体数字)|[。？?]|$)",
        question,
    )
    if quantitative:
        return _clean_subject(quantitative.group("subject"))
    if analysis.intent == "fact" and not any(
        marker in question
        for marker in ("什么", "哪个", "哪些", "多少", "谁", "哪里", "如何", "怎么", "吗")
    ):
        nominal = question.strip(" 的是请，,。；;？?")
        if len(nominal) >= 2:
            return _clean_subject(nominal)
    patterns = {
        "definition": r"^(?:请简要介绍|请介绍|简要介绍|介绍)?(?P<subject>.+?)(?:是什么|是谁|指的是什么)[。？?]?$",
        "cause": r"^(?:导致)?(?P<subject>.+?)(?:是)?(?:因为什么原因|为什么|为何|的原因|是哪些因素造成)",
        "procedure": r"^(?P<left>.+?)(?:如何|怎么)(?P<right>.+?)[。？?]?$",
        "time": r"^(?P<subject>.+?)(?:是什么时候|是在什么时候|是哪一年|在哪一年|什么时候|多久|哪一年|何时)",
        "location": r"^(?P<subject>.+?)(?:位于哪里|位于哪|在哪里|在哪儿|在哪个球场|在哪座球场|在哪个场馆)",
        "birthplace": r"^(?P<subject>.+?)(?:出生于哪里|出生在哪里|哪里出生)",
    }
    pattern = patterns.get(analysis.intent)
    if not pattern:
        if analysis.intent == "definition":
            for prefix in ("请简要介绍", "请介绍", "简要介绍", "介绍"):
                if question.startswith(prefix):
                    return _clean_definition_subject(question[len(prefix):])
        return ""
    match = re.search(pattern, question.strip())
    if not match:
        if analysis.intent == "definition":
            for prefix in ("请简要介绍", "请介绍", "简要介绍", "介绍"):
                if question.startswith(prefix):
                    return _clean_definition_subject(question[len(prefix):])
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
        if counted_list_size(question) is not None:
            return ("是指", "包括", "分别是", "分别为")
        if any(marker in question for marker in ("哪几个", "哪几种", "哪几类", "哪几项", "哪几篇", "哪几部", "哪几本")):
            return ("是指", "包括", "分别是", "分别为")
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
        r"(?:的|背后的)?(开国皇帝|总统|副总统|总理|首相|主席|创始人|创办人|创办者|建立者|创建者|负责人|老板|首席执行官|CEO|发明者|发现者|"
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
    return _agent_relation_equivalents(relation)


def _agent_relation_equivalents(relation: str) -> tuple[str, ...]:
    equivalents = {
        "开国皇帝": ("开国皇帝", "高祖", "太祖", "称帝", "登基"),
        "总统": ("总统", "国家元首", "现任总统"),
        "副总统": ("副总统", "副總統"),
        "总理": ("总理", "首相"),
        "首相": ("首相", "总理"),
        "主席": ("主席", "现任主席"),
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
        "老板": ("老板", "创始人", "创办人", "创办者", "负责人", "首席执行官", "CEO"),
        "负责人": ("负责人", "老板", "创始人", "创办人", "创办者", "负责"),
        "首席执行官": ("首席执行官", "CEO", "负责人", "老板"),
        "建立者": ("建立者", "建立", "创立", "创建"),
        "创建者": ("创建者", "创建", "创立", "建立"),
        "发明者": ("发明者", "发明", "研制", "创造"),
        "发现者": ("发现者", "发现", "首次发现"),
        "建造者": ("建造者", "建造", "修建", "建设"),
        "执导者": ("执导", "导演", "执导者"),
        "纪念": ("纪念", "纪念人物", "起源", "由来"),
        "发起": ("发起", "发动", "组织", "策划", "领导"),
        "提出": ("提出", "发起", "倡议", "主张"),
        "杀害": ("杀害", "害死", "处死", "遇害"),
        "害死": ("害死", "杀害", "处死", "遇害"),
        "处死": ("处死", "杀害", "害死", "遇害"),
    }
    return equivalents.get(relation, (relation,))


def _clean_subject(value: str) -> str:
    cleaned = clean_subject_scope(value)
    cleaned = re.sub(r"^(?:现在|目前|当前|如今)", "", cleaned)
    cleaned = re.sub(r"(?:究竟|到底)$", "", cleaned)
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
