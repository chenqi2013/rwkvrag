from dataclasses import dataclass
import re

import jieba.posseg as posseg

from .generation import EvidenceAnswerGenerator, EvidenceAssessment
from .lexical_index import lexical_tokens, normalize_search_text
from .qa_analysis import QuestionAnalysis
from .schemas import SourceItem


_RELATION_GROUPS = {
    "cause": ("原因", "因为", "由于", "导致", "造成", "引发", "使得", "从而", "灭亡", "衰亡"),
    "time": ("年", "月", "日", "时候", "期间", "成立", "创立", "创建", "建立", "组建", "出生", "生于", "逝世", "去世", "上映", "发行", "开通", "回归"),
    "location": ("位于", "坐落", "地处", "属于", "隶属", "北部", "南部", "东部", "西部", "境内", "附近"),
    "birthplace": ("出生于", "生于", "出生地"),
    "agent": (
        "由", "创立", "创建", "建立", "发明", "发现", "开启", "开辟", "开通", "出使",
        "凿空", "作者", "导演", "主演", "领导", "现在", "目前", "当前", "现任", "得主", "获得", "授予",
    ),
    "ordinal": ("第一个", "第一位", "首位", "最早"),
    "list": ("列表", "包括", "包含", "分别", "站名", "车站", "条目"),
    "comparison": ("区别", "不同", "相比", "分别", "而", "但"),
    "definition": ("是", "为", "指", "属于"),
}
_RELATION_EQUIVALENTS = {
    "成立": ("成立", "创立", "创建", "建立", "组建"),
    "创建": ("创建", "创立", "成立", "建立"),
    "出生": ("出生", "生于"),
    "逝世": ("逝世", "去世", "病逝", "卒于"),
    "上映": ("上映", "首映", "发行"),
    "位于": ("位于", "坐落", "地处", "在", "省", "市", "区", "县", "州", "国"),
    "开启": ("开启", "开辟", "开通", "出使", "凿空"),
    "开辟": ("开启", "开辟", "开通", "出使", "凿空"),
    "开通": ("开启", "开辟", "开通", "出使", "凿空"),
    "现在": ("现在", "目前", "当前", "现任"),
    "目前": ("现在", "目前", "当前", "现任"),
    "当前": ("现在", "目前", "当前", "现任"),
    "现任": ("现在", "目前", "当前", "现任"),
    "得主": ("得主", "获得", "获奖", "授予"),
    "获得者": ("得主", "获得", "获奖", "授予"),
    "作者": ("作者", "创作", "编剧", "作曲", "作词", "导演", "设计"),
}
_STRICT_RELATION_INTENTS = {"ordinal", "time", "location", "birthplace", "agent"}


@dataclass(frozen=True)
class EvidenceGateResult:
    assessment: EvidenceAssessment
    relation_terms: tuple[str, ...]
    matched_relation_terms: tuple[str, ...]
    passed: bool
    issues: tuple[str, ...]


@dataclass(frozen=True)
class AnswerSupportResult:
    passed: bool
    coverage: float
    supported_terms: tuple[str, ...]
    unsupported_terms: tuple[str, ...]
    issues: tuple[str, ...]


_CITATION = re.compile(r"\[资料\s*([1-9]\d*)\]")
_ANSWER_NOISE = {
    "根据", "资料", "问题", "答案", "可以", "包括", "相关", "一个", "这个",
    "以及", "并且", "因此", "其中", "主要", "的是", "属于", "成为",
}
_REFUSAL_MARKERS = ("无法确定", "未检索到", "无法从资料")
_CJK_TERM = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")


def evaluate_evidence_gate(
    question: str,
    analysis: QuestionAnalysis,
    sources: list[SourceItem],
    *,
    subject: str = "",
) -> EvidenceGateResult:
    assessment = EvidenceAnswerGenerator.assess_evidence(question, sources)
    evidence = normalize_search_text(
        "\n".join(f"{source.title}\n{source.snippet}" for source in sources)
    )
    relation_terms = _relation_terms(question, analysis.intent)
    matched = tuple(
        term for term in relation_terms
        if normalize_search_text(term) in evidence
        or (analysis.intent == "time" and re.search(r"(?:公元|西元)?\d{3,4}年", evidence))
    )
    issues: list[str] = []
    if not sources:
        issues.append("no_evidence")
    normalized_subject = normalize_search_text(subject).replace(" ", "")
    exact_title_match = bool(normalized_subject) and any(
        title_matches_subject(source.title, normalized_subject)
        or (
            analysis.intent in {"cause", "time", "list"}
            and title_matches_subject_event(source.title, normalized_subject, question)
        )
        for source in sources
    )
    relation_topic_match = bool(normalized_subject and matched) and any(
        title_matches_subject_topic(source.title, normalized_subject)
        and all(
            token in normalize_search_text(f"{source.title}\n{source.snippet}")
            for token in lexical_tokens(subject)
            if len(token) >= 2
        )
        for source in sources
    )
    embedded_definition_match = (
        analysis.intent == "definition"
        and bool(normalized_subject)
        and any(
            _contains_embedded_definition(source.snippet, normalized_subject)
            for source in sources
        )
    )
    acceptable_title_match = (
        exact_title_match or relation_topic_match or embedded_definition_match
    )
    if not acceptable_title_match:
        if assessment.anchors and not assessment.matched_anchors:
            issues.append("subject_mismatch")
        elif not assessment.grounded:
            issues.append("weak_subject_coverage")
    requires_exact_title = analysis.intent in {
        "definition", "location", "birthplace", "cause",
    } or (analysis.intent == "list" and analysis.expects_complete_list)
    if normalized_subject and requires_exact_title:
        if not acceptable_title_match:
            issues.append("subject_title_mismatch")
    if (
        analysis.intent in _STRICT_RELATION_INTENTS
        and relation_terms
        and not matched
    ):
        issues.append("relation_mismatch")
    question_years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", question))
    evidence_years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", evidence))
    if question_years and not question_years <= evidence_years:
        issues.append("temporal_mismatch")
    return EvidenceGateResult(
        assessment=assessment,
        relation_terms=relation_terms,
        matched_relation_terms=matched,
        passed=not issues,
        issues=tuple(issues),
    )


def evaluate_answer_support(
    answer: str,
    sources: list[SourceItem],
) -> AnswerSupportResult:
    if any(marker in answer for marker in _REFUSAL_MARKERS):
        return AnswerSupportResult(True, 1.0, (), (), ())
    citation_indexes = {
        int(match.group(1)) - 1
        for match in _CITATION.finditer(answer)
        if 1 <= int(match.group(1)) <= len(sources)
    }
    issues: list[str] = []
    if not citation_indexes:
        issues.append("missing_valid_citation")
        cited_sources = sources
    else:
        cited_sources = [sources[index] for index in sorted(citation_indexes)]
        cited_document_ids = {source.document_id for source in cited_sources}
        cited_sources = [
            source for source in sources
            if source.document_id in cited_document_ids
        ]
    evidence_terms = {
        term
        for source in cited_sources
        for term in lexical_tokens(f"{source.title}\n{source.snippet}")
        if term not in _ANSWER_NOISE
    }
    answer_body = _CITATION.sub("", answer)
    answer_terms = {
        term
        for term in lexical_tokens(answer_body)
        if term not in _ANSWER_NOISE and len(term.strip()) >= 2
    }
    supported = answer_terms & evidence_terms
    unsupported = answer_terms - evidence_terms
    coverage = len(supported) / max(1, len(answer_terms))
    unsupported_entity_terms = {
        term
        for term in unsupported
        if _is_potential_entity_term(term)
        and not any(
            len(evidence_term) >= 2
            and (evidence_term in term or term in evidence_term)
            for evidence_term in evidence_terms
        )
    }
    if len(answer_terms) >= 4 and coverage < 0.45:
        issues.append("weak_answer_evidence_overlap")
    if unsupported_entity_terms:
        issues.append("unsupported_entity_term")
    return AnswerSupportResult(
        passed=not issues,
        coverage=coverage,
        supported_terms=tuple(sorted(supported)),
        unsupported_terms=tuple(sorted(unsupported)),
        issues=tuple(issues),
    )


def repair_answer_citations(answer: str, sources: list[SourceItem]) -> str:
    if not sources or any(marker in answer for marker in _REFUSAL_MARKERS):
        return answer
    source_terms = [
        {
            term for term in lexical_tokens(f"{source.title}\n{source.snippet}")
            if term not in _ANSWER_NOISE
        }
        for source in sources
    ]
    document_indexes: dict[str, list[int]] = {}
    for index, source in enumerate(sources):
        document_indexes.setdefault(source.document_id, []).append(index)
    segments = re.findall(r"[^；;。！？!?]+[；;。！？!?]?", _CITATION.sub("", answer))
    repaired: list[str] = []
    for segment in segments:
        body = segment.rstrip("；;。！？!? ")
        punctuation = segment[len(body):].strip() or ""
        terms = {
            term for term in lexical_tokens(body)
            if term not in _ANSWER_NOISE and len(term.strip()) >= 2
        }
        scores = [len(terms & evidence_terms) for evidence_terms in source_terms]
        best_score = max(scores, default=0)
        if best_score:
            best_index = scores.index(best_score)
            citation_index = best_index + 1
            repaired.append(f"{body}[资料 {citation_index}]{punctuation}")
        else:
            repaired.append(segment.strip())
    return "".join(repaired).strip()


def _is_potential_entity_term(term: str) -> bool:
    if len(term) < 3:
        return False
    if term.isascii():
        return any(character.isalnum() for character in term)
    if not _CJK_TERM.fullmatch(term):
        return False
    flags = {item.flag for item in posseg.cut(term) if item.word.strip()}
    return bool(flags) and all(flag.startswith(("nr", "ns", "nt", "nz")) for flag in flags)


def title_matches_subject(title: str, normalized_subject: str) -> bool:
    normalized_title = normalize_search_text(title).replace(" ", "")
    if normalized_title == normalized_subject or normalized_title.startswith(
        (f"{normalized_subject}(", f"{normalized_subject}（")
    ):
        return True
    def canonical(value: str) -> str:
        for qualifier in ("国际足协", "国际足总", "国际足球联合会"):
            value = value.replace(qualifier, "")
        value = value.replace("反潜护卫艇", "猎潜艇")
        return value

    return canonical(normalized_title) == canonical(normalized_subject)


def title_matches_subject_event(title: str, normalized_subject: str, question: str) -> bool:
    normalized_title = normalize_search_text(title).replace(" ", "")
    normalized_question = normalize_search_text(question).replace(" ", "")
    if not normalized_title.startswith(normalized_subject):
        return False
    suffix = normalized_title[len(normalized_subject):]
    if not suffix:
        return False
    suffix_tokens = {
        token for token in lexical_tokens(suffix)
        if len(token) >= 2 and token not in {"发生", "出现", "进行", "相关"}
    }
    return bool(suffix_tokens) and any(token in normalized_question for token in suffix_tokens)


def title_matches_subject_topic(title: str, normalized_subject: str) -> bool:
    normalized_title = normalize_search_text(title).replace(" ", "")
    return (
        len(normalized_title) >= 2
        and normalized_title != normalized_subject
        and normalized_subject.endswith(normalized_title)
    )


def _contains_embedded_definition(text: str, normalized_subject: str) -> bool:
    normalized = normalize_search_text(text).replace(" ", "")
    return bool(re.search(
        rf"{re.escape(normalized_subject)}(?:[（(][^）)\n]{{0,40}}[）)])?"
        r"(?:是|为|指|属于|由[^。！？\n]{1,40}(?:发展|放大|缩小|改造|研制|设计)而来)",
        normalized,
    ))


def _relation_terms(question: str, intent: str) -> tuple[str, ...]:
    candidates = _RELATION_GROUPS.get(intent, ())
    normalized_question = normalize_search_text(question)
    explicit = tuple(term for term in candidates if normalize_search_text(term) in normalized_question)
    if explicit:
        expanded = [
            equivalent
            for term in explicit
            for equivalent in _RELATION_EQUIVALENTS.get(term, (term,))
        ]
        if intent == "ordinal":
            expanded.extend(_RELATION_GROUPS["ordinal"])
        return tuple(dict.fromkeys(expanded))
    question_tokens = set(lexical_tokens(question))
    inferred = tuple(term for term in candidates if question_tokens & set(lexical_tokens(term)))
    return inferred or tuple(candidates if intent in _STRICT_RELATION_INTENTS else ())
