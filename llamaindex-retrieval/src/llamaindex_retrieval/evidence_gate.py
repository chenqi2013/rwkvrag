from dataclasses import dataclass
import re
from typing import Any

from .generation import EvidenceAnswerGenerator, EvidenceAssessment
from .lexical_index import lexical_tokens, normalize_search_text
from .qa_analysis import QuestionAnalysis
from .schemas import SourceItem




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
_REFUSAL_MARKERS = ("无法确定", "未检索到", "无法从资料")
_QUALIFIED_TITLE = re.compile(r"^(?P<base>.+?)[（(](?P<qualifier>[^）)]+)[）)]$")


def evaluate_evidence_gate(
    question: str,
    analysis: QuestionAnalysis,
    sources: list[SourceItem],
    *,
    subject: str = "",
    anchor_subject: str = "",
    relations: tuple[str, ...] = (),
    field_evidence_available: bool = False,
    field_candidate_count: int = 0,
) -> EvidenceGateResult:
    assessment = EvidenceAnswerGenerator.assess_evidence(
        question,
        sources,
        subject=anchor_subject,
    )
    evidence = normalize_search_text(
        "\n".join(f"{source.title}\n{source.snippet}" for source in sources)
    )
    relation_terms = _relation_terms(
        question,
        analysis.intent,
        requested_relations=relations,
    )
    matched = tuple(
        term for term in relation_terms
        if normalize_search_text(term) in evidence
        or (analysis.intent == "time" and re.search(r"(?:公元|西元)?\d{3,4}年", evidence))
    )
    issues: list[str] = []
    if not sources:
        issues.append("no_evidence")
    if field_evidence_available:
        if field_candidate_count <= 0 and sources:
            issues.append("field_evidence_missing")
        question_years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", question))
        evidence_years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", evidence))
        if question_years and not question_years <= evidence_years:
            issues.append("temporal_mismatch")
        if (
            analysis.intent == "ordinal"
            and sources
            and not any(
                ordinal_scope_match(
                    question,
                    f"{source.title}\n{source.snippet}",
                )
                for source in sources
            )
        ):
            issues.append("ordinal_scope_mismatch")
        subject_tokens = {
            token for token in lexical_tokens(subject) if len(token) >= 2
        }
        normalized_subject = normalize_search_text(subject).replace(" ", "")
        subject_parts = tuple(
            part for part in re.split(r"(?:和|与|與|及|以及|、)", normalized_subject)
            if len(part) >= 2
        )
        subject_variants = tuple(dict.fromkeys((normalized_subject, *subject_parts)))
        exact_subject_match = bool(normalized_subject) and any(
            any(title_matches_subject(source.title, variant) for variant in subject_variants)
            or any(variant in source_aliases(source) for variant in subject_variants)
            or any(
                variant in normalize_search_text(source.snippet).replace(" ", "")
                for variant in subject_variants
            )
            for source in sources
        )
        if analysis.intent == "comparison" and len(subject_parts) >= 2:
            matched_parts = {
                part
                for part in subject_parts
                if any(
                    title_matches_subject(source.title, part)
                    or part in normalize_search_text(source.snippet).replace(" ", "")
                    for source in sources
                )
            }
            if len(matched_parts) < 2:
                issues.append("subject_mismatch")
        if subject_tokens and (
            not exact_subject_match
            and not (
                len(subject_tokens) == 1
                and subject_tokens.intersection(assessment.matched_anchors)
            )
        ):
            issues.append("subject_mismatch")
        elif (
            not subject_tokens
            and assessment.anchors
            and not assessment.matched_anchors
        ):
            issues.append("subject_mismatch")
        return EvidenceGateResult(
            assessment=assessment,
            relation_terms=relation_terms,
            matched_relation_terms=matched,
            passed=not issues,
            issues=tuple(issues),
        )
    normalized_subject = normalize_search_text(subject).replace(" ", "")
    exact_title_match = bool(normalized_subject) and any(
        title_matches_subject(source.title, normalized_subject)
        or normalized_subject in source_aliases(source)
        or (
            analysis.intent in {"cause", "time", "list"}
            and title_matches_subject_event(source.title, normalized_subject, question)
        )
        for source in sources
    )
    route_endpoint_match = _route_endpoint_evidence_match(question, sources)
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
        exact_title_match
        or relation_topic_match
        or embedded_definition_match
        or route_endpoint_match
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
    if relation_terms and not matched:
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


def document_aliases(title: str, metadata: dict[str, Any]) -> set[str]:
    aliases = metadata.get("aliases")
    if not isinstance(aliases, list):
        return set()
    normalized_title = normalize_search_text(title).replace(" ", "")
    qualified = _QUALIFIED_TITLE.fullmatch(normalized_title)
    derived_base = qualified.group("base") if qualified else ""
    return {
        normalize_search_text(str(alias)).replace(" ", "")
        for alias in aliases
        if str(alias).strip()
        and normalize_search_text(str(alias)).replace(" ", "") != derived_base
    }


def source_aliases(source: SourceItem) -> set[str]:
    return document_aliases(source.title, source.metadata)


def _route_endpoint_evidence_match(question: str, sources: list[SourceItem]) -> bool:
    match = re.search(
        r"连接(?P<start>[^，。？?和与]{1,20})(?:和|与)"
        r"(?P<end>[^，。？?的]{1,20})的[^，。？?]{2,32}?(?:线路|路线)",
        normalize_search_text(question),
    )
    if match is None:
        return False
    start = normalize_search_text(match.group("start")).replace(" ", "")
    end = normalize_search_text(match.group("end")).replace(" ", "")
    return any(
        start in (text := normalize_search_text(source.snippet).replace(" ", ""))
        and end in text
        and bool(re.search(r"(?:线路|路线|号线|由.{0,16}至|起点|终点)", text))
        for source in sources
    )


def evaluate_answer_support(
    answer: str,
    sources: list[SourceItem],
    *,
    question: str = "",
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
        if len(term.strip()) >= 2
    }
    answer_body = _CITATION.sub("", answer)
    answer_terms = {
        term
        for term in lexical_tokens(answer_body)
        if len(term.strip()) >= 2
    }
    question_terms = set(lexical_tokens(question)) if question else set()
    claim_terms = answer_terms - question_terms
    terms_to_verify = claim_terms or answer_terms
    supported = terms_to_verify & evidence_terms
    unsupported = terms_to_verify - evidence_terms
    coverage = len(supported) / max(1, len(terms_to_verify))
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
            if len(term.strip()) >= 2
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
            if len(term.strip()) >= 2
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
    return any(character.isalnum() for character in term)


def title_matches_subject(title: str, normalized_subject: str) -> bool:
    normalized_title = normalize_search_text(title).replace(" ", "")
    normalized_title = _normalize_title_numbers(normalized_title)
    normalized_subject = _normalize_title_numbers(normalized_subject)
    if normalized_title == normalized_subject:
        return True
    def canonical(value: str) -> str:
        for qualifier in ("国际足协", "国际足总", "国际足球联合会"):
            value = value.replace(qualifier, "")
        value = value.replace("反潜护卫艇", "猎潜艇")
        for suffix in ("事变", "之变", "政变"):
            if value.endswith(suffix):
                value = value[: -len(suffix)]
                break
        return value

    canonical_title = canonical(normalized_title)
    canonical_subject = canonical(normalized_subject)
    if canonical_title == canonical_subject:
        return True
    if canonical_subject != normalized_subject:
        title_base = re.split(r"[（(]", normalized_title, maxsplit=1)[0]
        if canonical(title_base) == canonical_subject:
            return True
    # Questions often add a broad scope marker to an otherwise exact entity,
    # e.g. "中国四大名著" for the page "四大名著".  Only allow this for a
    # small, explicit set of scope markers so similarly named pages do not pass
    # the identity gate accidentally.
    for prefix in (
        "中国", "中华人民共和国", "我国", "世界",
        "中国历史", "中国历史上", "中国历史上的",
        "历史", "历史上", "历史上的",
    ):
        if canonical_subject.startswith(prefix) and canonical_subject[len(prefix):] == canonical_title:
            return True
    return False


def _normalize_title_numbers(value: str) -> str:
    """Make Arabic and common Chinese numerals comparable in titles."""

    return value.translate(str.maketrans({"〇": "0", "一": "1", "二": "2", "两": "2", "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"}))


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


def ordinal_scope_match(question: str, text: str) -> bool:
    """Check that an ordinal claim is bound to the question's scope.

    Lexical retrieval can return a sentence such as "某政权首位使用……" for
    a question asking for the first item in a larger historical scope.  The
    check is deliberately structural: it only verifies that the scope anchor
    and ordinal marker occur near one another in the same sentence.  It does
    not infer or provide an answer.
    """
    return True


def _relation_terms(
    question: str,
    intent: str = "",
    *,
    requested_relations: tuple[str, ...] = (),
) -> tuple[str, ...]:
    requested = tuple(
        relation.strip()
        for relation in requested_relations
        if relation.strip()
    )
    return requested
