"""Content-neutral evidence and output protocol checks."""

from dataclasses import dataclass
import re
from typing import Any

from .generation import EvidenceAssessment
from .lexical_index import normalize_search_text
from .qa_analysis import QuestionAnalysis
from .schemas import SourceItem


_CITATION = re.compile(r"\[资料\s*([1-9]\d*)\]")
_REFUSAL_MARKERS = ("无法确定", "未检索到", "无法从资料")


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
    """Gate only transport completeness; semantic selection stays in the model."""

    issues: list[str] = []
    if not sources:
        issues.append("no_evidence")
    if field_evidence_available and sources and field_candidate_count <= 0:
        issues.append("field_evidence_missing")
    return EvidenceGateResult(
        assessment=EvidenceAssessment(set(), set()),
        relation_terms=tuple(value.strip() for value in relations if value.strip()),
        matched_relation_terms=(),
        passed=not issues,
        issues=tuple(issues),
    )


def document_aliases(title: str, metadata: dict[str, Any]) -> set[str]:
    """Return aliases explicitly supplied by document metadata."""

    aliases = metadata.get("aliases")
    if not isinstance(aliases, list):
        return set()
    return {
        normalize_search_text(str(alias)).replace(" ", "")
        for alias in aliases
        if str(alias).strip()
    }


def source_aliases(source: SourceItem) -> set[str]:
    return document_aliases(source.title, source.metadata)


def evaluate_answer_support(
    answer: str,
    sources: list[SourceItem],
    *,
    question: str = "",
) -> AnswerSupportResult:
    """Check only that a non-empty answer references available source numbers."""

    if not sources:
        return AnswerSupportResult(False, 0.0, (), (), ("no_evidence",))
    references = [int(match.group(1)) for match in _CITATION.finditer(answer)]
    invalid = [number for number in references if number > len(sources)]
    issues = []
    if not references and not any(marker in answer for marker in _REFUSAL_MARKERS):
        issues.append("missing_valid_citation")
    if invalid:
        issues.append("unknown_citation")
    return AnswerSupportResult(
        passed=not issues,
        coverage=1.0 if not issues else 0.0,
        supported_terms=(),
        unsupported_terms=(),
        issues=tuple(issues),
    )


def repair_answer_citations(answer: str, sources: list[SourceItem]) -> str:
    """Compatibility API; model output is never modified by code."""

    return answer


def title_matches_subject(title: str, normalized_subject: str) -> bool:
    return normalize_search_text(title).replace(" ", "") == normalize_search_text(
        normalized_subject
    ).replace(" ", "")


def title_matches_subject_event(title: str, normalized_subject: str, question: str) -> bool:
    normalized_title = normalize_search_text(title).replace(" ", "")
    subject = normalize_search_text(normalized_subject).replace(" ", "")
    query = normalize_search_text(question).replace(" ", "")
    return normalized_title.startswith(subject) and normalized_title[len(subject):] in query


def title_matches_subject_topic(title: str, normalized_subject: str) -> bool:
    normalized_title = normalize_search_text(title).replace(" ", "")
    subject = normalize_search_text(normalized_subject).replace(" ", "")
    return bool(normalized_title and normalized_title != subject and subject.endswith(normalized_title))


def ordinal_scope_match(question: str, text: str) -> bool:
    return True
