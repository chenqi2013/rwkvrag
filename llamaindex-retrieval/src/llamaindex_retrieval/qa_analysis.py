import re
from dataclasses import dataclass

from .lexical_index import normalize_search_text
from .schemas import SourceItem


@dataclass(frozen=True)
class QuestionAnalysis:
    intent: str = "fact"
    entity_type: str = "unknown"
    subjects: tuple[str, ...] = ()
    expects_list: bool = False
    expects_complete_list: bool = False


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
    return None


def analyze_question(question: str) -> QuestionAnalysis:
    return QuestionAnalysis()


def counted_list_size(question: str) -> int | None:
    return None


def clean_question_shell(question: str) -> str:
    return normalize_search_text(question).strip()


def clean_subject_scope(value: str) -> str:
    return value.strip(" ？?，,。；;:")


def ambiguity_candidates(question: str, sources: list[SourceItem]) -> list[str]:
    return []


def validate_grounding(answer: str, sources: list[SourceItem]) -> GroundingValidation:
    number_pattern = re.compile(r"\d+(?:\.\d+)?%?")
    citation_pattern = re.compile(r"\[资料\s*([1-9]\d*)\]")
    evidence_numbers = set(number_pattern.findall("\n".join(source.snippet for source in sources)))
    body = citation_pattern.sub("", answer)
    unsupported = tuple(sorted(set(number_pattern.findall(body)) - evidence_numbers))
    citations = [int(value) for value in citation_pattern.findall(answer)]
    issues: list[str] = []
    if answer and not citations:
        issues.append("missing_citation")
    if any(value > len(sources) for value in citations):
        issues.append("invalid_citation")
    if unsupported:
        issues.append("unsupported_number")
    return GroundingValidation(answer, not issues, tuple(issues), unsupported)


def remove_unsupported_number_sentences(answer: str, unsupported: tuple[str, ...]) -> str:
    return answer


def validate_list_answer(question: str, answer: str, sources: list[SourceItem]) -> ListValidation:
    return ListValidation(None, None, None, ())


def _is_pure_definition_question(question: str) -> bool:
    return False


def _time_search_queries(question: str) -> tuple[str, ...]:
    return ()


def _list_search_queries(question: str) -> tuple[str, ...]:
    return ()


def _agent_search_queries(question: str) -> tuple[str, ...]:
    return ()


def _ordinal_search_queries(question: str) -> tuple[str, ...]:
    return ()


def _definition_subject(question: str) -> str:
    return ""
