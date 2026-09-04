"""Stage-level diagnosis for unsuccessful grounded answers.

The diagnosis deliberately uses only runtime signals.  It does not inspect the
meaning of an individual question, so the same classification works for all
knowledge-base domains.
"""

from dataclasses import dataclass
from typing import Any, Literal

FailureCategory = Literal[
    "data_missing",
    "retrieval_failed",
    "evidence_extraction_failed",
    "generation_failed",
]


@dataclass(frozen=True)
class FailureDiagnosis:
    category: FailureCategory | None
    reason: str | None = None
    stage: Literal["retrieval", "evidence_extraction", "generation"] | None = None


_REFUSAL_MARKERS = (
    "无法确定",
    "未检索到",
    "无法从资料",
    "资料不足",
    "不能确定",
)
_GENERATION_FAILURE_STRATEGIES = {
    "generation_timeout_fallback",
    "generation_error_fallback",
    "answer_grounding_blocked",
}


def _is_refusal(answer: object) -> bool:
    text = str(answer or "").strip()
    return not text or any(marker in text for marker in _REFUSAL_MARKERS)


def diagnose_failure(
    *,
    answer: object,
    sources: list[Any],
    retrieval: dict[str, Any] | None,
    generation: dict[str, Any] | None,
) -> FailureDiagnosis:
    """Classify a failed answer, or return ``category=None`` when successful.

    Retrieval errors are checked first, then the presence/quality of evidence,
    then extraction errors, and finally answer generation/grounding signals.
    This ordering reflects the pipeline and prevents a downstream refusal from
    masking an earlier, more useful diagnosis.
    """

    retrieval = retrieval or {}
    generation = generation or {}
    if not _is_refusal(answer):
        return FailureDiagnosis(None)

    if retrieval.get("retrieval_error") or retrieval.get("retrieval_error_type"):
        return FailureDiagnosis(
            "retrieval_failed",
            "retrieval_error",
            "retrieval",
        )

    returned = retrieval.get("returned")
    if returned == 0:
        return FailureDiagnosis("data_missing", "no_evidence_returned", "retrieval")

    gate_passed = generation.get("evidence_gate_passed") is True
    gate_issues = {str(issue) for issue in generation.get("evidence_gate_issues") or ()}
    matched_terms = generation.get("matched_evidence_terms") or ()
    matched_anchors = generation.get("matched_evidence_anchors") or ()
    field_errors = generation.get("field_evidence_errors") or ()
    field_strategy = str(generation.get("field_evidence_strategy") or "")
    field_available = generation.get("field_evidence_available") is True
    field_candidates = generation.get("field_evidence") or ()

    evidence_count = int(generation.get("evidence_count") or len(sources))

    # Results that only share a loose token (for example “阿尔法” for the
    # unknown entity “阿尔法泽”) are not evidence for the requested object.
    # Treat this as a data-coverage miss before considering extraction errors.
    if generation.get("evidence_anchors") and not matched_anchors and "field_evidence_missing" in gate_issues:
        return FailureDiagnosis("data_missing", "entity_not_found", "retrieval")

    # A returned set with no subject/relation match means the search stage did
    # not find the requested object, even though BM25 returned documents.
    if not gate_passed and (
        gate_issues.intersection({
            "subject_mismatch",
            "subject_title_mismatch",
            "weak_subject_coverage",
            "relation_mismatch",
            "temporal_mismatch",
            "ordinal_scope_mismatch",
        })
        or (not matched_terms and not matched_anchors)
    ):
        return FailureDiagnosis(
            "retrieval_failed",
            "evidence_not_relevant",
            "retrieval",
        )

    # Relevant evidence reached the extractor, but it produced no usable
    # candidates or reported per-source/model errors.
    if field_errors or (
        field_available
        and not field_candidates
        and field_strategy in {"model_empty", "model_empty_remapped"}
    ):
        if not field_errors and generation.get("answer_shape") in {"summary", "narrative"}:
            return FailureDiagnosis(
                "data_missing",
                "insufficient_answer_evidence",
                "evidence_extraction",
            )
        return FailureDiagnosis(
            "evidence_extraction_failed",
            "evidence_candidates_missing" if not field_errors else "evidence_extraction_error",
            "evidence_extraction",
        )

    if evidence_count <= 0:
        return FailureDiagnosis("data_missing", "no_answer_evidence", "retrieval")

    answer_strategy = str(generation.get("answer_strategy") or "")
    answer_block_reason = str(generation.get("answer_block_reason") or "")
    if answer_strategy in _GENERATION_FAILURE_STRATEGIES or answer_block_reason in {
        "answer_support_failed",
        "TimeoutError",
        "AnswerGenerationError",
    }:
        return FailureDiagnosis(
            "generation_failed",
            answer_block_reason or answer_strategy,
            "generation",
        )

    if gate_passed and (
        generation.get("answer_support_passed") is False
        or generation.get("grounding_valid") is False
    ):
        return FailureDiagnosis("generation_failed", "answer_not_grounded", "generation")

    if gate_passed:
        return FailureDiagnosis("generation_failed", "answer_not_generated", "generation")

    # A non-empty result set without enough evidence to decide is most useful
    # to report as a retrieval failure rather than silently calling it missing.
    return FailureDiagnosis("retrieval_failed", "evidence_gate_failed", "retrieval")
