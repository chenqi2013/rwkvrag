from llamaindex_retrieval.failure_diagnosis import diagnose_failure


def test_empty_retrieval_is_data_missing() -> None:
    diagnosis = diagnose_failure(
        answer="根据检索到的资料，无法确定。",
        sources=[],
        retrieval={"returned": 0},
        generation={"evidence_count": 0},
    )

    assert diagnosis.category == "data_missing"
    assert diagnosis.stage == "retrieval"


def test_retrieval_error_is_retrieval_failure() -> None:
    diagnosis = diagnose_failure(
        answer="根据检索到的资料，无法确定。",
        sources=[],
        retrieval={"returned": 0, "retrieval_error_type": "ConnectionError"},
        generation={"evidence_count": 0},
    )

    assert diagnosis.category == "retrieval_failed"
    assert diagnosis.reason == "retrieval_error"


def test_irrelevant_results_are_retrieval_failure() -> None:
    diagnosis = diagnose_failure(
        answer="根据检索到的资料，无法确定。",
        sources=[object()],
        retrieval={"returned": 1},
        generation={
            "evidence_count": 1,
            "evidence_gate_passed": False,
            "evidence_gate_issues": ["subject_title_mismatch"],
        },
    )

    assert diagnosis.category == "retrieval_failed"


def test_relevant_sources_with_extraction_error_are_extraction_failure() -> None:
    diagnosis = diagnose_failure(
        answer="根据检索到的资料，无法确定。",
        sources=[object()],
        retrieval={"returned": 1},
        generation={
            "evidence_count": 1,
            "evidence_gate_passed": False,
            "evidence_gate_issues": ["field_evidence_missing"],
            "matched_evidence_terms": ["作者"],
            "field_evidence_available": True,
            "field_evidence_strategy": "model_empty",
            "field_evidence_errors": [],
            "field_evidence": [],
        },
    )

    assert diagnosis.category == "evidence_extraction_failed"


def test_unmatched_entity_is_data_missing_before_extraction_error() -> None:
    diagnosis = diagnose_failure(
        answer="根据检索到的资料，无法确定。",
        sources=[object()],
        retrieval={"returned": 1},
        generation={
            "evidence_count": 1,
            "evidence_gate_passed": False,
            "evidence_gate_issues": ["field_evidence_missing"],
            "evidence_anchors": ["阿尔法泽"],
            "matched_evidence_anchors": [],
            "field_evidence_errors": ["ValueError"],
        },
    )

    assert diagnosis.category == "data_missing"
    assert diagnosis.reason == "entity_not_found"


def test_narrative_without_extractable_detail_is_data_missing() -> None:
    diagnosis = diagnose_failure(
        answer="根据检索到的资料，无法确定。",
        sources=[object()],
        retrieval={"returned": 1},
        generation={
            "evidence_count": 1,
            "evidence_gate_passed": False,
            "evidence_gate_issues": ["field_evidence_missing"],
            "matched_evidence_terms": ["草船借箭"],
            "matched_evidence_anchors": ["草船借箭"],
            "field_evidence_available": True,
            "field_evidence_strategy": "model_empty",
            "field_evidence_errors": [],
            "field_evidence": [],
            "answer_shape": "narrative",
        },
    )

    assert diagnosis.category == "data_missing"
    assert diagnosis.reason == "insufficient_answer_evidence"


def test_generation_timeout_is_generation_failure() -> None:
    diagnosis = diagnose_failure(
        answer="根据检索到的资料，无法确定。",
        sources=[object()],
        retrieval={"returned": 1},
        generation={
            "evidence_count": 1,
            "evidence_gate_passed": True,
            "answer_strategy": "generation_timeout_fallback",
            "answer_block_reason": "TimeoutError",
        },
    )

    assert diagnosis.category == "generation_failed"


def test_grounded_answer_has_no_failure_category() -> None:
    diagnosis = diagnose_failure(
        answer="北京。[资料 1]",
        sources=[object()],
        retrieval={"returned": 1},
        generation={"evidence_count": 1, "evidence_gate_passed": True},
    )

    assert diagnosis.category is None
