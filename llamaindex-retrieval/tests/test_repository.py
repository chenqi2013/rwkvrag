from llamaindex_retrieval.repository import search_answer_status, search_failure_category, search_failure_reason


def test_search_answer_status_uses_exact_refusal_answer() -> None:
    assert search_answer_status({"answer": "根据检索到的资料，无法确定。"}) == "refused"
    assert search_answer_status({"answer": " 北京。 "}) == "answered"
    assert search_answer_status({"answer": "根据检索到的资料，可以确定是北京。"}) == "answered"


def test_search_failure_fields_are_read_from_generation() -> None:
    response = {"generation": {
        "failure_category": "retrieval_failed",
        "failure_reason": "evidence_not_relevant",
    }}

    assert search_failure_category(response) == "retrieval_failed"
    assert search_failure_reason(response) == "evidence_not_relevant"
    assert search_failure_category({"generation": {"failure_category": "unknown"}}) is None
