from llamaindex_retrieval.repository import search_answer_status


def test_search_answer_status_uses_exact_refusal_answer() -> None:
    assert search_answer_status({"answer": "根据检索到的资料，无法确定。"}) == "refused"
    assert search_answer_status({"answer": " 北京。 "}) == "answered"
    assert search_answer_status({"answer": "根据检索到的资料，可以确定是北京。"}) == "answered"
