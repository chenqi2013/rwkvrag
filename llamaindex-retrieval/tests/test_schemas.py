import pytest
from pydantic import ValidationError

from llamaindex_retrieval.schemas import SearchRequest


def test_search_request_trims_question() -> None:
    request = SearchRequest(question="  中国首都在哪里  ")
    assert request.question == "中国首都在哪里"


def test_search_request_rejects_empty_question() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(question="   ")


def test_search_request_accepts_knowledge_base_filter() -> None:
    request = SearchRequest(question="报销流程", knowledge_base_id="finance")
    assert request.knowledge_base_id == "finance"
