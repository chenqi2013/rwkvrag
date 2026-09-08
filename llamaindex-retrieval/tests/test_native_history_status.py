"""Native history records execution state without inferring answer semantics."""
from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from llamaindex_retrieval.dependencies import repository
from llamaindex_retrieval.repository import (
    MongoRepository,
    search_answer_status,
    search_failure_category,
    search_failure_reason,
)
from llamaindex_retrieval.routers.admin import router
from llamaindex_retrieval.schemas import SearchTestItem


def native(status, answer=">未改动的思考</think>原始答案🙂"):
    return {"answer": answer, "sources": [], "retrieval": {"returned": 0},
            "generation": {"pipeline": "rwkv", "status": status,
                           "raw_model_answer": answer,
                           "citation_audit": {"semantic_support_verified": False}}}


@pytest.mark.parametrize("status,expected,category", [
    ("planner_failed", "failed", "retrieval_failed"),
    ("retrieval_failed", "failed", "retrieval_failed"),
    ("resolver_partial_failure", "partial", "evidence_extraction_failed"),
    ("invalid_materials", "failed", "evidence_extraction_failed"),
    ("length", "failed", "generation_failed"),
    ("timeout", "failed", "generation_failed"),
    ("budget_exceeded", "failed", "generation_failed"),
    ("http_error", "failed", "generation_failed"),
    ("transport_error", "failed", "generation_failed"),
    ("invalid_request", "failed", "generation_failed"),
    ("invalid_response", "failed", "generation_failed"),
    ("cancelled", "failed", "generation_failed"),
])
def test_native_failure_summary_comes_from_runtime_status_not_answer(status, expected, category):
    response = native(status)
    before = deepcopy(response)
    assert search_answer_status(response) == expected
    assert search_failure_category(response) == category
    assert search_failure_reason(response) == f"native_{status}"
    assert response == before


@pytest.mark.parametrize("answer", ["", "资料不足。", "根据检索到的资料，无法确定。", "原始答案"])
def test_completed_without_sources_is_not_inferred_to_be_verified_or_refused(answer):
    response = native("completed", answer)
    response["generation"]["failure_category"] = "data_missing"
    response["generation"]["failure_reason"] = "stale legacy diagnosis"
    before = deepcopy(response)
    assert search_answer_status(response) == "completed"
    assert search_failure_category(response) is None
    assert search_failure_reason(response) is None
    assert response == before
    assert response["generation"]["citation_audit"]["semantic_support_verified"] is False


@pytest.mark.parametrize("status", [None, "unrecognized", [], {}])
def test_missing_or_malformed_native_status_cannot_be_marked_answered(status):
    response = native(status)
    assert search_answer_status(response) == "failed"
    assert search_failure_category(response) == "generation_failed"
    assert search_failure_reason(response) == (
        "native_status_missing" if status is None else "native_status_unknown"
    )


@pytest.mark.parametrize("generation", [None, {}, {"pipeline": "existing", "status": "timeout"}])
def test_existing_pipeline_keeps_its_exact_previous_summary_behavior(generation):
    response = {"answer": "  根据检索到的资料，无法确定。  ", "generation": generation}
    assert search_answer_status(response) == "refused"
    response["answer"] = ""
    assert search_answer_status(response) == "answered"
    response["generation"] = {"failure_category": "data_missing", "failure_reason": "legacy reason"}
    assert search_failure_category(response) == "data_missing"
    assert search_failure_reason(response) == "legacy reason"


@pytest.mark.parametrize("state", ["answered", "refused", "completed", "partial", "failed"])
def test_native_states_are_supported_by_history_schema_and_admin_filter(state):
    now = datetime.now(timezone.utc)
    item = SearchTestItem(id="test", question="question", request={}, run_count=1,
                          created_at=now, updated_at=now, latest_answer_status=state)
    assert item.latest_answer_status == state
    app = FastAPI()
    app.include_router(router)
    repo = AsyncMock()
    repo.list_search_tests.return_value = {"items": [], "total": 0, "page": 1, "page_size": 20}
    app.dependency_overrides[repository] = lambda: repo
    result = TestClient(app).get("/v1/admin/search-history", params={"answer_status": state})
    assert result.status_code == 200
    assert repo.list_search_tests.await_args.kwargs["answer_status"] == state


def test_history_filter_does_not_offer_a_semantic_verified_state():
    app = FastAPI()
    app.include_router(router)
    repo = AsyncMock()
    app.dependency_overrides[repository] = lambda: repo
    result = TestClient(app).get("/v1/admin/search-history", params={"answer_status": "verified"})
    assert result.status_code == 422
    repo.list_search_tests.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("status,expected", [
    ("planner_failed", "failed"), ("length", "failed"),
    ("resolver_partial_failure", "partial"), ("completed", "completed"),
])
async def test_future_history_write_stores_runtime_summary_without_modifying_run(status, expected):
    repo = MongoRepository.__new__(MongoRepository)
    repo.search_tests = AsyncMock()
    repo.search_test_runs = AsyncMock()
    repo.search_tests.find_one_and_update.return_value = {"id": "history-item", "run_count": 1}
    response = native(status)
    request = {"question": "当前需求", "history": [{"role": "user", "content": "撤回原比较"}]}
    original = deepcopy({"request": request, "response": response})
    run = await repo.record_search_test_run(request, response)
    summary = repo.search_tests.update_one.await_args.args[1]["$set"]
    assert summary["latest_answer_status"] == expected
    assert summary["latest_failure_category"] == search_failure_category(response)
    assert summary["latest_failure_reason"] == search_failure_reason(response)
    assert run["response"] == original["response"]
    assert run["request"] == original["request"]
    assert response == original["response"]
    assert request == original["request"]
    assert repo.search_test_runs.insert_one.await_args.args[0]["response"] == original["response"]
