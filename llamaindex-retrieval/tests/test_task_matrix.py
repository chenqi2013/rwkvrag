import json
import asyncio

import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline
from llamaindex_retrieval.schemas import SearchRequest, SourceItem
from llamaindex_retrieval.task_matrix import Assessment, Followup, Review, TaskPlan, cell_review_prompt, matrix_writer_prompt, parse_model
from test_rwkv_pipeline import FakeIndex, hit, native_result


PLAN = {"objects": ["A", "B"], "dimensions": ["capacity"], "conditions": [], "cells": [
    {"id": "a", "object": "A", "dimension": "capacity", "question": "A capacity"},
    {"id": "b", "object": "B", "dimension": "capacity", "question": "B capacity"}]}


@pytest.mark.parametrize("value,schema", [
    ({**PLAN, "cells": PLAN["cells"][:1]}, TaskPlan),
    ({**PLAN, "cells": PLAN["cells"] * 2}, TaskPlan),
    ({"status": "missing", "source_ids": ["invented"]}, Assessment),
    ({"status": "supported", "source_ids": []}, Assessment),
    ({"stop": True, "queries": [{"cell_id": "b", "query": "q"}]}, Followup),
    ({"valid": True, "issues": ["wrong source"]}, Review),
    ({"valid": "false", "issues": []}, Review),
])
def test_reject_invalid_contracts(value, schema):
    with pytest.raises(ValueError):
        parse_model(json.dumps(value), schema)


def test_duplicate_keys_are_never_overwritten():
    with pytest.raises(ValueError):
        parse_model('{"valid":true,"valid":false,"issues":[]}', Review)


def test_separately_requested_attributes_do_not_require_unrequested_cross_product():
    value = {"coverage": "listed", "objects": ["A", "B"], "dimensions": ["power", "cooling"],
        "conditions": [], "cells": [
            {"id": "a", "object": "A", "dimension": "power", "question": "A power"},
            {"id": "b", "object": "B", "dimension": "cooling", "question": "B cooling"}]}
    assert len(parse_model(json.dumps(value), TaskPlan).cells) == 2
    with pytest.raises(ValueError):
        parse_model(json.dumps({**value, "coverage": "grid"}), TaskPlan)
    with pytest.raises(ValueError):
        parse_model(json.dumps({**value, "objects": ["A", "B", "forgotten"]}), TaskPlan)


def test_internal_cell_ids_are_not_writer_citations_and_missing_bindings_fail():
    source = SourceItem(id="s", document_id="d", source="test", title="记录3", score=1, snippet="A8L")
    row = {**PLAN["cells"][0], "id": "internal-cell-must-not-be-cited", "status": "supported", "source_ids": ["s"]}
    prompt = matrix_writer_prompt("task", [row], [source])
    assert "internal-cell-must-not-be-cited" not in prompt
    assert '"source_labels": ["[资料 1]"]' in prompt
    with pytest.raises(ValueError, match="unavailable evidence"):
        matrix_writer_prompt("task", [row], [])


def test_cell_review_keeps_global_citation_numbers_and_all_reader_selected_conflicts():
    sources = [SourceItem(id=str(i), document_id=str(i), source="test", title="record", score=1,
        snippet=text) for i, text in enumerate(["unrelated B12L", "A8L", "A9L conflicting record"])]
    row = {**PLAN["cells"][0], "source_ids": ["1", "2"], "assessment_source_ids": ["1"]}
    prompt = cell_review_prompt("compare", row, "original answer", sources)
    assert "unrelated B12L" not in prompt
    assert "A8L" in prompt and "A9L conflicting record" in prompt
    assert '"label": "资料 2"' in prompt and '"label": "资料 3"' in prompt
    assert '"label": "资料 1"' not in prompt


class MatrixModel:
    def __init__(self, *, reject_b=False, review_fail=False, bad_assessment=False):
        self.calls = []
        self.reject_b, self.review_fail, self.bad_assessment = reject_b, review_fail, bad_assessment

    async def complete(self, messages, **kwargs):
        prompt = messages[0]["content"]
        self.calls.append((prompt, kwargs))
        stage = kwargs["stage"]
        if "核验一个回答项目" in prompt:
            raw = {"answer": "NO" if self.review_fail else "YES"}
        elif stage == "resolver":
            raw = {"answer": "NO" if self.reject_b and kwargs["evidence_ids"] == ("b",) else "YES"}
        elif stage == "writer":
            # Even deliberately invalid labels must not be changed by code.
            return native_result(stage, ">思考</think>完整模型输出[资料 99]")
        elif "制定检索计划" in prompt:
            raw = PLAN
        elif "核对一个检索项目" in prompt:
            ids = list(kwargs["evidence_ids"])
            raw = {"status": "supported", "source_ids": ["unknown"] if self.bad_assessment else ids}
        elif "下一轮检索" in prompt:
            raw = {"stop": False, "queries": [{"cell_id": "b", "query": "B revised capacity"}]}
        elif "检查答案" in prompt:
            raw = {"valid": not self.review_fail, "issues": ["wrong source"] if self.review_fail else []}
        else:
            raise AssertionError(prompt)
        return native_result(stage, ">思考</think>" + json.dumps(raw))


def settings(**kw):
    return Settings(native_task_matrix_enabled=True, native_resolver_protocol="binary_query",
        native_task_source="queries", native_resolver_task_grouping="individual", **kw)


@pytest.mark.asyncio
async def test_every_cell_has_its_own_reader_query_and_source_binding():
    model = MatrixModel()
    index = FakeIndex({"A capacity": [hit("a", "A 8L")], "B capacity": [hit("b", "B 12L")]})
    result = await RWKVPipeline(settings(), index, model).ask(SearchRequest(question="Compare A B", knowledge_base_id="private"))
    assert result.generation["status"] == "answer_quality_failed"
    assert result.generation["quality_failure_reason"] == "citation_syntax_or_identity"
    assert result.retrieval["stop_reason"] == "all_cells_supported_by_model"
    rows = result.retrieval["task_matrix"]
    assert len(rows) == 2 and all(row["status"] == "supported" for row in rows)
    assert rows[0]["source_ids"] != rows[1]["source_ids"]
    assert all(call[2] == "private" for call in index.calls)
    readers = [p for p, k in model.calls if k["stage"] == "resolver" and "核验一个回答项目" not in p]
    assert len(readers) == 2
    assert result.answer == ">思考</think>完整模型输出[资料 99]"
    assert result.generation["semantic_support_verified"] is False
    assert result.generation["citation_audit"]["unknown_label_ids"] == [99]


@pytest.mark.asyncio
async def test_followup_only_missing_cell_and_writer_never_sees_rejected_candidate():
    model = MatrixModel(reject_b=True)
    index = FakeIndex({"A capacity": [hit("a", "A 8L")], "B capacity": [hit("b", "rejected secret")],
                       "B revised capacity": [hit("new-b", "B revised 12L")]})
    result = await RWKVPipeline(settings(), index, model).ask(SearchRequest(question="compare"))
    assert [c[0] for c in index.calls] == ["A capacity", "B capacity", "B revised capacity"]
    assert len(result.retrieval["rounds"]) == 2
    writer = next(p for p, k in model.calls if k["stage"] == "writer")
    assert "rejected secret" not in writer
    assert "B revised 12L" in writer


@pytest.mark.asyncio
async def test_review_failure_retries_writer_once_keeps_both_answers_and_fails_truthfully():
    model = MatrixModel(review_fail=True)
    index = FakeIndex({"A capacity": [], "B capacity": [], "B revised capacity": []})
    result = await RWKVPipeline(settings(native_matrix_answer_repairs=1), index, model).ask(SearchRequest(question="compare"))
    writers = [c for c in result.generation["model_calls"] if c["stage"] == "writer"]
    assert len(writers) == 2
    assert all(c["raw_text"] == result.answer for c in writers)
    assert result.generation["status"] == "answer_quality_failed"


@pytest.mark.asyncio
async def test_default_binary_review_rejection_keeps_first_answer_without_speculative_rewrite():
    model = MatrixModel(review_fail=True)
    index = FakeIndex({"A capacity": [], "B capacity": [], "B revised capacity": []})
    result = await RWKVPipeline(settings(), index, model).ask(SearchRequest(question="compare"))
    writers = [c for c in result.generation["model_calls"] if c["stage"] == "writer"]
    assert len(writers) == 1
    assert result.answer == writers[0]["raw_text"]
    assert result.generation["status"] == "answer_quality_failed"


@pytest.mark.asyncio
async def test_unknown_assessment_sources_are_not_accepted():
    model = MatrixModel(bad_assessment=True)
    index = FakeIndex({"A capacity": [hit("a", "A8")], "B capacity": [hit("b", "B12")],
                       "B revised capacity": []})
    result = await RWKVPipeline(settings(native_matrix_max_rounds=1), index, model).ask(SearchRequest(question="compare"))
    assert all(row["assessment_status"] == "failed" for row in result.retrieval["task_matrix"])
    assert result.generation["status"] == "matrix_partial_failure"


@pytest.mark.asyncio
async def test_reader_budget_is_global_and_bounded():
    model = MatrixModel()
    index = FakeIndex({"A capacity": [hit("a", "A8")], "B capacity": [hit("b", "B12")]})
    result = await RWKVPipeline(settings(native_matrix_max_reader_calls=1), index, model).ask(SearchRequest(question="compare"))
    assert result.retrieval["reader_calls"] == 1
    assert result.retrieval["stop_reason"] == "reader_budget"
    assert sum(k["stage"] == "resolver" and "核验一个回答项目" not in p for p, k in model.calls) == 1


@pytest.mark.asyncio
async def test_timeout_keeps_started_prompt_and_cancellation_trace():
    class Slow:
        async def complete(self, *args, **kwargs):
            await asyncio.sleep(1)
    limited = settings().model_copy(update={"native_matrix_timeout_seconds": 0.02})
    result = await RWKVPipeline(limited, FakeIndex({}), Slow()).ask(SearchRequest(question="compare"))
    assert result.generation["status"] == "timeout"
    event = result.generation["model_calls"][0]
    assert event["purpose"] == "matrix_plan"
    assert event["status"] == "cancelled"
    assert "compare" in event["prompt"]


@pytest.mark.asyncio
async def test_total_retrieval_failure_does_not_generate_an_answer():
    model = MatrixModel()
    # FakeIndex raises for unrecognized queries, simulating unavailable retrieval.
    result = await RWKVPipeline(settings(), FakeIndex({}), model).ask(SearchRequest(question="compare"))
    assert result.generation["status"] == "retrieval_failed"
    assert not any(k["stage"] == "writer" for _, k in model.calls)


@pytest.mark.asyncio
async def test_assessor_cannot_hide_reader_selected_conflicting_record():
    class Narrow(MatrixModel):
        async def complete(self, messages, **kwargs):
            if "核对一个检索项目" in messages[0]["content"]:
                return native_result("planner", ">思考</think>" + json.dumps({
                    "status": "supported", "source_ids": list(kwargs["evidence_ids"][:1])}))
            return await super().complete(messages, **kwargs)
    model = Narrow()
    index = FakeIndex({"A capacity": [hit("a", "A8L"), hit("conflict", "A9L conflicting record")],
                       "B capacity": [hit("b", "B12L")]})
    result = await RWKVPipeline(settings(), index, model).ask(SearchRequest(question="compare"))
    row = result.retrieval["task_matrix"][0]
    assert len(row["source_ids"]) == 2 and len(row["assessment_source_ids"]) == 1
    writer = next(p for p, k in model.calls if k["stage"] == "writer")
    assert "A8L" in writer and "A9L conflicting record" in writer


@pytest.mark.asyncio
async def test_web_quota_defers_unsearched_objects_without_replanning_or_claiming_search():
    class WebPipeline(RWKVPipeline):
        queries = []

        async def retrieve_groups(self, request, queries, candidate_k):
            self.queries.append(queries)
            return [[hit(query, query + " 8L")] for query in queries], {}

    model = MatrixModel()
    pipeline = WebPipeline(settings(web_search_max_queries=1), FakeIndex({}), model)
    result = await pipeline.ask(SearchRequest(question="compare", retrieval_mode="web"))
    assert pipeline.queries == [["A capacity"], ["B capacity"]]
    assert result.retrieval["rounds"][0]["deferred_queries"] == [{"cell_id": "b", "query": "B capacity"}]
    assert not any("下一轮检索" in prompt for prompt, _ in model.calls)
    assert all(row["source_ids"] for row in result.retrieval["task_matrix"])


@pytest.mark.asyncio
async def test_physical_index_is_pinned_for_request_without_mutating_shared_alias():
    from types import SimpleNamespace
    from llamaindex_retrieval.lexical_index import LexicalIndex

    class Index(LexicalIndex):
        def __init__(self):
            self.index_name = "active-alias"
            self.versions = SimpleNamespace(current=lambda: "physical-v7")
            self.calls = []

        def search_chunks(self, query, *, candidate_k, knowledge_base_id=None):
            self.calls.append((self.index_name, knowledge_base_id))
            return [hit(query, query + "8L")]

    index = Index()
    pipeline = RWKVPipeline(settings(), index, MatrixModel())
    result = await pipeline.ask(SearchRequest(question="compare", knowledge_base_id="private"))
    assert index.index_name == "active-alias"
    assert pipeline.index is index
    assert index.calls == [("physical-v7", "private")] * 2
    assert result.retrieval["index_version"] == "physical-v7"
    assert result.retrieval["index_snapshot"] == "physical_index_pinned_not_point_in_time"
