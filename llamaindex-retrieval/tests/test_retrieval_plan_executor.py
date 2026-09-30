import json

import pytest

from llamaindex_retrieval.retrieval_plan import RetrievalPlanV1
from llamaindex_retrieval.retrieval_plan_executor import Document, execute_plan, parse_read


def plan():
    return RetrievalPlanV1(coverage="grid", objects=["A", "B", "C"],
                           dimensions=["安装", "授权", "备份"], conditions=["离线"],
                           listed_pairs=[], initial_queries=[
                               {"object": obj, "query": obj + " 安装"} for obj in ("A", "B", "C")])


async def search(query):
    return [Document(query, "a complete source paragraph", "https://example.invalid/doc")]


async def read(cell, documents):
    return json.dumps({"status": "supported", "quotes": [
        {"source_id": documents[0].id, "quote": documents[0].text}]})


@pytest.mark.asyncio
async def test_three_queries_keep_all_nine_cells():
    result = await execute_plan(plan(), search, read, max_searches=3)
    assert result["counts"]["searches"] == 3
    assert result["counts"]["reads"] == 9
    assert len(result["cells"]) == 9 and result["pending_cell_ids"] == []
    assert [r["object"] for r in result["trace"] if r["stage"] == "search"] == ["A", "B", "C"]
    assert not result["semantic_quality_verified"]
    json.dumps(result)


@pytest.mark.asyncio
async def test_budget_does_not_delete_later_object_or_claim_no_evidence():
    result = await execute_plan(plan(), search, read, max_searches=2)
    pending = [c for c in result["cells"] if c["object"] == "C"]
    assert len(result["cells"]) == 9
    assert len(pending) == 3
    assert {c["state"] for c in pending} == {"not_searched"}
    assert result["pending_cell_ids"] == ["c7", "c8", "c9"]


@pytest.mark.asyncio
async def test_read_budget_keeps_unread_state():
    result = await execute_plan(plan(), search, read, max_reads=1)
    assert len(result["pending_cell_ids"]) == 8
    assert {c["state"] for c in result["cells"][1:]} == {"not_read"}


@pytest.mark.asyncio
async def test_provider_failure_is_distinct_and_exception_text_is_not_logged():
    async def failing(query):
        if query.startswith("B"):
            raise RuntimeError("PRIVATE_CREDENTIAL")
        return await search(query)
    result = await execute_plan(plan(), failing, read)
    assert {c["state"] for c in result["cells"] if c["object"] == "B"} == {"search_error"}
    assert "PRIVATE_CREDENTIAL" not in json.dumps(result)
    assert result["counts"]["reads"] == 6


@pytest.mark.asyncio
async def test_invented_reference_rejects_decision_and_keeps_raw():
    raw = '{"status":"supported","quotes":[{"source_id":"invented","quote":"a"}]}'
    async def bad_reader(cell, documents):
        return raw
    result = await execute_plan(plan(), search, bad_reader)
    assert {c["state"] for c in result["cells"]} == {"invalid_read_output"}
    assert all(not c["evidence"] for c in result["cells"])
    assert all(r["raw"] == raw for r in result["trace"] if r["stage"] == "read")


@pytest.mark.asyncio
async def test_followup_uses_execution_state_and_resolves_only_target_cell():
    calls = []
    async def conditional_read(cell, documents):
        if cell["id"] == "c1" and documents[0].id == "A 安装":
            return '{"status":"insufficient","quotes":[]}'
        return await read(cell, documents)
    async def followup(pending, attempts):
        calls.append(pending)
        assert len(attempts) == 3
        assert pending[0]["state"] == "read_without_verified_evidence"
        return '{"action":"search","queries":[{"object":"A","query":"A 离线安装文档","cell_ids":["c1"]}]}'
    result = await execute_plan(plan(), search, conditional_read, followup)
    assert len(calls) == 1 and len(calls[0]) == 1
    assert result["pending_cell_ids"] == []
    assert result["counts"]["searches"] == 4 and result["counts"]["reads"] == 10


@pytest.mark.asyncio
async def test_repeated_followup_query_is_not_sent_again():
    async def empty_read(cell, documents):
        return '{"status":"insufficient","quotes":[]}'
    async def repeat(pending, attempts):
        return '{"action":"search","queries":[{"object":"A","query":"A 安装","cell_ids":["c1"]}]}'
    result = await execute_plan(plan(), search, empty_read, repeat)
    assert result["counts"]["searches"] == 3
    assert len(result["pending_cell_ids"]) == 9
    assert result["stop_reason"] == "followup_invalid_or_failed"


@pytest.mark.asyncio
async def test_changed_source_identity_is_not_silently_overwritten():
    async def collision(query):
        return [Document("same-id", query, "https://example.invalid/doc")]
    result = await execute_plan(plan(), collision, read)
    assert len(result["sources"]) == 1
    assert result["sources"][0]["text"] == "A 安装"
    assert result["pending_cell_ids"] == ["c4", "c5", "c6", "c7", "c8", "c9"]


@pytest.mark.asyncio
async def test_callback_cannot_mutate_authoritative_conditions_or_sources():
    async def mutate(cell, documents):
        cell["conditions"].clear()
        documents.append(Document("forged", "forged text", ""))
        return '{"status":"supported","quotes":[{"source_id":"forged","quote":"forged text"}]}'
    result = await execute_plan(plan(), search, mutate)
    assert all(c["conditions"] == ["离线"] for c in result["cells"])
    assert all(c["state"] == "invalid_read_output" for c in result["cells"])


def test_duplicate_model_keys_rejected():
    with pytest.raises(ValueError, match="duplicate JSON"):
        parse_read('{"status":"supported","status":"insufficient","quotes":[]}', [])
