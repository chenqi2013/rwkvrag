"""Reader handoff + retained-byte binding only; canned model, no quality claims."""
import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from llama_index.core import Document

from llamaindex_retrieval import parsers
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.dependencies import repository
from llamaindex_retrieval.lexical_index import LexicalResult
from llamaindex_retrieval.native_rwkv import NativeRWKVResult
from llamaindex_retrieval.routers import admin
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline, evidence_units, source_from_hit
from llamaindex_retrieval.source_catalog import check_binding
from llamaindex_retrieval.source_revisions import prepare_revision, sha256
from llamaindex_retrieval.verbatim_chunking import verbatim_nodes


def digest(text):
    return sha256(text.encode())


@pytest.fixture(params=["\n", "\r\n"])
def retained(tmp_path, monkeypatch, request):
    newline = request.param
    text = "# 原件😀" + newline + newline + ("α你😀e\u0301 repeated clause; " + newline) * 55
    document = Document(text=text, id_="doc-a", metadata={"document_id": "doc-a", "title": "fixture"})
    # Controlled parser output keeps CRLF in the parsed text. This is not a
    # parser quality test. Actual revision writer and verbatim chunker run.
    monkeypatch.setattr(parsers, "parse_uploaded_file", lambda *a, **kw: [document.model_copy(deep=True)])
    settings = Settings(_env_file=None, upload_dir=tmp_path / "uploads", native_resolver_sources=2,
        native_resolver_window_characters=512, native_resolver_overlap_characters=80,
        native_resolver_batch_characters=8000, native_writer_pipeline="single",
        native_writer_budget_policy="disabled", native_writer_prompt_protocol="task_first",
        native_preserve_original_question=False, native_resolver_protocol="fields",
        native_task_source="fields", native_plan_protocol="queries_fields")
    path = tmp_path / "original.md"
    path.write_bytes(text.encode())
    documents, revision = prepare_revision(settings, path, "file-a", "kb-a")
    parent = max(verbatim_nodes(documents[0]), key=lambda node: len(node.text))
    hit = LexicalResult(node_id=parent.node_id, document_id="doc-a", text=parent.text,
                        metadata=copy.deepcopy(parent.metadata), score=1)
    item = {"id": "file-a", "knowledge_base_id": "kb-a", "sha256": revision["source_sha256"]}
    source = source_from_hit(hit)
    return SimpleNamespace(settings=settings, hit=hit, source=source, item=item, path=path, text=text)


def selection(parent, start, end):
    text = parent.snippet[start:end]
    return parent.model_copy(deep=True, update={
        "id": f"{parent.id}@{start}:{end}:{digest(text)[:12]}", "snippet": text,
        "metadata": {**copy.deepcopy(parent.metadata), "parent_source_id": parent.id,
            "parent_text_sha256": digest(parent.snippet), "span_start": start, "span_end": end,
            "span_sha256": digest(text), "offset_unit": "unicode_characters_in_indexed_chunk",
            "selection_scope": "fields", "field_ids": ["f1"]}})


@pytest.mark.asyncio
async def test_public_native_selection_binds_without_exposing_unselected_text_or_rewriting(retained):
    env = retained
    calls = []
    answer = ">原始思考</think>\n原样保留。[资料 99]\r\n"
    class Index:
        def search_chunks(self, query, **kwargs):
            return [env.hit, LexicalResult(node_id="unselected", document_id="other", score=0.5,
                        text="UNSELECTED_ONLY_SOURCE", metadata={"title": "unselected"})]
    class Model:
        async def complete(self, messages, *, stage, evidence_ids=(), **kwargs):
            prompt = messages[0]["content"]
            calls.append({"stage": stage, "prompt": prompt, "ids": list(evidence_ids)})
            if stage == "planner":
                raw = '>fixed</think>\n{"queries":["fixture"],"fields":["field"]}'
            elif stage == "resolver":
                raw = ">fixed</think>\nf1: " + ("E2" if evidence_ids[0] == env.source.id else "NONE")
            else:
                raw = answer
            return NativeRWKVResult(status="completed", raw_text=raw, finish_reason="stop",
                trace={"stage": stage, "status": "completed", "raw_text": raw})
    from llamaindex_retrieval.schemas import SearchRequest
    response = await RWKVPipeline(env.settings, Index(), Model()).ask(SearchRequest(question="fixture task"))
    units = list(evidence_units(0, env.source.snippet, 512, 80))
    expected = selection(env.source, units[1].start, units[1].end)
    expected.score = 1 / 61  # One first-ranked hit in one query, unchanged RRF.
    assert len(response.sources) == 1 and response.sources[0].model_dump() == expected.model_dump()
    writer = next(call for call in calls if call["stage"] == "writer")
    assert writer["ids"] == [expected.id]
    assert "UNSELECTED_ONLY_SOURCE" not in writer["prompt"]
    assert env.source.snippet not in writer["prompt"]
    assert json.loads(writer["prompt"].split("逐字证据：", 1)[1])[0]["text"] == expected.snippet
    assert response.answer == response.generation["raw_model_answer"] == answer
    assert response.generation["citation_map"] == {"1": expected.id}
    before = response.model_dump()
    row = check_binding(env.settings, response.sources[0], env.item)
    assert row["status"] == "bound"
    assert row["binding_scope"] == "resolver_selection"
    start = env.source.metadata["source_span"]["start"] + units[1].start
    assert row["bound_document_span"] == {"start": start, "end": start + len(expected.snippet),
        "unit": "unicode_code_points", "sha256": digest(expected.snippet)}
    assert env.text[start:row["bound_document_span"]["end"]] == expected.snippet
    assert response.model_dump() == before


@pytest.mark.parametrize("whole", [False, True])
def test_plain_chunk_and_explicit_whole_selection_remain_distinct(retained, whole):
    env = retained
    source = selection(env.source, 0, len(env.source.snippet)) if whole else env.source
    row = check_binding(env.settings, source, env.item)
    assert row["status"] == "bound"
    assert row["binding_scope"] == ("resolver_selection" if whole else "indexed_chunk")
    assert row["bound_document_span"] == env.source.metadata["source_span"]


@pytest.mark.parametrize("key,value,status", [
    ("parent_source_id", "", "invalid_selection_parent_id"),
    ("parent_text_sha256", "0" * 64, "selection_parent_hash_mismatch"),
    ("span_start", True, "invalid_selection_span"),
    ("span_start", -1, "invalid_selection_span"),
    ("span_start", "2", "invalid_selection_span"),
    ("span_end", 100000, "invalid_selection_span"),
    ("span_end", 2, "invalid_selection_span"),
    ("span_end", 10.5, "invalid_selection_span"),
    ("offset_unit", "utf8_bytes", "invalid_selection_span"),
    ("span_sha256", "0" * 64, "selection_hash_mismatch"),
])
def test_selected_lineage_is_not_coerced_or_repaired(retained, key, value, status):
    env = retained
    source = selection(env.source, 2, 25)
    source.metadata[key] = value
    before = source.model_dump()
    row = check_binding(env.settings, source, env.item)
    assert row["status"] == status and not row["chunk_verified"]
    assert row["original_url"] is None and row["bound_document_span"] is None
    assert source.model_dump() == before


@pytest.mark.parametrize("missing", ["parent_source_id", "parent_text_sha256", "span_start", "span_end",
                                      "span_sha256", "offset_unit"])
def test_partial_selection_metadata_cannot_fall_back_to_full_chunk_binding(retained, missing):
    env = retained
    source = selection(env.source, 0, len(env.source.snippet))
    del source.metadata[missing]
    row = check_binding(env.settings, source, env.item)
    assert row["status"] == "incomplete_selection_metadata" and row["original_url"] is None


def test_corrupt_parent_hashes_are_not_replaced_by_child_hashes(retained):
    env = retained
    source = selection(env.source, 2, 25)
    source.metadata["chunk_text_sha256"] = digest(source.snippet)
    source.metadata["source_span"]["sha256"] = digest(source.snippet)
    assert check_binding(env.settings, source, env.item)["status"] == "chunk_hash_mismatch"


def test_offset_binding_does_not_search_for_another_repeated_occurrence(retained):
    env = retained
    source = selection(env.source, 2, 25)
    source.metadata["span_start"] = 3
    source.metadata["span_end"] = 26
    assert check_binding(env.settings, source, env.item)["status"] == "selection_snippet_mismatch"
    source = selection(env.source, 2, 25)
    source.id = "unrelated-id"
    assert check_binding(env.settings, source, env.item)["status"] == "selection_identity_mismatch"


@pytest.mark.asyncio
async def test_read_only_selected_binding_route_keeps_original_revision_and_all_rows(retained, monkeypatch):
    env = retained
    good = selection(env.source, 2, 25)
    bad = good.model_copy(deep=True)
    bad.snippet += " invented"
    env.path.unlink()  # Only the retained old revision is available.
    calls = []
    class Repo:
        async def get_file(self, identity):
            calls.append(identity)
            return {**env.item, "sha256": "0" * 64}  # Catalog's current hash is not the old revision.
    app = FastAPI()
    app.include_router(admin.router)
    app.dependency_overrides[repository] = Repo
    monkeypatch.setattr(admin, "get_settings", lambda: env.settings)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/admin/source-catalog/audit", json={"sources": [good.model_dump(), bad.model_dump()]})
        assert response.status_code == 200
        audit = response.json()
        assert not audit["all_bound"] and calls == ["file-a"]
        assert [r["status"] for r in audit["results"]] == ["bound", "selection_snippet_mismatch"]
        assert not audit["context_verified"] and not audit["semantic_verified"] and not audit["index_membership_verified"]
        original = await client.get(audit["results"][0]["original_url"])
        assert original.status_code == 200 and original.content == env.text.encode()
