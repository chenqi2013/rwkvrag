"""Offline protocol/transport tests, not retrieval relevance or model quality scores."""
import asyncio
from collections import Counter
import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.lexical_index import LexicalResult
from llamaindex_retrieval.native_rwkv import NativeRWKVClient, NativeRWKVResult
from llamaindex_retrieval.schemas import ConversationMessage, SearchRequest, SourceItem
from llamaindex_retrieval.service import SearchService
from llamaindex_retrieval.unified_contracts import Reading, digest, parse, reduction_calls
from llamaindex_retrieval.unified_materials import EvidenceCatalog, SourceAdapters
from llamaindex_retrieval.unified_retrieval import UnifiedRetrievalPipeline
from llamaindex_retrieval.web_retrieval import WebSearchAdapter


def settings(**limits):
    return Settings(_env_file=None, native_unified=limits,
        native_completion_protocol="g1j_plain", native_require_model_identity=True,
        native_planner_prefill="<think></think", native_resolver_prefill="<think></think",
        native_writer_prefill="<think></think")


def material(index=0, text=None):
    return SourceItem(id=f"s{index}", document_id=f"d{index}", source="fixture",
                      title=f"source{index}", score=1, snippet=text or f"原文编号{index}。")


class Model:
    def __init__(self, *, mode="provided", channels=(), overrides=None):
        self.mode, self.channels = mode, list(channels)
        self.overrides = overrides or {}
        self.calls = []
        self.counts = Counter()
        self.closed = False

    async def complete(self, messages, **options):
        purpose = options["stage"].removeprefix("unified_")
        data = json.loads(messages[0]["content"].split("\n输入数据：\n", 1)[1])
        self.counts[purpose] += 1
        call = {"purpose": purpose, "data": data, "options": options}
        self.calls.append(call)
        event = options["trace"]
        event.update(call_id=str(uuid4()), status="pending")
        override = self.overrides.get((purpose, self.counts[purpose]), self.overrides.get(purpose))
        if callable(override):
            override = override(data)
            if asyncio.iscoroutine(override):
                override = await override
        status = "completed"
        if isinstance(override, tuple):
            status, value = override
        elif override is not None:
            value = override
        elif purpose == "task":
            last = data["latest_user_message"]
            value = {"goals": [{"id": "G1", "text": last["content"],
                "refs": [{"message_id": last["id"], "quote": last["content"]}]}],
                "constraints": [], "retracted": [], "unresolved": []}
        elif purpose == "direction":
            value = {"choice": ("hybrid" if len(self.channels) == 2 else self.channels[0])
                     if self.mode == "retrieve" else self.mode}
        elif purpose == "plan":
            value = {"actions": [{"channel": self.channels[0], "query": f"query{self.counts[purpose]}",
                                  "goal_ids": ["G1"]}], "unplanned_goal_ids": []}
        elif purpose == "review":
            value = {"next": "stop", "missing": [], "reason": "fixture"}
        elif purpose in {"leaf", "merge"}:
            ids = ([data["original_material"]["evidence_id"]] if purpose == "leaf" else
                   list(dict.fromkeys(e for n in data["children"] for e in n["model_selected_evidence_ids"])))
            value = {"answer": "fixture " + " ".join(f"[资料 {e[1:]}]" for e in ids),
                     "evidence_ids": ids, "missing": []}
        else:
            value = "请明确需要检索的对象？\n"
        raw = value if isinstance(value, str) or value is None else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        event.update(raw_text=raw, status=status, completion_attempted=True)
        call["raw"] = raw
        return NativeRWKVResult(status, raw, "length" if status == "length" else "stop", event)

    async def aclose(self):
        self.closed = True


class Adapters:
    def __init__(self, values=None):
        self.values = values or [[material()]]
        self.calls = []

    async def execute(self, action, request):
        self.calls.append((action, request))
        value = self.values[min(len(self.calls) - 1, len(self.values) - 1)]
        if isinstance(value, Exception):
            raise value
        return value, {"provider": action.channel, "status": "completed",
                       "snapshots": ["UNSELECTED_PROVIDER_SNAPSHOT_SENTINEL"]}


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 9, 10, 27])
async def test_real_call_topology_and_immutable_parent_raw(n):
    model = Model()
    engine = UnifiedRetrievalPipeline(settings(max_units=40), None, model=model)
    response = await engine.ask_materials("检索目标", [material(i) for i in range(n)])
    graph = response.generation["nodes"]
    assert model.counts["leaf"] == n
    assert model.counts["merge"] == reduction_calls(n)
    assert len(graph) == n + reduction_calls(n)
    by_id = {node["id"]: node for node in graph}
    for call in model.calls:
        if call["purpose"] == "merge":
            assert 1 <= len(call["data"]["children"]) <= 3
            assert call["options"]["evidence_ids"] == ()
            for child in call["data"]["children"]:
                original = by_id[child["node_id"]]
                assert child["raw_text"] == original["raw_text"]
                assert child["raw_text_sha256"] == digest(original["raw_text"])
            assert all(not binding["original_text_in_this_prompt"]
                       for binding in call["data"]["inherited_bindings_not_original_text"].values())
    if n:
        root = by_id[response.generation["root_node_id"]]
        assert response.answer == response.generation["raw_model_answer"] == root["raw_text"]
        assert response.generation["status"] == "completed"
        assert response.generation["answer_modified"] is False
        assert len(response.sources) == n
        assert response.generation["semantic_support_verified"] is False
    else:
        assert response.answer == "" and response.generation["root_node_id"] is None
        assert response.generation["status"] == "no_read_output"
    if n in {4, 10}:
        assert any(len(node["parents"]) == 1 for node in graph if node["kind"] == "merge")
    await engine.aclose()
    assert model.closed


@pytest.mark.parametrize("raw", ["NONE", '```json\n{"answer":"NO"}\n```',
                                   '{"answer":"a","answer":"b","evidence_ids":[],"missing":[]}'])
async def test_invalid_raw_not_repaired_or_promoted(raw):
    model = Model(overrides={"leaf": raw})
    engine = UnifiedRetrievalPipeline(settings(), None, model=model)
    response = await engine.ask_materials("目标", [material(0), material(1)])
    assert [n["raw_text"] for n in response.generation["nodes"][:2]] == [raw, raw]
    assert response.generation["status"] == "blocked_dependency"
    assert response.answer == "" and model.counts["merge"] == 0
    assert response.sources == []


async def test_one_failed_branch_is_retained_and_valid_branches_remain_partial():
    model = Model(overrides={("leaf", 2): ("length", "未完成的原始输出")})
    engine = UnifiedRetrievalPipeline(settings(), None, model=model)
    response = await engine.ask_materials("目标", [material(i) for i in range(4)])
    assert response.generation["status"] == "partial"
    assert response.generation["nodes"][1]["raw_text"] == "未完成的原始输出"
    assert "E2" not in json.loads(response.answer)["evidence_ids"]
    assert "2" not in response.generation["citation_map"]
    assert set(response.generation["citation_map"]) == {"1", "3", "4"}
    assert any(c["data"]["children"][1]["status"] == "length"
               for c in model.calls if c["purpose"] == "merge" and len(c["data"]["children"]) > 1)


@pytest.mark.parametrize("failed", [("length", "FAILED_BRANCH_SENTINEL"), "FAILED_BRANCH_SENTINEL"])
async def test_failed_raw_retained_in_audit_but_not_given_to_merge(failed):
    model = Model(overrides={("leaf", 2): failed})
    response = await UnifiedRetrievalPipeline(settings(), None, model=model).ask_materials(
        "目标", [material(i) for i in range(4)])
    node = response.generation["nodes"][1]
    assert node["raw_text"] == "FAILED_BRANCH_SENTINEL"
    assert node["raw_text_sha256"] == digest(node["raw_text"])
    assert response.generation["status"] == "partial"
    merges = [call for call in model.calls if call["purpose"] == "merge"]
    assert merges
    for call in merges:
        assert "FAILED_BRANCH_SENTINEL" not in json.dumps(call["data"])
    child = next(child for call in merges for child in call["data"]["children"]
                 if child["node_id"] == node["id"])
    assert child["raw_text"] is None and child["raw_withheld"] is True
    assert child["raw_text_sha256"] == node["raw_text_sha256"]
    assert child["call_id"] == node["call_id"] and child["status"] == node["status"]


@pytest.mark.parametrize("selected", ["E2", "E999"])
async def test_model_selection_is_audited_separately_from_format_admission(selected):
    raw = json.dumps({"answer": "NO_INLINE_CITATION", "evidence_ids": [selected], "missing": []})
    model = Model(overrides={("leaf", 2): raw})
    response = await UnifiedRetrievalPipeline(settings(), None, model=model).ask_materials(
        "目标", [material(0), material(1)])
    node = response.generation["nodes"][1]
    assert node["status"] == "invalid_contract" and node["raw_text"] == raw
    assert node["parsed_model_evidence_ids"] == [selected]
    assert node["evidence_ids"] == []  # admission, not a claim that the model selected nothing
    call = next(c for c in model.calls if c["purpose"] == "merge")
    assert "NO_INLINE_CITATION" not in json.dumps(call["data"])
    assert response.generation["status"] == "partial"


async def test_failed_raw_not_given_to_review_or_reused_as_evidence():
    model = Model(mode="retrieve", channels=["web"], overrides={"leaf": ("length", "FAILED_REVIEW_SENTINEL")})
    response = await UnifiedRetrievalPipeline(settings(), None, model=model, adapters=Adapters()).search(
        SearchRequest(question="目标", retrieval_mode="web"))
    review = next(call["data"] for call in model.calls if call["purpose"] == "review")
    assert "FAILED_REVIEW_SENTINEL" not in json.dumps(review)
    assert review["leaf_outputs"][0]["raw_withheld"] is True
    assert response.results == []
    assert response.retrieval["nodes"][0]["raw_text"] == "FAILED_REVIEW_SENTINEL"


async def test_unknown_evidence_and_citation_do_not_get_rebound():
    raw = '{"answer":"不支持的断言[资料 99]","evidence_ids":["E1"],"missing":[]}'
    model = Model(overrides={"merge": raw})
    response = await UnifiedRetrievalPipeline(settings(), None, model=model).ask_materials(
        "目标", [material(0), material(1)])
    assert response.answer == raw
    assert response.generation["status"] == "invalid_contract"
    assert response.generation["citation_audit"]["unknown_label_ids"] == [99]
    assert response.generation["citation_map"] == {}


async def test_tree_budget_reserved_and_unread_population_not_hidden():
    model = Model()
    response = await UnifiedRetrievalPipeline(settings(max_calls=5), None, model=model).ask_materials(
        "目标", [material(i) for i in range(10)])
    assert model.counts == {"task": 1, "leaf": 3, "merge": 1}
    assert response.generation["status"] == "partial"
    assert len(response.retrieval["evidence_catalog"]["unread_unit_ids"]) == 7
    assert response.generation["request_budget"]["admitted_calls"] == 5
    assert response.generation["root_node_id"] == "N4"


async def test_timeout_does_not_return_most_recent_intermediate_summary():
    async def slow(_):
        await asyncio.sleep(10)
    model = Model(overrides={("merge", 2): slow})
    response = await UnifiedRetrievalPipeline(settings(timeout_seconds=0.03), None, model=model).ask_materials(
        "目标", [material(i) for i in range(10)])
    assert response.generation["status"] == "request_timeout"
    assert response.generation["nodes"][10]["raw_text"]
    assert response.generation["root_node_id"] is None
    assert response.answer == ""
    assert response.generation["model_calls"][-1]["status"] == "cancelled"


@pytest.mark.parametrize("method", ["ask", "search"])
async def test_shared_core_followup_preserves_history_permissions_and_receipts(method):
    model = Model(mode="retrieve", channels=["knowledge_base", "web"], overrides={
        ("review", 1): {"next": "search", "missing": ["尚有缺口"], "reason": "fixture"}})
    adapter = Adapters([[material(0)], [material(1)]])
    engine = UnifiedRetrievalPipeline(settings(), None, model=model, adapters=adapter)
    request = SearchRequest(question="改查当前对象", retrieval_mode="auto", knowledge_base_id="scope",
                            history=[ConversationMessage(role="user", content="此前对象与要求")])
    response = await getattr(engine, method)(request)
    assert len(adapter.calls) == 2
    assert all(req.knowledge_base_id == "scope" for _, req in adapter.calls)
    assert model.counts["leaf"] == 2 and model.counts["review"] == 2
    assert model.counts["merge"] == (1 if method == "ask" else 0)
    for call in model.calls:
        assert call["data"]["history"][0]["content"] == "此前对象与要求"
        assert call["data"]["latest_user_message"]["content"] == "改查当前对象"
        assert "UNSELECTED_PROVIDER_SNAPSHOT_SENTINEL" not in json.dumps(call["data"])
    assert len(response.retrieval["actions"]) == 2
    assert response.retrieval["actions"][0]["receipt"]["snapshots"]


@pytest.mark.parametrize("actions", [
    [{"channel": "web", "query": f"q{i}", "goal_ids": ["G1"]} for i in range(3)],
    [{"channel": "knowledge_base", "query": "q", "goal_ids": ["G1"]}],
    [{"channel": "web", "query": "q", "goal_ids": ["G99"]}],
    [{"channel": "web", "query": "q", "goal_ids": ["G1"]}] * 2,
])
async def test_illegal_plan_fails_instead_of_truncating_inserting_or_fallback(actions):
    model = Model(mode="retrieve", channels=["web"], overrides={
        "plan": {"actions": actions, "unplanned_goal_ids": []}})
    adapter = Adapters()
    result = await UnifiedRetrievalPipeline(settings(), None, model=model, adapters=adapter).ask(
        SearchRequest(question="目标", retrieval_mode="web"))
    assert result.generation["status"] == "plan_failed" and adapter.calls == []
    assert model.counts["leaf"] == 0


async def test_kb_permission_prevents_model_choosing_web():
    model = Model(mode="retrieve", channels=["web"])
    adapter = Adapters()
    response = await UnifiedRetrievalPipeline(settings(), None, model=model, adapters=adapter).ask(
        SearchRequest(question="目标"))  # API default remains knowledge_base
    assert response.generation["status"] == "routing_failed" and not adapter.calls


async def test_provider_failure_is_not_empty_evidence_and_has_no_fallback():
    model = Model(mode="retrieve", channels=["web"])
    adapter = Adapters([RuntimeError("secret API key must not leak")])
    response = await UnifiedRetrievalPipeline(settings(), None, model=model, adapters=adapter).ask(
        SearchRequest(question="目标", retrieval_mode="web"))
    action = response.retrieval["actions"][0]
    assert action["status"] == "failed" and action["error_type"] == "RuntimeError"
    assert "secret API" not in response.model_dump_json()
    assert len(adapter.calls) == 1 and model.counts["leaf"] == 0


async def test_infrastructure_model_failure_stops_further_calls_without_fallback():
    model = Model(overrides={("leaf", 2): ("http_error", None)})
    result = await UnifiedRetrievalPipeline(settings(), None, model=model).ask_materials(
        "目标", [material(i) for i in range(4)])
    assert model.counts["leaf"] == 2 and model.counts["merge"] == 0
    assert result.generation["status"] == "model_failed"
    assert result.answer == "" and result.generation["root_node_id"] is None


def test_full_snapshot_ranking_context_coordinates_and_deduplication():
    text = "# 开头\n\n" + ("ordinary unrelated text " * 300) + "\n\n# 目标段\n\n" + "needle_xyz 的原始内容。"
    corpus = EvidenceCatalog(256)
    order = corpus.add(material(text=text), "needle_xyz", "A1")
    winner = corpus.units[order[0]]
    assert "needle_xyz" in winner.snippet
    assert winner.metadata["source_span"]["start"] > 3000
    for unit in corpus.units.values():
        span = unit.metadata["source_span"]
        assert text[span["start"]:span["end"]] == unit.snippet
        assert digest(unit.snippet) == span["sha256"]
        for context in unit.metadata["context_spans"]:
            assert text[context["start"]:context["end"]] == context["text"]
    count = len(corpus.units)
    corpus.add(material(text=text), "other query", "A2")
    assert len(corpus.units) == count
    assert len(corpus.observations) == 2
    corpus.add(material(text=text + "changed"), "needle_xyz", "A3")
    assert len(corpus.snapshots) == 2
    assert len(corpus.units) > count


@pytest.mark.parametrize("text", ["| A | B |\n|---|---|\n" + "| a | b |\n" * 100,
                                 "- " + "list item " * 100 + "\n",
                                 "```\n" + "code line\n" * 100 + "```\n"])
def test_oversized_structural_atom_not_cut_to_character_budget(text):
    corpus = EvidenceCatalog(256)
    corpus.add(material(text=text), "a", "A1")
    assert len(corpus.units) == 1
    assert next(iter(corpus.units.values())).snippet == text


def test_declared_qa_pair_remains_one_atom_including_its_headings():
    text = "# 问题\n\n" + "问题上下文。" * 90 + "\n\n# 答案\n\n" + "回答原文。" * 90
    source = material(text=text).model_copy(update={"metadata": {"kind": "qa-markdown"}})
    corpus = EvidenceCatalog(256)
    corpus.add(source, "问题", "A1")
    assert len(corpus.units) == 1
    unit = next(iter(corpus.units.values()))
    assert unit.snippet == text and unit.metadata["atomic"] is True
    assert unit.metadata["content_type"] == "qa"


async def test_full_web_adapter_bypasses_prefix_not_shared_guard_or_snapshot_recording(monkeypatch):
    web = WebSearchAdapter(settings())
    text = " " * 3500 + "正文在前缀之后"
    observed = []
    async def execute(request, timeout):
        observed.append(request)
        return {"provider": "tavily", "source_hashes": {}, "hits": [{"url": "https://example.org/doc",
            "reader_text": text[:request["material_characters"]], "snapshot": text,
            "title": "page", "retrieved_at": "time", "published_date": "", "content_status": "fetched",
            "material_limited": False, "score": 1, "error": ""}]}
    monkeypatch.setattr(web, "_execute", execute)
    hits, receipt = await web.search("query", full_snapshot=True)
    assert hits[0].text == text and observed[0]["material_characters"] is None
    assert receipt["snapshots"][0]["text"] == text
    assert observed[0]["max_results"] == 2


async def test_kb_adapter_keeps_scope_and_original_candidate_budget():
    seen = []
    def search_chunks(query, **kwargs):
        seen.append((query, kwargs))
        return [LexicalResult(node_id="id", document_id="doc", text="原文", score=1, metadata={})]
    adapter = SourceAdapters(settings(), SimpleNamespace(search_chunks=search_chunks), None)
    action = SimpleNamespace(channel="knowledge_base", query="模型写的查询")
    values, trace = await adapter.execute(action, SearchRequest(question="原问题", candidate_k=20,
                                                               knowledge_base_id="scope"))
    assert seen == [("模型写的查询", {"candidate_k": 20, "knowledge_base_id": "scope"})]
    assert values[0].snippet == "原文"
    assert trace["snapshot_scope"] == "returned_index_chunks_not_entire_documents"


async def test_real_native_client_checks_tokens_before_completion_without_truncation():
    called = []
    def handler(request):
        called.append(request)
        return httpx.Response(200, json={"count": 16000, "tokens": [1] * 16000, "max_model_len": 16384})
    cfg = settings()
    client = NativeRWKVClient(base_url=cfg.native_base_url, model=cfg.native_model,
        prompt_protocol="g1j_plain", require_model_identity=True, transport=httpx.MockTransport(handler))
    engine = UnifiedRetrievalPipeline(cfg, None, model=client)
    response = await engine.ask_materials("目标", [material()])
    assert len(called) == 1 and called[0].url.path == "/tokenize"
    event = response.generation["model_calls"][0]
    assert event["status"] == "budget_exceeded" and event["completion_attempted"] is False
    assert event["budget"]["input_tokens"] == 16000
    assert event["prompt_sha256"] == digest(event["prompt"])
    await engine.aclose()


async def test_concurrent_requests_do_not_share_catalog_or_task_state():
    async def delayed(data):
        await asyncio.sleep(0.005)
        e = data["original_material"]["evidence_id"]
        return {"answer": data["task"]["goals"][0]["text"] + f" [资料 {e[1:]}]", "evidence_ids": [e], "missing": []}
    model = Model(overrides={"leaf": delayed})
    engine = UnifiedRetrievalPipeline(settings(), None, model=model)
    first, second = await asyncio.gather(engine.ask_materials("任务甲", [material(1)]),
                                        engine.ask_materials("任务乙", [material(2)]))
    assert "任务甲" in first.answer and "任务乙" not in first.answer
    assert "任务乙" in second.answer and "任务甲" not in second.answer
    assert first.sources[0].id != second.sources[0].id
    assert set(first.generation["citation_map"]) == set(second.generation["citation_map"]) == {"1"}


@pytest.mark.parametrize("change", [
    {"native_json_constraints": "schema_v1"}, {"native_preserve_original_question": True},
    {"native_planner_format_repair": True}, {"native_writer_pipeline": "funnel_v1"},
    {"native_require_model_identity": False}, {"native_task_matrix_enabled": True},
])
def test_candidate_rejects_mixed_protocols(change):
    values = settings().model_dump()
    values.update(change)
    with pytest.raises(ValueError):
        Settings(_env_file=None, **values)


async def test_service_explicit_opt_in_and_no_legacy_fallback(monkeypatch):
    model = Model()
    monkeypatch.setattr("llamaindex_retrieval.unified_retrieval.NativeRWKVClient", lambda **kwargs: model)
    service = SearchService(settings(), None)
    assert isinstance(service.native_pipeline, UnifiedRetrievalPipeline)
    response = await service.ask_materials("目标", [material()])
    assert response.generation["protocol"] == "rwkv-unified-v2"
    assert Settings(_env_file=None).native_unified is None
    await service.aclose()


def test_json_contract_rejects_extra_keys_and_coercion():
    for raw in ['{"answer":123,"evidence_ids":[],"missing":[]}',
                '{"answer":"a","evidence_ids":[],"missing":[],"keep":true}']:
        with pytest.raises(ValueError):
            parse(raw, Reading)
