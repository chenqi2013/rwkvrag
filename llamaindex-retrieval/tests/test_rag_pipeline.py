import json
from dataclasses import replace
from hashlib import sha256
from typing import Any

import httpx
import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.evidence_extraction import (
    EvidenceExtractionResult,
    EvidenceSpan,
    LanguageModelEvidenceExtractor,
)
from llamaindex_retrieval.generation import EvidenceAnswerGenerator
from llamaindex_retrieval.lexical_index import LexicalResult
from llamaindex_retrieval.query_planning import build_query_plan
from llamaindex_retrieval.schemas import SearchRequest
from llamaindex_retrieval.semantic_query_planning import LanguageModelQueryPlanner
from llamaindex_retrieval.service import SearchService


class MemoryIndex:
    def __init__(self, results: list[LexicalResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def search(self, question: str, **kwargs: Any) -> list[LexicalResult]:
        self.calls.append((question, kwargs))
        return self.results


def record(document: str, text: str, score: float = 1.0) -> LexicalResult:
    return LexicalResult(
        node_id=document,
        document_id=document,
        text=text,
        score=score,
        metadata={"title": document, "source": "test", "uri": f"https://example.org/{document}"},
    )


def stream_response(content: str) -> httpx.Response:
    event = json.dumps({"choices": [{"delta": {"content": content}}]}, ensure_ascii=False)
    return httpx.Response(200, text=f"data: {event}\n\ndata: [DONE]\n\n")


@pytest.mark.parametrize(
    "question",
    [
        "这个项目最早由谁建立？",
        "创始人是哪位？",
        "能告诉我它是谁创办的吗？",
    ],
)
@pytest.mark.parametrize("top_k", [1, 10])
@pytest.mark.asyncio
async def test_model_planning_to_resolver_to_writer_preserves_provenance(question, top_k) -> None:
    selected = "示例项目由乙研究者创立。"
    distractor = "示例项目的办公大楼位于河岸。"
    raw_answer = "乙研究者。[资料 1]\n"
    query_variants = ["示例项目 创建者", "示例项目 创立人"]
    planner_output = json.dumps(
        {
            "subject": "示例项目",
            "intent": "agent",
            "answer_shape": "single_fact",
            "set_semantics": "specific",
            "fields": [{"field_id": "f1", "question": question, "relations": ["创建者"]}],
            "relations": ["创建者"],
            "queries": query_variants,
        },
        ensure_ascii=False,
    )
    prompts: dict[str, list[str]] = {"planner": [], "map": [], "resolver": [], "writer": []}

    async def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["contents"][0]
        if "BM25 查询规划器" in prompt:
            prompts["planner"].append(prompt)
            return stream_response(planner_output)
        if "你是证据抽取器" in prompt:
            prompts["map"].append(prompt)
            assert (selected in prompt) != (distractor in prompt)
            return stream_response("f1:s1")
        if "你是证据裁决器" in prompt:
            prompts["resolver"].append(prompt)
            evidence_id = next(
                line.split("]", 1)[0].removeprefix("[")
                for line in prompt.splitlines()
                if line.startswith("[e") and selected in line
            )
            return stream_response(f"f1:{evidence_id}")
        prompts["writer"].append(prompt)
        assert selected in prompt
        assert distractor not in prompt
        assert prompt.index(selected) < prompt.rindex(question)
        return stream_response(raw_answer)

    settings = Settings(
        generation_password="secret",
        answer_point_fanout_enabled=False,
        generation_output_mode="immutable",
        semantic_pipeline_enabled=True,
    )
    transport = httpx.MockTransport(handler)
    index = MemoryIndex([record("background", distractor), record("origin", selected, 0.8)])
    service = SearchService(
        settings,
        index,
        query_planner=LanguageModelQueryPlanner(settings, transport=transport),
        evidence_extractor=LanguageModelEvidenceExtractor(settings, transport=transport),
        generator=EvidenceAnswerGenerator(settings, transport=transport),
    )

    response = await service.ask(
        SearchRequest(question=question, top_k=top_k, knowledge_base_id="kb-test")
    )

    assert response.answer == raw_answer
    assert [item.snippet for item in response.sources] == [selected]
    assert response.sources[0].document_id == "origin"
    assert response.sources[0].uri == "https://example.org/origin"
    assert response.generation["raw_model_answer"] == raw_answer
    assert response.generation["writer_trace"]["output_modified"] is False
    assert (
        response.generation["field_evidence"][0]["sha256"] == sha256(selected.encode()).hexdigest()
    )
    assert {query for query, _kwargs in index.calls} >= {*query_variants, question}
    assert all(kwargs["knowledge_base_id"] == "kb-test" for _query, kwargs in index.calls)
    assert {stage: len(values) for stage, values in prompts.items()} == {
        "planner": 1,
        "map": 2,
        "resolver": 1,
        "writer": 1,
    }


@pytest.mark.parametrize(
    "state", ["empty_index", "empty_extraction", "extraction_error", "no_extractor"]
)
@pytest.mark.asyncio
async def test_missing_evidence_never_calls_writer_and_preserves_failure_reason(state) -> None:
    class Resolver:
        async def extract(self, question, plan, sources):
            if state == "extraction_error":
                raise TimeoutError("resolver timed out")
            return EvidenceExtractionResult((), len(sources), len(sources))

    class Writer:
        async def generate(self, question, sources):
            pytest.fail("writer must not run without selected evidence")

    index = MemoryIndex([] if state == "empty_index" else [record("record", "资料正文。")])
    service = SearchService(
        Settings(answer_point_fanout_enabled=False),
        index,
        generator=Writer(),
        evidence_extractor=None if state == "no_extractor" else Resolver(),
    )
    response = await service.ask(SearchRequest(question="任意问题？", top_k=1))
    assert response.generation["answer_strategy"] == "writer_not_called"
    assert response.generation["writer_trace"] is None
    assert response.generation["raw_model_answer"] is None
    assert response.sources == []
    assert not response.generation["evidence_gate_passed"]
    if state == "extraction_error":
        assert "resolver timed out" in response.generation["field_evidence_errors"][0]
    else:
        assert response.generation["field_evidence_errors"] == []


@pytest.mark.asyncio
async def test_writer_failure_is_distinguished_from_missing_evidence() -> None:
    class Resolver:
        async def extract(self, question, plan, sources):
            span = sources[0].snippet
            return EvidenceExtractionResult(
                (EvidenceSpan("f1", 0, span, sha256(span.encode()).hexdigest()),),
                1,
                1,
            )

    class Writer:
        async def generate(self, question, sources):
            raise RuntimeError("generation unavailable")

    service = SearchService(
        Settings(answer_point_fanout_enabled=False),
        MemoryIndex([record("record", "逐字证据。")]),
        evidence_extractor=Resolver(),
        generator=Writer(),
    )
    response = await service.ask(SearchRequest(question="任意问题？"))
    assert response.generation["answer_strategy"] == "generation_failed"
    assert response.generation["evidence_gate_passed"]
    assert "generation unavailable" in response.generation["generation_error"]
    assert response.generation["raw_model_answer"] is None


@pytest.mark.parametrize(("display", "expected"), [(1, 5), (5, 5), (10, 10), (50, 20)])
def test_evidence_budget_depends_on_limits_instead_of_question_vocabulary(
    display, expected
) -> None:
    service = SearchService(Settings(max_top_k=20), MemoryIndex([]))
    for question in ("首都在哪里？", "列出全部车站", "请介绍某个项目", "为什么灭亡？"):
        budget, trace = service._adaptive_evidence_top_k(
            build_query_plan(question), display_top_k=display
        )
        assert budget == expected
        assert trace["effective_top_k"] == expected


def test_rrf_does_not_boost_document_using_subject_or_relation_rules() -> None:
    groups = ([record("b", "普通片段"), record("a", "原因 创始人 首都")],)
    original = build_query_plan("任意问题")
    selected = SearchService._merge_rank_fusion(groups, plan=original)
    changed = SearchService._merge_rank_fusion(
        groups, plan=replace(original, subject="a", relations=("原因",))
    )
    assert [item.document_id for item in selected] == ["b", "a"]
    assert selected == changed
