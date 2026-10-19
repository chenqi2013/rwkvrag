import asyncio
import json
import httpx
import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.model_client import model_client_class, model_client_options
from llamaindex_retrieval.native_json_contract import json_output_options
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline
from llamaindex_retrieval.schemas import ConversationMessage, SearchRequest
from llamaindex_retrieval.structured_native import StructuredNativeRWKVClient
from test_rwkv_pipeline import FakeIndex, hit


def test_concurrent_schemas_are_request_scoped_and_wire_matches_trace():
    seen=[]
    async def handle(request):
        data=json.loads(request.content)
        if request.url.path == '/tokenize':
            return httpx.Response(200, json={'count': 2, 'tokens': [0, 2], 'max_model_len': 16384})
        await asyncio.sleep(.002)
        seen.append(data)
        return httpx.Response(200,json={'choices':[{'text':'<think></think>\n{}','finish_reason':'stop'}],'usage':{'prompt_tokens':2,'completion_tokens':4}})
    async def run():
        async with StructuredNativeRWKVClient(base_url='http://test/v1',model='test',transport=httpx.MockTransport(handle)) as client:
            return await asyncio.gather(*(client.complete([{'role':'user','content':str(i)}],structured_schema=schema) for i,schema in enumerate([{'type':'object'},{'type':'array'},None])))
    results=asyncio.run(run())
    assert len(seen)==3
    for result in results:
        payload=json.loads(__import__('base64').b64decode(result.trace['http'][-1]['request_body_base64']))
        assert payload.get('structured_outputs')==result.trace['parameters'].get('structured_outputs')
    assert 'structured_outputs' not in seen[-1]
    assert seen[0]['structured_outputs']!=seen[1]['structured_outputs']


def constrained_settings(**overrides):
    values = dict(
        _env_file=None, native_json_constraints="schema_v1",
        native_transport="native", native_completion_protocol="g1j_plain",
        native_require_model_identity=True, native_model="contract-test",
        native_planner_prefill="<think></think", native_resolver_prefill="<think></think",
        native_writer_prefill="<think></think", native_writer_pipeline="single",
        native_history_protocol="current-question-v2", native_plan_protocol="fact_queries_v1",
        native_resolver_protocol="binary_query", native_resolver_task_grouping="individual",
        native_task_source="queries", native_empty_evidence_policy="fail",
        native_answer_quality_policy="fail_citation", native_max_queries=2,
    )
    return Settings(**(values | overrides))


@pytest.mark.parametrize("overrides", [
    {"native_transport": "rwkvos_batch"}, {"native_completion_protocol": "native"},
    {"native_require_model_identity": False}, {"native_plan_protocol": "queries_fields"},
    {"native_resolver_protocol": "task_units"}, {"native_writer_pipeline": "typed_funnel_v10"},
    {"native_task_matrix_enabled": True}, {"native_planner_format_repair": True},
    {"native_resolver_format_repair": True},
])
def test_constraint_profile_rejects_unimplemented_combinations(overrides):
    with pytest.raises(ValueError, match="schema_v1"):
        constrained_settings(**overrides)


def test_schema_contracts_are_fresh_bounded_and_keep_no_as_a_valid_decision():
    settings = constrained_settings()
    schemas = {role: json_output_options(settings, role)["structured_schema"]
               for role in ("current_question", "plan", "reader")}
    for schema in schemas.values():
        assert schema["required"] == list(schema["properties"])
        assert schema["additionalProperties"] is False
    queries = schemas["plan"]["properties"]["queries"]
    assert (queries["minItems"], queries["maxItems"]) == (1, 2)
    assert queries["items"] == {
        "type": "string", "minLength": 1, "maxLength": 2000, "pattern": r"\S"}
    assert schemas["current_question"]["properties"]["question"]["maxLength"] == 8000
    assert schemas["reader"]["properties"]["answer"]["enum"] == ["YES", "NO"]
    queries["maxItems"] = 999
    assert json_output_options(settings, "plan")["structured_schema"]["properties"]["queries"]["maxItems"] == 2
    assert json_output_options(Settings(_env_file=None), "reader") == {}
    with pytest.raises(ValueError, match="purpose"):
        json_output_options(settings, "writer")


async def ordinary_request(outputs, *, history=False, **overrides):
    """Exercise the actual transport and pipeline with a deterministic HTTP boundary."""
    seen, completions = [], []
    settings = constrained_settings(**overrides)
    pending = iter(outputs)

    def handle(request):
        payload = json.loads(request.content)
        seen.append((request.url.path, payload))
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 1, "tokens": [11], "max_model_len": 16384})
        completions.append(payload)
        code, raw, reason = next(pending)
        if code != 200:
            return httpx.Response(code, json={"error": "schema backend rejected the request"})
        return httpx.Response(200, json={"model": settings.native_model, "choices": [{
            "text": raw, "finish_reason": reason, "prompt_token_ids": [11], "token_ids": [12],
        }], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    model = model_client_class(settings)(**model_client_options(settings),
                                         transport=httpx.MockTransport(handle))
    index = FakeIndex({"测试检索": [hit("source-one", "逐字测试材料。") ]})
    pipeline = RWKVPipeline(settings, index, model=model)
    try:
        response = await pipeline.ask(SearchRequest(
            question="当前完整测试问题",
            history=[ConversationMessage(role="user", content="历史输入原样保留")] if history else [],
        ))
        return response, seen, completions, index.calls
    finally:
        await pipeline.aclose()


async def test_ordinary_history_plan_reader_have_distinct_wire_schemas_writer_stays_raw():
    writer = " \n原始答案，即使引用不存在也不修改。[资料 999]\n"
    outputs = [(200, raw, "stop") for raw in [
        '{"question":"当前完整测试问题"}', '{"queries":["测试检索"]}',
        '{"answer":"YES"}', writer,
    ]]
    constrained, wire, calls, _ = await ordinary_request(outputs, history=True)
    baseline, old_wire, _, _ = await ordinary_request(
        outputs, history=True, native_json_constraints="disabled")
    assert len(calls) == 4
    assert [list(c["structured_outputs"]["json"]["properties"]) for c in calls[:3]] == [
        ["question"], ["queries"], ["answer"]]
    assert "structured_outputs" not in calls[-1]
    # The new profile changes the decoding contract, not prompt/State/model/budget.
    assert [(p, {k: v for k, v in body.items() if k != "structured_outputs"})
            for p, body in wire] == old_wire
    for result in (constrained, baseline):
        assert result.answer == result.generation["raw_model_answer"] == writer
        assert result.generation["answer_modified"] is False
        assert result.generation["status"] == "answer_quality_failed"
    for event, payload in zip(constrained.generation["model_calls"], calls, strict=True):
        assert event["parameters"].get("structured_outputs") == payload.get("structured_outputs")
        sent = next(h["payload"] for h in event["http"] if h["stage"] == "completion")
        assert sent == payload


@pytest.mark.parametrize("failure", [
    (400, "", "stop"),
    (200, '{"queries":["测试检索"]}', "length"),
    (200, '{"queries":["a","b","c"]}', "stop"),
    (200, '{"queries":["a"],"answer":"unexpected"}', "stop"),
    (200, '{"queries":["a"],"queries":["b"]}', "stop"),
    (200, '{"queries":["   "]}', "stop"),
])
async def test_rejected_or_invalid_constrained_plan_is_not_retried_or_used_as_fallback(failure):
    result, _, calls, searches = await ordinary_request([failure])
    assert result.generation["status"] == "planner_failed"
    assert result.answer == "" and not result.sources and not searches
    assert len(calls) == 1 and "structured_outputs" in calls[0]
    event = result.generation["model_calls"][0]
    assert event.get("parse_error")
    if failure[0] == 200:
        assert event["raw_text"] == failure[1]


@pytest.mark.parametrize("decision,status", [
    ('{"answer":"NO"}', "no_evidence"),
    ('{"answer":"NO","answer":"YES"}', "resolver_failed"),
    ('{"answer":"MAYBE"}', "resolver_failed"),
])
async def test_constrained_reader_never_forces_evidence_or_repairs_bad_decisions(decision, status):
    result, _, calls, _ = await ordinary_request([
        (200, '{"queries":["测试检索"]}', "stop"), (200, decision, "stop"),
    ])
    assert result.generation["status"] == status
    assert result.answer == "" and result.sources == [] and len(calls) == 2
    assert result.generation["model_calls"][-1]["raw_text"] == decision
