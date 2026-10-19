"""Task interpretation and retrieval capability are separate, strictly bound contracts."""
import json

import pytest
from pydantic import ValidationError

from llamaindex_retrieval.schemas import SearchRequest
from llamaindex_retrieval.unified_contracts import Direction, Task, parse, parse_issues, task_input
from llamaindex_retrieval.unified_retrieval import UnifiedRetrievalPipeline
from test_unified_retrieval import Adapters, Model, material, settings

MESSAGES = [
    {"id": "M1", "role": "user", "content": "查甲对象的旧模式。"},
    {"id": "M2", "role": "user", "content": "改查它的新模式，不再使用旧模式。"},
]


def item(identity, text="当前目标", message="M2", quote=None):
    return {"id": identity, "text": text, "refs": [{"message_id": message,
            "quote": quote if quote is not None else MESSAGES[int(message[1:]) - 1]["content"]}]}


def task(**updates):
    return {"goals": [item("G1")], "constraints": [], "retracted": [], "unresolved": [], **updates}


def test_all_task_categories_have_same_shape_and_keep_their_ids():
    raw = json.dumps(task(constraints=[item("C1")], retracted=[item("R1", message="M1")]), ensure_ascii=False)
    value = parse(raw, Task)
    value.check(MESSAGES)
    assert value.constraints[0].id == "C1" and value.retracted[0].id == "R1"
    assert set(value.model_dump()) == {"goals", "constraints", "retracted", "unresolved"}


def test_latest_is_explicit_without_rewriting_or_losing_any_turn():
    value = task_input(MESSAGES)
    assert value == {"history": MESSAGES[:-1], "latest_user_message": MESSAGES[-1]}
    assert MESSAGES[-1]["content"] == "改查它的新模式，不再使用旧模式。"


def test_referencing_only_history_is_not_a_current_request_binding():
    value = parse(json.dumps(task(goals=[item("G1", message="M1")])), Task)
    with pytest.raises(ValueError, match="latest_request_not_referenced"):
        value.check(MESSAGES)


def test_latest_quote_does_not_prove_retraction_or_object_semantics():
    # Intentionally wrong interpretation with a real current-turn quote. A
    # structural check cannot certify that the meaning follows from the quote.
    value = parse(json.dumps(task(goals=[item("G1", text="继续使用旧模式")])), Task)
    assert value.check(MESSAGES) is None
    assert value.goals[0].text == "继续使用旧模式"


@pytest.mark.parametrize("changes", [{"question": "旧字段"}, {"mode": "provided"},
                                     {"channels": ["knowledge_base"]}])
def test_old_routing_fields_are_rejected_not_silently_ignored(changes):
    with pytest.raises(ValidationError):
        parse(json.dumps(task(**changes)), Task)


def test_schema_errors_record_location_not_raw_values():
    bad = task(constraints=[{**item("C1"), "surprise": "secret sentinel"}])
    with pytest.raises(ValidationError) as error:
        parse(json.dumps(bad), Task)
    issues = parse_issues(error.value)
    assert issues == [{"location": ["constraints", 0, "surprise"], "type": "extra_forbidden"}]
    assert "secret sentinel" not in json.dumps(issues)


def test_invented_quote_is_not_repaired():
    value = parse(json.dumps(task(goals=[item("G1", quote="未出现在任何原消息中的引用")])), Task)
    with pytest.raises(ValueError, match="unbound_conversation_quote"):
        value.check(MESSAGES)


@pytest.mark.parametrize("choice", ["provided", "knowledge_base", "clarify"])
def test_direction_single_choice_respects_scope(choice):
    value = Direction(choice=choice)
    value.check(["knowledge_base"])
    assert value.channels() in ([], ["knowledge_base"])


@pytest.mark.parametrize("choice", ["web", "hybrid"])
def test_direction_cannot_cross_request_permission(choice):
    with pytest.raises(ValueError, match="channel_not_permitted"):
        Direction(choice=choice).check(["knowledge_base"])


async def test_materials_entry_binding_is_not_forged_router_output():
    model = Model(mode="retrieve", channels=["web"])
    adapter = Adapters()
    engine = UnifiedRetrievalPipeline(settings(), None, model=model, adapters=adapter)
    response = await engine.ask_materials("目标", [material()])
    assert model.counts["direction"] == 0 and not adapter.calls
    assert response.retrieval["routing"] == {"source": "explicit_materials_entry", "choice": "provided",
                                            "allowed_channels": [], "model_decision": False}
    assert set(model.calls[0]["data"]) == {"history", "latest_user_message"}
    assert response.generation["status"] == "completed"


async def test_router_failure_does_not_default_to_provided_or_kb():
    model = Model(overrides={"direction": "invalid raw direction"})
    adapter = Adapters()
    result = await UnifiedRetrievalPipeline(settings(), None, model=model, adapters=adapter).ask(
        SearchRequest(question="目标", retrieval_mode="auto"))
    assert result.generation["status"] == "routing_failed"
    assert not adapter.calls and model.counts["leaf"] == 0
    assert result.generation["model_calls"][-1]["raw_text"] == "invalid raw direction"
    assert result.generation["model_calls"][-1]["contract_errors"][0]["type"] == "json_syntax"


async def test_route_and_plan_remain_separate_model_calls():
    model = Model(mode="retrieve", channels=["web"])
    adapter = Adapters()
    result = await UnifiedRetrievalPipeline(settings(), None, model=model, adapters=adapter).search(
        SearchRequest(question="目标", retrieval_mode="auto"))
    assert [call["purpose"] for call in model.calls][:3] == ["task", "direction", "plan"]
    plan = next(call["data"] for call in model.calls if call["purpose"] == "plan")
    assert plan["selected_channels"] == ["web"]
    assert plan["allowed_channels"] == ["knowledge_base", "web"]
    assert result.retrieval["routing"]["model_decision"] is True
