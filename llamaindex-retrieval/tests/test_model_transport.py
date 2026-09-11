import asyncio

import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.model_client import model_answer_bounds, model_client_options
from llamaindex_retrieval.native_rwkv import NativeRWKVResult
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline, structured_body
from llamaindex_retrieval.schemas import SourceItem


def batch_trace(raw, **extra):
    return {"transport": "rwkvos_batch", "stage": "writer", "raw_text": raw,
            "termination_verified": False, "provider_finish_reason": "stop",
            "envelope": {"valid": True, "answer_span": {
                "start": 0, "end": len(raw), "unit": "unicode_code_points"}}, **extra}


def test_plain_batch_body_is_interpreted_without_forging_native_envelope():
    raw = ' \n{"queries":["A"],"fields":["B"]}\n'
    result = NativeRWKVResult("completed", raw, "stop", batch_trace(raw))
    assert structured_body(result) == raw.strip()
    assert result.raw_text == raw
    native = NativeRWKVResult("completed", raw, "stop", {"prefill": "<think"})
    with pytest.raises(ValueError):
        structured_body(native)
    with pytest.raises(ValueError):
        structured_body(NativeRWKVResult("length", raw, "length", batch_trace(raw)))


@pytest.mark.parametrize("span", [None, [], {},
    {"start": True, "end": 2, "unit": "unicode_code_points"},
    {"start": -1, "end": 2, "unit": "unicode_code_points"},
    {"start": 0, "end": 20, "unit": "unicode_code_points"},
    {"start": 0, "end": 2, "unit": "utf8_bytes"}])
def test_invalid_batch_boundaries_are_not_recovered(span):
    trace = batch_trace("正文")
    trace["envelope"]["answer_span"] = span
    assert model_answer_bounds("正文", trace) is None


def test_unknown_transport_and_empty_body_are_rejected():
    assert model_answer_bounds("正文", batch_trace("正文", transport="unknown")) is None
    assert model_answer_bounds("  ", batch_trace("  ")) is None
    assert model_answer_bounds(None, {}) is None


def test_batch_writer_retains_raw_unicode_citations_and_unknown_termination():
    raw = "\n甲😀于2024年首航。[资料 1]\n"
    class Model:
        async def complete(self, messages, **kwargs):
            assert kwargs["assistant_prefill"] == "<think></think"
            assert kwargs["evidence_ids"] == ("source1",)
            return NativeRWKVResult("completed", raw, "stop", batch_trace(raw))
    settings = Settings(_env_file=None, native_transport="rwkvos_batch",
                        native_writer_prefill="<think></think")
    pipeline = RWKVPipeline(settings, object(), model=Model())
    source = SourceItem(id="source1", document_id="d1", source="fixture", title="甲",
                        snippet="甲于2024年首航。", score=1, metadata={})
    response = asyncio.run(pipeline.ask_materials("何时首航？", [source]))
    assert response.answer == raw
    assert response.generation["answer_span"] == [0, len(raw)]
    assert response.generation["citation_audit"]["label_ids"] == [1]
    assert response.generation["citation_map"] == {"1": "source1"}
    assert response.generation["termination_verified"] is False
    assert response.generation["answer_modified"] is False


def test_cf_credentials_are_secret_in_settings_and_only_present_in_headers():
    settings = Settings(_env_file=None, native_transport="rwkvos_batch",
                        rwkvos_cf_access_client_id="test-private-id",
                        rwkvos_cf_access_client_secret="test-private-secret")
    assert "test-private-id" not in settings.model_dump_json()
    assert "test-private-secret" not in settings.model_dump_json()
    options = model_client_options(settings)
    assert options["headers"] == {"CF-Access-Client-Id": "test-private-id",
                                  "CF-Access-Client-Secret": "test-private-secret"}
    assert "api_key" not in options


def test_continuation_prefill_changes_only_final_prompt_character_and_body_bounds():
    from llamaindex_retrieval.rwkvos_batch import render_batch_prompt, inspect_batch_envelope
    messages = [{"role": "user", "content": "根据原文回答。\n\n原文含\"引用\"和😀。"}]
    complete, full_prefill = render_batch_prompt(messages, "<think></think")
    continuation, partial_prefill = render_batch_prompt(messages, "<think></think", "continuation")
    assert complete == continuation + ">"
    assert full_prefill == partial_prefill + ">"
    raw = ">答案😀[资料 1]"
    envelope = inspect_batch_envelope(raw, partial_prefill)
    assert envelope["answer_span"] == {"start": 1, "end": len(raw), "unit": "unicode_code_points"}
    assert raw == ">答案😀[资料 1]"
    assert not inspect_batch_envelope("答案", partial_prefill)["valid"]
    assert not inspect_batch_envelope(">尚未结束的思考", "<think")["valid"]
    assert inspect_batch_envelope(">先想一下</think>答案", "<think")["valid"]


@pytest.mark.parametrize("overrides", [
    {"rwkvos_reader_state_id": "reader-trained"},
    {"rwkvos_reader_prompt_protocol": "rwkv_g1j_no_think_v1"},
    {"rwkvos_reader_input_layout": "task_last"},
])
def test_reader_state_configuration_rejects_mismatched_pipeline(overrides):
    with pytest.raises(ValueError):
        Settings(_env_file=None, **overrides)


def test_canonical_reader_configuration_routes_only_reader_options():
    settings = Settings(_env_file=None, native_transport="rwkvos_batch",
        native_resolver_protocol="task_units", native_resolver_prefill="<think></think",
        rwkvos_reader_prompt_protocol="rwkv_g1j_no_think_v1", rwkvos_reader_state_id="reader-trained",
        rwkvos_reader_input_layout="task_last", native_resolver_max_tokens=32)
    options = model_client_options(settings)
    assert options["state_id"] is None
    assert options["reader_state_id"] == "reader-trained"
    assert options["reader_prompt_protocol"] == "rwkv_g1j_no_think_v1"
    assert options["reader_input_layout"] == "task_last"
    assert settings.native_resolver_max_tokens == 32
