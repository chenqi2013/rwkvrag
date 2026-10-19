import json

import pytest

from llamaindex_retrieval.unified_contracts import digest, parse
from llamaindex_retrieval.unified_reading import LocalReading, local_prompt


def material(text="甲在离线模式为0，禁止拆开。乙为8GiB。", start=31):
    return {"evidence_id": "E9", "source_id": "unit-hash", "snapshot_id": "snapshot-hash",
            "title": "仅标题中的词", "text": text,
            "span": {"start": start, "end": start + len(text), "unit": "unicode_code_points",
                     "sha256": digest(text)}, "context_spans": [{"text": "仅上下文中的词"}]}


def reading(quote, statement="模型解释"):
    return parse(json.dumps({"findings": [{"quote": quote, "statement": statement}], "missing": []}), LocalReading)


def test_unicode_offsets_hash_and_raw_unchanged():
    source = material()
    raw = '{ "findings": [{"quote": "禁止拆开", "statement": "本条明确禁止拆开"}], "missing": [] }'
    value = parse(raw, LocalReading)
    binding = value.bind(source)[0]
    start = source["span"]["start"] + source["text"].index("禁止拆开")
    assert binding["spans"] == [{"start": start, "end": start + 4}]
    assert binding["quote_sha256"] == digest("禁止拆开")
    assert binding["evidence_id"] == "E9" and binding["semantic_support_verified"] is False
    assert raw.startswith('{ "findings"')


@pytest.mark.parametrize("quote", ["仅标题中的词", "仅上下文中的词", "甲为8GiB", "甲…乙", "甲在在线模式为0"])
def test_no_context_quote_rewriting_cross_unit_or_scope_repair(quote):
    with pytest.raises(ValueError, match="unbound_quote:0"):
        reading(quote).bind(material())


def test_overlapping_repeated_quotes_keep_all_occurrences():
    bound = reading("aa").bind(material("aaa", start=10))[0]
    assert bound["spans"] == [{"start": 10, "end": 12}, {"start": 11, "end": 13}]


def test_literal_binding_is_not_entailment_or_relevance_judgment():
    value = reading("禁止拆开", "可以拆开")
    assert value.bind(material())[0]["semantic_support_verified"] is False
    assert value.findings[0].statement == "可以拆开"  # never corrected by the validator


@pytest.mark.parametrize("update", [{"start": True}, {"end": 32}, {"sha256": "wrong"},
                                    {"unit": "bytes"}, {"start": -1}])
def test_invalid_original_coordinates_are_rejected(update):
    source = material()
    source["span"].update(update)
    with pytest.raises(ValueError, match="invalid_original_span"):
        reading("禁止拆开").bind(source)


def test_empty_findings_are_model_choice_not_program_negative_label():
    value = parse('{"findings":[],"missing":["当前对象的容量未获依据"]}', LocalReading)
    assert value.bind(material()) == []
    assert value.missing == ["当前对象的容量未获依据"]


@pytest.mark.parametrize("raw", [
    '```json\n{"findings":[],"missing":[]}\n```',
    '{"findings":[],"findings":[],"missing":[]}',
    '{"findings":[],"missing":[],"evidence_ids":["E9"]}',
    '{"findings":[{"quote":"禁止拆开","statement":3}],"missing":[]}',
    '{"findings":[{"quote":" ","statement":"解释"}],"missing":[]}',
])
def test_strict_no_repair(raw):
    with pytest.raises(ValueError):
        parse(raw, LocalReading)


def test_prompt_payload_byte_values_unchanged_and_no_answers_injected():
    payload = {"history": [{"role": "user", "content": "原先查甲"}],
               "latest_user_message": {"role": "user", "content": "改成乙，保留单位。"},
               "task": {"goals": ["可能不完整的旧模型解释"]}, "original_material": material()}
    before = json.dumps(payload, ensure_ascii=False)
    prompt = local_prompt(payload)
    assert json.loads(prompt.split("\n输入数据：\n", 1)[1]) == payload
    assert json.dumps(payload, ensure_ascii=False) == before
    assert '"additionalProperties"' not in prompt
