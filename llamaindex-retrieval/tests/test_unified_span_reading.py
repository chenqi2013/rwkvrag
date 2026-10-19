from dataclasses import replace
import json

import pytest

from llamaindex_retrieval.unified_contracts import digest, parse
from llamaindex_retrieval.unified_span_reading import SpanCatalog, SpanReading, span_prompt


def material(text="第一行。\r\n  第二行保留缩进。\n不能拆开。", start=17, evidence_id="E9", **extra):
    return {"evidence_id": evidence_id, "source_id": "source", "snapshot_id": "snapshot",
            "title": "标题不是正文", "text": text, "context_spans": [{"text": "外部标题"}],
            "span": {"start": start, "end": start + len(text), "unit": "unicode_code_points",
                     "sha256": digest(text)}, **extra}


def reading(span_id="E9.S1", statement="模型主张"):
    return parse(json.dumps({"findings": [{"span_id": span_id, "statement": statement}], "missing": []}), SpanReading)


def test_body_coverage_original_offsets_overlap_and_whitespace():
    source = material("中文🙂\r\n  abcdefg\n" * 100)
    catalog = SpanCatalog.build(source, window=80, overlap=11)
    parts = catalog.model_material()["body_spans"]
    assert len(parts) > 1
    covered = set()
    for part in parts:
        a, b = part["span"]["start"] - 17, part["span"]["end"] - 17
        assert source["text"][a:b] == part["text"]
        assert digest(part["text"]) == part["span"]["sha256"]
        covered.update(range(a, b))
    assert covered == set(range(len(source["text"])))
    assert parts[1]["span"]["start"] == parts[0]["span"]["end"] - 11


@pytest.mark.parametrize("text,kind", [
    ("| 名称 | 数值 |\n| --- | --- |\n" + "| 甲 | 0 |\n" * 40, "table"),
    ("```python\n" + "x = 0\n" * 40 + "```\n", "code"),
    ("- 甲\n  继续\n\n- 乙\n" * 40, "list"),
])
def test_structural_atoms_not_truncated_for_window(text, kind):
    parts = SpanCatalog.build(material(text), window=32, overlap=4).model_material()["body_spans"]
    assert len(parts) == 1 and parts[0]["text"] == text
    assert parts[0]["kind"] == kind and parts[0]["atomic"]


def test_declared_qa_whole_atom():
    source = material("# 问\n问题？\n# 答\n" + "答文\n" * 100, kind="qa-markdown")
    parts = SpanCatalog.build(source, window=32, overlap=4).model_material()["body_spans"]
    assert len(parts) == 1 and parts[0]["text"] == source["text"]


def test_context_coordinates_remain_original_but_no_external_context_ids():
    catalog = SpanCatalog.build(material("# 标题\n\n正文" * 2))
    parts = catalog.model_material()["body_spans"]
    assert all(p["id"].startswith("E9.S") for p in parts)
    assert not any(p["text"] == "外部标题" for p in parts)
    # Internal headings stay in the packed body, not duplicate external context.
    assert len(parts) == 1 and parts[0]["text"] == "# 标题\n\n正文" * 2
    long_parts = SpanCatalog.build(material("# 标题\n" + "正文" * 1000)).model_material()["body_spans"]
    context = next(c for p in long_parts for c in p["context_spans"])
    assert context["start"] == 17 and context["text"] == "# 标题\n"


def test_selected_id_receipt_is_lookup_not_model_quote_or_semantic_proof():
    source = material("禁止拆开。")
    raw = '{"findings":[{"span_id":"E9.S1","statement":"可以拆开。"}],"missing":[]}'
    value = parse(raw, SpanReading)
    bound = value.bind(SpanCatalog.build(source))[0]
    assert bound["selected_original_text"] == "禁止拆开。"
    assert bound["text_source"] == "original_material" and not bound["model_generated_quote"]
    assert not bound["semantic_support_verified"]
    assert value.findings[0].statement == "可以拆开。" and "禁止" not in raw


@pytest.mark.parametrize("span_id", ["E9.S0", "E9.S999", "E1.S1", "S1", " E9.S1", "E9.S1 ", "标题"])
def test_unknown_ids_not_repaired_or_aliased(span_id):
    with pytest.raises(ValueError, match="unknown_span_id:0"):
        reading(span_id).bind(SpanCatalog.build(material()))


def test_selection_order_duplicates_and_no_unselected_evidence_added():
    catalog = SpanCatalog.build(material("正文" * 100), window=40, overlap=5)
    value = parse('{"findings":[{"span_id":"E9.S2","statement":"a"},{"span_id":"E9.S2","statement":"b"}],"missing":[]}', SpanReading)
    result = value.bind(catalog)
    assert [b["selected_span_id"] for b in result] == ["E9.S2", "E9.S2"]


def test_empty_selection_is_model_choice_with_no_invented_evidence():
    value = parse('{"findings":[],"missing":["未找到依据"]}', SpanReading)
    assert value.bind(SpanCatalog.build(material())) == []


@pytest.mark.parametrize("update", [{"start": True}, {"end": 19}, {"sha256": "bad"}, {"unit": "bytes"}])
def test_bad_material_coordinates_rejected(update):
    source = material()
    source["span"].update(update)
    with pytest.raises(ValueError, match="invalid_original_span"):
        SpanCatalog.build(source)


def test_catalog_immutable_and_tampering_rejected():
    source = material()
    catalog = SpanCatalog.build(source)
    source["text"] = "changed"
    projected = catalog.model_material()
    projected["body_spans"][0]["text"] = "changed"
    assert catalog.model_material()["body_spans"][0]["text"] != "changed"
    parts = json.loads(catalog.spans_json)
    parts[0]["span"]["start"] += 1
    with pytest.raises(ValueError, match="invalid_span_catalog"):
        reading().bind(replace(catalog, spans_json=json.dumps(parts)))


def test_prompt_preserves_conversation_and_checks_catalog_identity():
    payload = {"history": [{"role": "assistant", "content": "原字节\r\n"}],
               "latest_user_message": {"role": "user", "content": "改问乙，不再查甲。"},
               "task": None, "original_material": material()}
    before = json.dumps(payload)
    prompt = span_prompt(payload, SpanCatalog.build(payload["original_material"]))
    data = json.loads(prompt.split("\n输入数据：\n", 1)[1])
    assert data["history"] == payload["history"]
    assert data["latest_user_message"] == payload["latest_user_message"]
    assert "text" not in data["original_material"]
    assert json.dumps(payload) == before
    with pytest.raises(ValueError, match="catalog_material_mismatch"):
        span_prompt(payload, SpanCatalog.build(material("different")))


@pytest.mark.parametrize("raw", [
    '```json\n{"findings":[],"missing":[]}\n```',
    '{"findings":[],"findings":[],"missing":[]}',
    '{"findings":[],"missing":[],"answer":"a"}',
    '{"findings":[{"span_id":1,"statement":"s"}],"missing":[]}',
    '{"findings":[{"span_id":"E9.S1","statement":"s","quote":"q"}],"missing":[]}',
])
def test_strict_contract_no_output_repair(raw):
    with pytest.raises(ValueError):
        parse(raw, SpanReading)
