import pytest
from contracts import compile_item, source_map, review_map


def atomic(value=False):
    return {"kind":"atomic", "source_id":"S1", "object":"甲设备",
        "field":{"name":"离线支持", "question":"是否支持离线", "value_type":"boolean"},
        "answer":{"evidence_ids":["E1"], "quote":"标准模式不支持离线。", "value":value, "source_scope":"标准模式"},
        "rationale":"原文明确否定", "skill":"否定不等于缺失"}


def test_false_is_preserved_in_training_target():
    out = compile_item(atomic(), {"S1":{"title":"设备手册", "text":"标准模式不支持离线。"}})
    assert '"value": false' in out["target"]
    assert out["state_role"] == "resolver"
    assert out["prompt"].endswith("Assistant: <think></think>\n")


def test_bad_scope_is_rejected_not_silently_dropped():
    x = atomic(); x["answer"]["source_scope"] = "设备手册"
    with pytest.raises(ValueError, match="scope"):
        compile_item(x, {"S1":{"title":"设备手册", "text":"标准模式不支持离线。"}})


def test_foreign_object_is_not_automatically_judged_correct_by_literal_check():
    x = atomic(); x["object"] = "乙设备"
    result = compile_item(x, {"S1":{"title":"甲设备手册", "text":"标准模式不支持离线。"}})
    assert "semantic_correct" not in result  # Teacher/human must reject the object mismatch.


def test_writer_cannot_use_foreign_citation():
    x = {"kind":"writer", "question":"甲重多少", "evidence":[{"source_id":"S1","quote":"甲重3kg。"}],
         "answer":"甲重3kg。[资料 2]", "rationale":"依据", "skill":"引用绑定"}
    with pytest.raises(ValueError, match="citation"):
        compile_item(x, {"S1":{"title":"记录", "text":"甲重3kg。"}})


def test_real_source_cannot_be_rewritten():
    with pytest.raises(ValueError, match="rewrote"):
        source_map({"material_mode":"real","sources":[]}, {"sources":[{"id":"S1"}], "items":[]})


def test_incomplete_or_duplicate_reviews_cannot_admit_data():
    row = {"index":0,"accept":True,"issues":[],"reason":"依据原文"}
    with pytest.raises(ValueError, match="Incomplete"):
        review_map({"source_set_valid":True,"reviews":[row]},2)
    with pytest.raises(ValueError, match="duplicate"):
        review_map({"source_set_valid":True,"reviews":[row,row]},2)
    assert not review_map({"source_set_valid":False,"reviews":[row]},1)[0]["accept"]


def test_status_counts_cannot_be_negative_or_boolean():
    x = {"kind":"status", "question":"能否比较", "flow":{"input_source_count":2,"writer_source_count":0,
         "failed_node_count":-1,"failed_cell_count":0,"unexamined_job_count":272},
         "answer":"不能可靠比较。", "rationale":"状态", "skill":"任务计数"}
    with pytest.raises(ValueError, match="counters"):
        compile_item(x,{})
    x["flow"]["failed_node_count"] = False
    with pytest.raises(ValueError, match="counters"):
        compile_item(x,{})
