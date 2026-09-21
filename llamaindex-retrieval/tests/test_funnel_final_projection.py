import json
from llamaindex_retrieval.funnel_final_projection import final_prompt


def test_last_writer_has_summaries_not_repeated_cell_and_debug_graph():
    task = json.dumps({"history": [], "latest_question": "比较甲乙"}, ensure_ascii=False)
    spec = {"mode": "comparison", "requirements": []}
    objects = {"O1": "甲", "O2": "乙"}
    fields = {"F1": {"name": "容量", "question": "容量是多少？", "value_type": "quantity"}}
    prompt = final_prompt(task, spec, objects, fields,
        [{"field_id": "F1", "execution_status": "completed", "summary": "甲8L乙12L"}], [], None,
        [{"label": "资料 1", "text": "甲8L，乙12L"}])
    assert "甲8L乙12L" in prompt and "甲8L，乙12L" in prompt
    assert '"cells"' not in prompt and '"fact_ids"' not in prompt and '"value_type"' not in prompt


def test_empty_evidence_prompt_has_no_citation_example_and_preserves_question():
    task = '{"latest_question":"甲乙哪个更好"}'
    prompt = final_prompt(task, {"mode":"comparison", "requirements":[]}, {}, {}, [], [], None, [])
    assert "[资料" not in prompt and task in prompt
    assert "没有可用于支持回答的来源材料" in prompt
