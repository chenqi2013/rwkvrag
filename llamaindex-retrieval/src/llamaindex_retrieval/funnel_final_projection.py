"""Give the last Writer summaries and their verbatim evidence, not the graph."""
import json
from .writer_prompt import writer_prompt_v2
from .typed_funnel_v9 import empty_evidence_prompt


def final_prompt(task, spec, objects, fields, summaries, candidates, decision, evidence):
    if not evidence:
        return empty_evidence_prompt(task)
    prompt = writer_prompt_v2(task, evidence, [field["name"] for field in fields.values()])
    if spec["mode"] == "fact":
        return prompt
    compact = {
        "required_objects": list(objects.values()),
        "active_requirements": [{"meaning": r["meaning"], "kind": r["kind"]}
                                for r in spec["requirements"] if r["state"] == "active"],
        "comparisons": [{"field": fields[row["field_id"]]["name"],
                         "execution_status": row["execution_status"], "summary": row.get("summary")}
                        for row in summaries],
        "candidate_eligibility": [{"object": objects[row["object_id"]], "status": row.get("status"),
                                   "explanation": row.get("explanation"), "execution_status": row["execution_status"]}
                                  for row in candidates],
        "selection": None if decision is None else {
            "recommended_objects": [objects[identity] for identity in decision["recommended_ids"]],
            "explanation": decision["explanation"]},
    }
    return prompt + ("\n以下是已完成的局部判断，不是新的证据。汇总其结论与限制，"
        "不再逐项重做比较或重复字段问题。每个事实仍须由上面的逐字证据支持；"
        "不得把未知写成不支持，不恢复撤回条件，不推荐被排除的候选。\n"
        + json.dumps(compact, ensure_ascii=False))
