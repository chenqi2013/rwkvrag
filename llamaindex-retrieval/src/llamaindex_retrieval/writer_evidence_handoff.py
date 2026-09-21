"""Carry execution provenance into a new Writer call, without editing its answer."""
import json

PROTOCOL = "writer-evidence-handoff-v1"


def with_evidence_handoff(prompt, flow):
    if flow.get("protocol") != "evidence-flow-v1" or flow.get("state") != "no_selected_evidence":
        return prompt
    if not isinstance(flow.get("input_source_count"), int) or flow["input_source_count"] <= 0 or flow.get("writer_source_count") != 0:
        raise ValueError("inconsistent evidence handoff")
    record = {key: flow[key] for key in ("input_source_count", "writer_source_count", "failed_node_count",
        "failed_cell_count", "unexamined_job_count")}
    return prompt + (
        "\n\n本次资料处理记录（执行信息，不是产品或原文事实）：" + json.dumps(record, ensure_ascii=False)
        + "\n已有资料进入处理，但没有形成最终可引用证据。请据此解释本次为何还不能可靠回答。"
        "失败数大于零表示对应处理失败；未检查数大于零表示还有工作未执行；"
        "这些数均为零也只表示没有选出证据，不证明原文没有答案。"
        "不要断言资料未记载、原文不存在某事实或产品不支持某能力。"
        "没有入选原文就不能列产品能力、事实数值、排名或推荐；问题中的数字和版本号也不是事实证据。"
        "没有可引用的来源编号，不编造引用。简洁说明这次处理的限制和需要恢复证据核对，解释一次后结束。"
    )
