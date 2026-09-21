"""Candidate Writer protocol: finish the user's decision, preserving raw evidence."""
from .writer_prompt import writer_prompt_v2

DECISION_INSTRUCTIONS = (
    "\n完成任务的方式：\n"
    "先直接回答用户问的结论，再给必要依据。比较题要说明同一口径下的关系，不能只抄两边参数；"
    "口径不同或一侧缺资料时明确说明不能确定比较关系。\n"
    "选择题先逐个核对用户仍有效的全部硬条件，再只在已证实合格的候选中按用户偏好选择。"
    "未知不能说成不满足；没有已证实合格者、并列或证据冲突时如实说明，不能强行选一个。"
    "推荐理由必须与原文数值、单位、版本和范围一致；选中的方案不必在每个指标上都胜过被排除方案。\n"
    "证据列表为空时只说明无法根据现有资料判断，不编造事实或来源。"
    "指令里的对象、版本、日期等是需要保留的限定，不是必须逐项扩写的答案栏目。\n"
    "每个所问结论和必要依据只陈述一次。覆盖用户问题后立即结束回答；"
    "不要继续编号、轮流复述候选参数、重复问题或重复已经说明的未知。"
)


def writer_prompt_decision(task: str, evidence: list[dict], fields: list[str]) -> str:
    return writer_prompt_v2(task, evidence, fields) + DECISION_INSTRUCTIONS
