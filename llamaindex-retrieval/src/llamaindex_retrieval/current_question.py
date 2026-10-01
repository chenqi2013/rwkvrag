"""Opt-in model-authored history correction, with immutable input/output traces."""
import json

from .offline_replay import strict_json

PROTOCOL = "current-question-v1"
G1K_PROTOCOL = "current-question-g1k-v1"


def correction_prompt(question, history):
    return (
        "更新检索问题，不回答问题。后续更正替换旧要求，其余对象、目标和条件保留。"
        "只输出JSON，question填更新后的完整问题。\n"
        + json.dumps({"previous_user_questions": [m.content for m in history if m.role == "user"],
                      "correction": question}, ensure_ascii=False)
    )


def g1k_correction_prompt(question, history):
    """G1K-oriented rewrite prompt; model resolves history, code only passes data."""
    previous = [m.content for m in history if m.role == "user"]
    return (
        '只输出JSON：{"question":"..."}。将最新问题改写为完整问题。\n'
        '如果最新问题含“第一个/第二个/它/刚才”，在上一用户问题中寻找对应的具体对象，'
        '例如“第一是X，第二是Y”时用X或Y替换指代词。\n'
        '如果最新问题含“更正/只/不再/不要”，不要保留被排除的旧对象或旧字段。\n'
        '不要加入最新问题没有要求的旧目标；不要回答问题。\n'
        + json.dumps({"previous_user_questions": previous, "latest_question": question}, ensure_ascii=False)
    )


def parse_question(text):
    value = strict_json(text)
    if not isinstance(value, dict) or set(value) != {"question"}:
        raise ValueError("current question must contain only question")
    question = value["question"]
    if not isinstance(question, str) or not question.strip() or len(question) > 8000:
        raise ValueError("current question must be a nonempty string of at most 8000 characters")
    return question


async def resolve_current_question(pipeline, question, history):
    from .rwkv_pipeline import conversation, structured_body

    original = conversation(question, history)
    protocol = pipeline.settings.native_history_protocol
    if protocol == "raw" or not history:
        return question, original, []
    prompt = g1k_correction_prompt(question, history) if protocol == G1K_PROTOCOL else correction_prompt(question, history)
    result = await pipeline._call(prompt, stage="planner", max_tokens=512)
    result.trace.update({"purpose": "current_question", "history_protocol": protocol,
                         "original_task": original})
    try:
        current = parse_question(structured_body(result))
    except (ValueError, TypeError) as error:
        result.trace["parse_error"] = str(error)
        return None, None, [result.trace]
    result.trace["parsed_question"] = current
    # Only this independently traced model output goes downstream. Code does
    # not infer which words are corrections, merge clauses, or repair its text.
    return current, conversation(current, []), [result.trace]
