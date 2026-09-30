"""Opt-in model-authored history correction, with immutable input/output traces."""
import json

from .offline_replay import strict_json

PROTOCOL = "current-question-v1"


def correction_prompt(question, history):
    return (
        "更新检索问题，不回答问题。后续更正替换旧要求，其余对象、目标和条件保留。"
        "只输出JSON，question填更新后的完整问题。\n"
        + json.dumps({"previous_user_questions": [m.content for m in history if m.role == "user"],
                      "correction": question}, ensure_ascii=False)
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
    if pipeline.settings.native_history_protocol != PROTOCOL or not history:
        return question, original, []
    result = await pipeline._call(correction_prompt(question, history), stage="planner", max_tokens=512)
    result.trace.update({"purpose": "current_question", "history_protocol": PROTOCOL,
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
