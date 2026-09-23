"""Guard source identity and multi-turn revision in question-first drafting."""

from generate_question_first_v5 import bind_questions


def draft(history):
    return {"items": [{"index": index, "history": history,
                       "question": f"现在请比较 owner/one 和 owner/two 的实际适用范围？第{index}个场景。"}
                      for index in range(8)]}


def test_history_requires_revision_and_previous_message():
    job = {"focus": "history", "repos": ["owner/one", "owner/two"]}
    good, failures = bind_questions(job, draft([{"role": "user", "content": "过去只看桌面版"}]))
    assert len(good) == 8 and not failures
    bad, failures = bind_questions(job, draft([]))
    assert not bad and len(failures) == 8


def test_missing_registered_repo_rejected():
    job = {"focus": "comparison", "repos": ["owner/one", "owner/two"]}
    value = draft([])
    value["items"][4]["question"] = "现在请比较 owner/one 的适用范围。"
    good, failures = bind_questions(job, value)
    assert len(good) == 7 and failures == [
        {"index": 4, "error": "question omitted registered project"}]
