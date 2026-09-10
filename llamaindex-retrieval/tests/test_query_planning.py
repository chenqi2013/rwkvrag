import pytest

from llamaindex_retrieval.query_planning import TaskField, build_query_plan


@pytest.mark.parametrize(
    "question",
    [
        "明朝是因为什么原因走上了灭亡？",
        "秦始皇有哪些伟大成就？",
        "中国四大名著是哪几个？",
        "中国四大名著是哪四个？",
        "西游记是哪个作者写的？",
        "宇树科技创始人是谁？",
        "深圳地铁一号线有哪些站点？",
        "中国有多少个名族？",
        "一个从未出现过的项目支持哪些协议？",
    ],
)
def test_fallback_preserves_question_without_inventing_semantic_contract(question: str) -> None:
    plan = build_query_plan(question)
    assert plan.original_question == question
    assert plan.normalized_question == question
    assert plan.queries == (question,)
    assert plan.subject == ""
    assert plan.relations == ()
    assert plan.analysis.subjects == ()
    assert plan.fields == (TaskField("f1", question, ()),)
    assert plan.context_policy == "none"
    assert plan.merge_strategy == "rank_fusion"
    assert not plan.query_rewritten


def test_fallback_normalizes_script_case_and_whitespace_only() -> None:
    plan = build_query_plan("  臺灣 RWKV 是什麼？  ")
    assert plan.original_question == "  臺灣 RWKV 是什麼？  "
    assert plan.normalized_question == "台湾 rwkv 是什么？"
    assert plan.queries == (plan.normalized_question,)
    assert plan.fields[0].question == plan.normalized_question
    assert plan.query_rewritten
