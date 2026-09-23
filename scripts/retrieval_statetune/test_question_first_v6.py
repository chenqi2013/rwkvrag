"""Project subsets must remain aligned to source families and query order."""

from generate_question_first_v6 import bind_questions, join, validate_plans


def test_subset_plan_uses_only_projects_named_in_latest_question():
    job = {"id": "train-example-000", "split": "train", "focus": "comparison",
           "coverage_mode": "grid", "repos": ["owner/one", "owner/two", "owner/three"],
           "source_families": ["repo:owner/one", "repo:owner/two", "repo:owner/three"],
           "source_hashes": ["1" * 64, "2" * 64, "3" * 64]}
    draft = {"items": [{"index": index, "history": [],
                       "question": f"我想比较 owner/one 和 owner/three 的部署方式，第{index}个场景怎么办？"}
                      for index in range(8)]}
    questions, failures = bind_questions(job, draft)
    assert not failures and all(item["selected_indices"] == [0, 2] for item in questions.values())
    raw = {"plans": [{"index": index, "dimensions": ["部署方式"], "conditions": [],
                     "assignments": [], "queries": [
                         {"object": "owner/one", "query": "owner/one 部署方式"},
                         {"object": "owner/three", "query": "owner/three 部署方式"}]}
                    for index in range(8)]}
    plans, failures = validate_plans(job, questions, raw)
    assert not failures and len(plans) == 8
    rows = join(job, questions, plans)
    assert all(row["plan"]["objects"] == ["owner/one", "owner/three"] and
               row["source_families"] == ["repo:owner/one", "repo:owner/three"] and
               row["source_hashes"] == ["1" * 64, "3" * 64] for row in rows)
