"""Model-authored dimensions and questions must keep registered coverage."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_label_first import bind_questions, build_plans


def test_listed_plan_and_question_are_bound_without_answer_rewrite():
    job = {"id": "train-demo-001", "split": "train", "focus": "comparison",
           "coverage_mode": "listed", "repos": ["a/A", "b/B"],
           "source_families": ["repo:a/a", "repo:b/b"],
           "source_hashes": ["1" * 64, "2" * 64]}
    draft = {"plans": [{"dimensions": ["部署", "许可证"], "conditions": [],
                        "assignments": [{"object": "a/A", "dimensions": ["部署"]},
                                        {"object": "b/B", "dimensions": ["许可证"]}],
                        "queries": [{"object": "a/A", "query": "a/A 部署 文档"},
                                    {"object": "b/B", "query": "b/B 许可证 文档"}]}
                       for _ in range(8)]}
    plans, failures = build_plans(job, draft)
    assert len(plans) == 8 and not failures
    questions = {"items": [{"index": i, "history": [],
                            "question": "请分别查 a/A 的部署和 b/B 的许可证。"} for i in range(8)]}
    rows, failures = bind_questions(job, plans, questions)
    assert len(rows) == 1  # Repeated wording does not inflate a training batch.
    assert len(failures) == 7
    assert rows[0]["plan"]["coverage"] == "listed"
    assert rows[0]["review"]["accepted"] is False


def test_question_omitting_project_is_rejected():
    job = {"id": "train-demo-002", "split": "train", "focus": "comparison",
           "coverage_mode": "grid", "repos": ["a/A", "b/B"],
           "source_families": ["repo:a/a", "repo:b/b"],
           "source_hashes": ["1" * 64, "2" * 64]}
    draft = {"plans": [{"dimensions": ["部署"], "conditions": [], "assignments": [],
                        "queries": [{"object": "a/A", "query": "a/A 部署"},
                                    {"object": "b/B", "query": "b/B 部署"}]}
                       for _ in range(8)]}
    plans, _ = build_plans(job, draft)
    questions = {"items": [{"index": i, "history": [], "question": f"请查 a/A 的部署方式第{i}项。"}
                           for i in range(8)]}
    rows, failures = bind_questions(job, plans, questions)
    assert not rows
    assert len(failures) == 8
