"""Teacher output is only a draft, even when its JSON structure is valid."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_plan_drafts import compile_draft, compile_draft_v2, compile_draft_v3


def test_plan_draft_records_only_selected_sources_and_is_not_approved():
    job = {"id": "train-demo-000", "split": "train", "focus": "comparison",
           "repos": ["a/A", "b/B", "c/C"],
           "source_families": ["repo:a/a", "repo:b/b", "repo:c/c"],
           "source_hashes": ["1" * 64, "2" * 64, "3" * 64]}
    item = {"kind": "comparison", "history": [],
            "question": "a/A 和 c/C 的部署与授权条件各是什么？",
            "plan": {"coverage": "grid", "objects": ["a/A", "c/C"],
                     "dimensions": ["部署", "授权条件"], "conditions": [], "listed_pairs": [],
                     "initial_queries": [{"object": "a/A", "query": "a/A 部署 授权条件 文档"},
                                         {"object": "c/C", "query": "c/C 部署 授权条件 文档"}]}}
    rows, failures = compile_draft(job, {"items": [item] * 8})
    assert not failures
    assert len(rows) == 8
    assert rows[0]["source_families"] == ["repo:a/a", "repo:c/c"]
    assert rows[0]["source_hashes"] == ["1" * 64, "3" * 64]
    assert rows[0]["review"]["accepted"] is False


def test_v2_uses_job_kind_without_rewriting_teacher_plan():
    job = {"id": "train-missing-000", "split": "train", "focus": "missing",
           "repos": ["a/A"], "source_families": ["repo:a/a"], "source_hashes": ["1" * 64]}
    item = {"history": [], "question": "a/A 的离线模式文档在哪里？找不到时请说明。",
            "plan": {"coverage": "grid", "objects": ["a/A"], "dimensions": ["离线模式"],
                     "conditions": [], "listed_pairs": [],
                     "initial_queries": [{"object": "a/A", "query": "a/A 离线模式 文档"}]}}
    rows, failures = compile_draft_v2(job, {"items": [item] * 8})
    assert not failures
    assert rows[0]["kind"] == "missing"
    assert rows[0]["plan"] == item["plan"]


def test_v3_rejects_dropped_project_in_registered_comparison():
    job = {"id": "train-compare-000", "split": "train", "focus": "comparison",
           "repos": ["a/A", "b/B", "c/C"],
           "source_families": ["repo:a/a", "repo:b/b", "repo:c/c"],
           "source_hashes": ["1" * 64, "2" * 64, "3" * 64],
           "coverage_mode": "grid", "required_objects": 3}
    item = {"history": [], "question": "比较 a/A 和 b/B 的部署方式。",
            "plan": {"coverage": "grid", "objects": ["a/A", "b/B"],
                     "dimensions": ["部署方式"], "conditions": [], "listed_pairs": [],
                     "initial_queries": [{"object": "a/A", "query": "a/A 部署方式"},
                                         {"object": "b/B", "query": "b/B 部署方式"}]}}
    rows, failures = compile_draft_v3(job, {"items": [item] * 8})
    assert not rows
    assert len(failures) == 8
    assert all("object count differs" in issue["error"] for issue in failures)
