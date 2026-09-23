"""Focused checks for exact source spans and source-separated jobs."""

from hashlib import sha256

from build_curated_jobs import build
from build_evidence_blocks import blocks


def test_blocks_keep_utf8_byte_spans_and_heading_context():
    raw = ("# 概览\n\n正文第一行\n正文第二行\n\n- 项目甲\n- 项目乙\n\n"
           "```md\n# 不是标题\n```\n\n## 支持\n| 功能 | 值 |\n| --- | --- |\n").encode()
    result = blocks(raw)
    assert [row["kind"] for row in result] == ["paragraph", "list", "code", "table"]
    assert result[2]["heading_path"] == ["概览"]
    assert result[3]["heading_path"] == ["概览", "支持"]
    assert all(raw[row["start_byte"]:row["end_byte"]] == row["text"].encode()
               and sha256(row["text"].encode()).hexdigest() == row["sha256"]
               for row in result)


def test_jobs_only_use_registered_split_and_balance_sources():
    sources = [{"repo": f"owner/{index}", "family": f"repo:owner/{index}",
                "sha256": f"{index:064x}", "split": "train", "cohort": "api_gateways",
                "role": "api_gateway", "comparison_rule": "Check each project."}
               for index in range(8)]
    sources += [{**source, "repo": f"other/{index}", "family": f"repo:other/{index}",
                 "split": "heldout", "cohort": "note_apps"}
                for index, source in enumerate(sources[:4])]
    first = build(sources, "train", 18, 20260924)
    second = build(sources, "train", 18, 20260924)
    assert first == second and len(first) == 18
    assert all(job["split"] == "train" and all(repo.startswith("owner/") for repo in job["repos"])
               for job in first)
    assert {family for job in first for family in job["source_families"]} == {
        f"repo:owner/{index}" for index in range(8)}
