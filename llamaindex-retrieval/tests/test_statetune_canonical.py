import json
from pathlib import Path

from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.sandbox import SandboxedEnvironment
import pytest

from llamaindex_retrieval.statetune import render


class GenerationRegion(Extension):
    """HF's generation tag tracks masks; pass through its text for comparison."""
    tags = {"generation"}

    def parse(self, parser):
        token = next(parser.stream)
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.Scope(body).set_lineno(token.lineno)


def test_every_canonical_case_matches_pinned_official_template_and_legacy_is_preserved():
    root = Path(__file__).resolve().parents[1]
    template = SandboxedEnvironment(extensions=[GenerationRegion]).from_string(
        (root / "eval/reader-v3-format-diagnosis-20260910/official-template/chat_template.jinja").read_text())
    count = 0
    for split in ("train", "dev", "heldout"):
        old = [json.loads(s) for s in (root / f"statetune/datasets/reader-v3-pilot/dataset/{split}.inputs.jsonl").read_text().splitlines()]
        new = [json.loads(s) for s in (root / f"statetune/datasets/reader-v4-canonical/dataset/{split}.inputs.jsonl").read_text().splitlines()]
        for a,b in zip(old,new,strict=True):
            original = render(a)
            body = original.removeprefix("User: ").removesuffix("\n\nAssistant: <think></think>")
            official = template.render(messages=[{"role":"user","content":body}],
                                       add_generation_prompt=True,thinking=False)
            assert original == a["prompt"]
            assert render(b) == b["prompt"] == official == original + "\n"
            assert b["input_token_count"] == a["input_token_count"] + 1
            count += 1
    assert count == 43


def test_unknown_prompt_protocol_is_rejected():
    root = Path(__file__).resolve().parents[1]
    row = json.loads((root/"statetune/datasets/reader-v4-canonical/dataset/train.inputs.jsonl").read_text().splitlines()[0])
    with pytest.raises(ValueError,match="unknown dataset prompt protocol"):
        render({**row,"prompt_protocol":"typo"})


def test_task_last_renderer_matches_every_frozen_development_prompt():
    root = Path(__file__).resolve().parents[1]
    old = [json.loads(s) for s in (root / "statetune/datasets/reader-v4-canonical/dataset/dev.inputs.jsonl").read_text().splitlines()]
    new = [json.loads(s) for s in (root / "eval/reader-v5-stability-20260910/question-last/dev.inputs.jsonl").read_text().splitlines()]
    for original, expected in zip(old, new, strict=True):
        assert render({**original, "input_layout": "task_last"}) == expected["prompt"]
        assert render(original) == original["prompt"]
    with pytest.raises(ValueError, match="unsupported Reader input layout"):
        render({**old[0], "input_layout": "unknown"})
