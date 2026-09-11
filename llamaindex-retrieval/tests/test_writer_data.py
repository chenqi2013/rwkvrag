"""Writer training/inference boundary and corruption regression checks."""
from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline, conversation
from llamaindex_retrieval.rwkvos_batch import render_batch_prompt
from llamaindex_retrieval.schemas import ConversationMessage, SourceItem
from llamaindex_retrieval.state_tokens import Vocabulary
from llamaindex_retrieval.state_training import read_training_tokens

ROOT = Path(__file__).resolve().parents[1]
BUILD = runpy.run_path(str(ROOT / "statetune/prepare_writer_data.py"))["build"]


@pytest.fixture(scope="module")
def data():
    inputs = [json.loads(line) for line in (ROOT / "statetune/datasets/reader-v6-expanded/dataset/train.inputs.jsonl").read_text().splitlines()]
    targets = json.loads((ROOT / "statetune/datasets/writer-v1/targets.json").read_text())
    vocab = Vocabulary(ROOT / "statetune/assets/rwkv_vocab_v20230424.txt")
    return inputs, targets, vocab


@pytest.mark.asyncio
async def test_all_writer_drafts_match_actual_pipeline_prompt_and_masked_tokens(data):
    drafts, tokens = BUILD(*data)
    calls = []

    class Model:
        async def complete(self, messages, *, stage, assistant_prefill, **kwargs):
            assert stage == "writer"
            prompt, _ = render_batch_prompt(messages, assistant_prefill)
            calls.append(prompt + "\n")

    pipeline = RWKVPipeline(Settings(native_writer_prompt_protocol="evidence_first",
        native_writer_prefill="<think></think"), None, Model())
    for row, encoded in zip(drafts, tokens, strict=True):
        sources = [SourceItem(id=e["id"], document_id=row["origin"]["document_id"],
            source="writer-curriculum", title=e["title"], uri=e["uri"], score=1,
            snippet=e["text"], metadata={"context_spans": e["context_spans"], "field_ids": e["fields"]})
            for e in row["evidence"]]
        task = conversation(row["question"], [ConversationMessage(**m) for m in row["history"]])
        await pipeline._write(task, sources, [])
        assert calls[-1] == row["prompt"]
        ids, n = encoded["input_ids"], encoded["prompt_tokens"]
        assert b"".join(data[2].by_id[i] for i in ids[:n]).decode() == row["prompt"]
        assert b"".join(data[2].by_id[i] for i in ids[n:-1]).decode() == row["target"]
        assert ids[-1] == 0 and encoded["labels"] == [-100] * n + ids[n:]
    assert len(drafts) == 54 and sum(not r["evidence"] for r in drafts) == 8
    assert len({r["origin"]["document_id"] for r in drafts}) == 8


@pytest.mark.parametrize("corruption", ["eval_split", "duplicate_id", "derived_collision", "span", "missing_answer", "empty_citation", "overflow"])
def test_writer_data_rejects_corruption_without_truncation(data, corruption):
    inputs, targets, vocab = deepcopy(data[0]), deepcopy(data[1]), data[2]
    if corruption == "eval_split":
        inputs[0]["split"] = "heldout"
    elif corruption == "duplicate_id":
        inputs[1]["id"] = inputs[0]["id"]
    elif corruption == "derived_collision":
        old = inputs[1]["id"]
        inputs[1]["id"] = inputs[0]["id"].replace("reader_", "writer_", 1)
        targets["answers"][inputs[1]["id"]] = targets["answers"].pop(old)
    elif corruption == "span":
        inputs[0]["units"][0]["start"] += 1
    elif corruption == "missing_answer":
        del targets["answers"][inputs[0]["id"]]
    elif corruption == "empty_citation":
        targets["empty_evidence_answers"][inputs[0]["id"]] += "[资料 1]"
    else:
        inputs[0]["question"] += "问题" * 9000
    with pytest.raises(ValueError):
        BUILD(inputs, targets, vocab)


@pytest.mark.parametrize("citation", ["[资料99]", "[资料 0]", "[资料abc]", "[资料 1", "[资料 01]"])
def test_invalid_citation_cannot_escape_target_check(data, citation):
    inputs, targets, vocab = data[0], deepcopy(data[1]), data[2]
    targets["empty_evidence_answers"][inputs[0]["id"]] += citation
    with pytest.raises(ValueError):
        BUILD(inputs, targets, vocab)


def test_writer_draft_cannot_be_consumed_under_old_reader_budget():
    folder = ROOT / "statetune/datasets/writer-v1/draft"
    receipt = json.loads((folder / "DRAFT.json").read_text())
    assert receipt["training_data_admitted"] is False
    with pytest.raises(ValueError, match="invalid full-sequence"):
        read_training_tokens(folder / "train.draft.tokens.jsonl",
            receipt["files"]["train.draft.tokens.jsonl"], 54)
