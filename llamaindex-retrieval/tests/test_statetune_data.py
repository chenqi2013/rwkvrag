from copy import deepcopy
import json
from pathlib import Path

import pytest

from llamaindex_retrieval.statetune import (
    audit, build, digest, export_training, prepare, render, units_for, validate_case, write_rows,
)
from llamaindex_retrieval.state_tokens import encode_training


class ByteVocab:
    def encode(self, text):
        return [b + 1 for b in text.encode()]


def annotation(identity, split):
    text = "原文" + identity + "。" * 800
    units = units_for(text)
    judgments = [{"unit_id": u["id"], "unit_sha256": u["sha256"], "supported": False,
                  "reason": "本单元没有目标信息", "evidence": []} for u in units]
    u = units[-1]
    quote = text[u["start"]:u["start"] + 2]
    judgments[-1].update(supported=True, reason="原文直接支持目标", evidence=[{
        "quote": quote, "quote_sha256": digest(quote),
        "source_span": {"start": u["start"], "end": u["start"] + 2}}])
    return {"id": identity, "split": split, "reviewed": True, "reviewer": "synthetic test",
            "source": {"id": identity, "document_id": identity, "page_id": identity,
                       "title": identity, "uri": None, "snippet": text,
                       "metadata": {"source_text_sha256": digest(text), "context_spans": []}},
            "units": units, "question": "需要什么？", "history": [], "active_tasks": ["目标"],
            "unit_judgments": judgments}


def built(tmp_path):
    annotations = tmp_path / "annotations.jsonl"
    write_rows(annotations, [annotation(s, s) for s in ("train", "dev", "heldout")])
    output = tmp_path / "dataset"
    build(annotations, output, ByteVocab(), 10000)
    return output


def test_target_mask_and_eos_keep_prompt_answer_boundary():
    row = encode_training("问😀", "E2", ByteVocab(), 100)
    n = row["prompt_tokens"]
    assert row["labels"][:n] == [-100] * n
    assert row["labels"][n:] == ByteVocab().encode("E2") + [0]
    assert row["input_ids"][-1] == 0
    with pytest.raises(ValueError, match="never truncate"):
        encode_training("问😀", "E2", ByteVocab(), 1)


def test_build_audit_export_roundtrip_and_train_export_never_reads_eval(tmp_path, monkeypatch):
    directory = built(tmp_path)
    first = json.loads((directory / "train.inputs.jsonl").read_text())
    assert first["prompt_protocol"] == "rwkv_g1j_no_think_v1"
    assert first["prompt"].endswith("Assistant: <think></think>\n")
    report = audit(directory, ByteVocab(), 10000)
    assert report["cross_split_verified"]
    original = Path.read_text
    def guarded(path, *args, **kwargs):
        if path.name.startswith(("dev.", "heldout.")):
            raise AssertionError("trainer read evaluation content")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)
    exported = export_training(directory, tmp_path / "train.jsonl", ByteVocab(), 10000)
    assert exported["exported"] == 1
    assert exported["evaluation_content_read"] is False
    assert not exported["cross_split_verified"]


def test_unreviewed_drafts_and_corrupt_support_cannot_become_gold(tmp_path):
    path = tmp_path / "annotations.jsonl"
    row = annotation("a", "train")
    row["reviewed"] = False
    write_rows(path, [row])
    with pytest.raises(ValueError, match="unreviewed"):
        build(path, tmp_path / "bad", ByteVocab(), 10000)
    row["reviewed"] = True
    row["unit_judgments"][-1]["evidence"][0]["quote"] = "伪造"
    write_rows(path, [row])
    with pytest.raises(ValueError, match="exact source"):
        build(path, tmp_path / "bad", ByteVocab(), 10000)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("key", ["document_id", "page_id"])
def test_same_document_across_splits_is_rejected(tmp_path, key):
    annotations = [annotation(s, s) for s in ("train", "dev", "heldout")]
    annotations[1]["source"][key] = "train"
    path = tmp_path / "annotations.jsonl"
    write_rows(path, annotations)
    with pytest.raises(ValueError, match="leaked"):
        build(path, tmp_path / "bad", ByteVocab(), 10000)


def test_document_identity_type_cannot_hide_cross_split_leakage(tmp_path):
    annotations = [annotation(s, s) for s in ("train", "dev", "heldout")]
    annotations[0]["source"]["document_id"] = 123
    annotations[1]["source"]["document_id"] = "123"
    path = tmp_path / "annotations.jsonl"
    write_rows(path, annotations)
    with pytest.raises(ValueError, match="leaked"):
        build(path, tmp_path / "bad", ByteVocab(), 10000)


def test_current_protocol_or_unit_tampering_is_detected(tmp_path):
    directory = built(tmp_path)
    item = json.loads((directory / "train.inputs.jsonl").read_text())
    gold = json.loads((directory / "train.gold.jsonl").read_text())
    for change in ("prompt", "offset"):
        changed = deepcopy(item)
        if change == "prompt": changed["prompt"] += "extra"
        else: changed["units"][0]["start"] += 1
        with pytest.raises(ValueError):
            validate_case(changed, gold, ByteVocab(), 10000)


def test_short_target_cannot_hide_insufficient_generation_room(tmp_path):
    directory = built(tmp_path)
    item = json.loads((directory / "train.inputs.jsonl").read_text())
    gold = json.loads((directory / "train.gold.jsonl").read_text())
    prompt_length = item["input_token_count"]
    # The target and EOS fit, but a capped natural generation would not.
    with pytest.raises(ValueError, match="32-token natural generation"):
        validate_case(item, gold, ByteVocab(), prompt_length + 30)
    encoded = validate_case(item, gold, ByteVocab(), prompt_length + 31)
    assert encoded["prompt_tokens"] == prompt_length


def test_prepare_screens_full_prompt_tokens_and_keeps_complete_source(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    texts = {"short": "短句。" * 850, "long": "长句。" * 1350}
    manifest = []
    for identity, text in texts.items():
        (corpus / (identity + ".md")).write_text(text)
        manifest.append({"id": identity, "page_id": identity, "title": identity,
                         "text_path": identity + ".md", "text_sha256": digest(text),
                         "url": "https://example.test/" + identity, "license": "test",
                         "revision_url": "https://example.test/revision/1", "dataset_revision": "test"})
    write_rows(corpus / "manifest.jsonl", manifest)
    output = tmp_path / "drafts"
    prepare(corpus, output, 1, [], "test", vocab=ByteVocab(), max_tokens=11000, reserve_tokens=512)
    drafts = [json.loads(line) for p in output.glob("*.drafts.jsonl") for line in p.read_text().splitlines()]
    assert len(drafts) == 1
    assert drafts[0]["source"]["snippet"] == texts["short"]
    assert not drafts[0]["reviewed"]
    assert drafts[0]["prompt_protocol"] == "rwkv_g1j_no_think_v1"
    assert drafts[0]["authoring_budget"]["final_prompt_must_be_rechecked"]
    with pytest.raises(ValueError, match="reserve"):
        prepare(corpus, tmp_path / "bad", 1, [], "test", vocab=ByteVocab(),
                max_tokens=4096, reserve_tokens=4096)


@pytest.mark.parametrize("identity_key", ["id", "page_id"])
def test_prepare_selects_only_one_revision_per_document(tmp_path, identity_key):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest = []
    for index in range(3):
        text = str(index) + "短句。" * 850
        (corpus / f"{index}.md").write_text(text)
        manifest.append({"id": str(index), "page_id": str(index), "title": str(index),
                         "text_path": f"{index}.md", "text_sha256": digest(text),
                         "url": "https://example.test", "license": "test",
                         "revision_url": "https://example.test/revision", "dataset_revision": "test"})
    manifest[0][identity_key] = 123
    manifest[1][identity_key] = "123"
    write_rows(corpus / "manifest.jsonl", manifest)
    output = tmp_path / "drafts"
    prepare(corpus, output, 2, [], "test")
    selected = [json.loads(s) for p in output.glob("*.drafts.jsonl") for s in p.read_text().splitlines()]
    assert len({str(r["source"]["document_id" if identity_key == "id" else "page_id"])
                for r in selected}) == 2
    with pytest.raises(ValueError, match="only 2 eligible"):
        prepare(corpus, tmp_path / "too_many", 3, [], "test")


def test_prepare_document_quotas_keep_content_hash_split_assignment(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest = []
    for index in range(30):
        text = str(index) + "短句。" * 850
        (corpus / f"{index}.md").write_text(text)
        manifest.append({"id": str(index), "page_id": str(index), "title": str(index),
                         "text_path": f"{index}.md", "text_sha256": digest(text),
                         "url": "https://example.test", "license": "test",
                         "revision_url": "https://example.test/revision", "dataset_revision": "test"})
    write_rows(corpus / "manifest.jsonl", manifest)
    quotas = {"train": 2, "dev": 2, "heldout": 2}
    output = tmp_path / "balanced"
    prepare(corpus, output, 6, [], "quota-test", split_counts=quotas)
    for split in quotas:
        selected = [json.loads(s) for s in (output / f"{split}.drafts.jsonl").read_text().splitlines()]
        assert len(selected) == quotas[split]
        for item in selected:
            bucket = int(digest("quota-test|" + digest(item["source"]["snippet"])), 16) % 10
            assert item["split"] == ("train" if bucket < 6 else "dev" if bucket < 8 else "heldout")
    with pytest.raises(ValueError, match="summing"):
        prepare(corpus, tmp_path / "invalid", 7, [], "test", split_counts=quotas)


def test_prepared_exclusion_ledger_blocks_page_document_and_content_aliases(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    def entry(identity, page, text, filename):
        (corpus / filename).write_text(text)
        return {"id": identity, "page_id": page, "title": str(identity), "text_path": filename,
                "text_sha256": digest(text), "url": "https://example.test", "license": "test",
                "revision_url": "https://example.test/revision", "dataset_revision": "test"}
    text = "原始" + "短句。" * 850
    original = entry("document", "page", text, "original.md")
    write_rows(corpus / "manifest.jsonl", [original])
    first = tmp_path / "first"
    prepare(corpus, first, 1, [], "test")
    candidates = [original, entry("newdoc", "newpage", text, "copy.md"),
                  entry("otherdoc", "page", text + "版本二", "revision.md"),
                  entry("document", "otherpage", text + "版本三", "doc-alias.md"),
                  entry("fresh", "fresh", "全新" + "短句。" * 850, "fresh.md")]
    write_rows(corpus / "manifest.jsonl", candidates)
    second = tmp_path / "second"
    prepare(corpus, second, 1, [first], "test")
    selected = [json.loads(s) for p in second.glob("*.drafts.jsonl") for s in p.read_text().splitlines()]
    assert [r["source"]["document_id"] for r in selected] == ["fresh"]
    ledger = json.loads((second / "EXCLUSIONS.json").read_text())
    assert set(ledger["excluded_page_ids"]) == {"page", "fresh"}
    # A partial dataset must not silently pretend its ledger covers missing splits.
    (first / "train.inputs.jsonl").write_text("")
    with pytest.raises(ValueError, match="incomplete split"):
        prepare(corpus, tmp_path / "rejected", 1, [first], "test")
