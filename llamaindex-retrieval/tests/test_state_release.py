from copy import deepcopy
import json
from pathlib import Path

import pytest

from llamaindex_retrieval import state_release
from llamaindex_retrieval.state_tokens import Vocabulary
from llamaindex_retrieval.statetune import build, digest, units_for, write_rows


@pytest.fixture
def prepared(tmp_path, request):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    annotations, manifest = [], []
    for split in ("train", "dev", "heldout"):
        text = split + "。" + "天地玄黄。" * 500
        units = units_for(text)
        assert len(units) >= 3
        (corpus / f"{split}.txt").write_text(text)
        metadata = {"source_text_sha256": digest(text), "license": "synthetic",
                    "revision_url": f"https://example.test/{split}/1", "dataset_revision": "test",
                    "context_spans": []}
        source = {"id": split, "document_id": split, "page_id": split, "title": split,
                  "uri": f"https://example.test/{split}", "snippet": text, "metadata": metadata}
        manifest.append({"id": split, "page_id": split, "title": split, "url": source["uri"],
                         "text_path": f"{split}.txt", "text_sha256": digest(text),
                         **{k: metadata[k] for k in ("license", "revision_url", "dataset_revision")}})
        for positive in (True, False):
            judgments = [{"unit_id": u["id"], "unit_sha256": u["sha256"], "supported": False,
                          "reason": "合成负例", "evidence": []} for u in units]
            if positive:
                u = units[-1]
                quote = text[u["start"]:u["start"]+2]
                judgments[-1].update(supported=True, reason="合成测试引文", evidence=[{
                    "quote": quote, "quote_sha256": digest(quote),
                    "source_span": {"start": u["start"], "end": u["start"]+2}}])
            annotations.append({"id": split + str(positive), "split": split, "source": source,
                                "input_layout": ("task_last" if split == "dev" else "original")
                                if getattr(request, "param", "original") == "mixed"
                                else getattr(request, "param", "original"),
                                "units": units, "question": "测试问题？", "history": [],
                                "active_tasks": ["测试任务"], "reviewed": True, "reviewer": "author",
                                "unit_judgments": judgments})
    write_rows(corpus / "manifest.jsonl", manifest)
    annotation_path = tmp_path / "annotations.jsonl"
    write_rows(annotation_path, annotations)
    directory = tmp_path / "dataset"
    vocab = Path(__file__).resolve().parents[1] / "statetune/assets/rwkv_vocab_v20230424.txt"
    build(annotation_path, directory, Vocabulary(vocab), 4096)
    review = tmp_path / "review.json"
    state_release.review_template(directory, review)
    receipt = json.loads(review.read_text())
    assert all(r["decision"] == "pending" for r in receipt["cases"])
    receipt.update(reviewer="independent synthetic reviewer", independent_of_authors=True)
    for r in receipt["cases"]:
        r.update(decision="approve", notes="Synthetic test approval, not real semantic evidence.")
    review.write_text(json.dumps(receipt))
    return directory, corpus, review, vocab


@pytest.mark.parametrize("prepared", ["original", "task_last"], indirect=True)
def test_release_binds_full_review_source_audit_and_train_only_tokens(prepared, tmp_path):
    directory, corpus, review, vocab = prepared
    output = tmp_path / "release"
    receipt = state_release.release(directory, output, corpus, review, vocab)
    assert receipt["training_data_admitted"] and not receipt["model_quality_validated"]
    assert receipt["input_layout"] == json.loads((directory / "train.inputs.jsonl").read_text().splitlines()[0])["input_layout"]
    assert receipt["review"]["approved_cases"] == 6
    assert len(receipt["provenance"]["sources"]) == 3
    assert receipt["export"]["evaluation_content_read"] is False
    assert len((output / "train.tokens.jsonl").read_text().splitlines()) == 2
    assert json.loads((output / "AUDIT.json").read_text())["cross_split_verified"]
    assert set(p.name for p in output.iterdir()) == {"RELEASE.json", "AUDIT.json", "train.tokens.jsonl"}
    for name, sha in receipt["files"].items():
        assert state_release.file_sha(output / name) == sha
    with pytest.raises(ValueError, match="already exists"):
        state_release.release(directory, output, corpus, review, vocab)


@pytest.mark.parametrize("change,match", [
    ("author", "differ"), ("pending", "unapproved"), ("duplicate", "coverage"),
    ("missing", "coverage"), ("declaration", "declaration"), ("hash", "binding")])
def test_invalid_review_never_produces_release(prepared, tmp_path, change, match):
    directory, corpus, review, vocab = prepared
    receipt = json.loads(review.read_text())
    if change == "author": receipt["reviewer"] = " AUTHOR "
    elif change == "pending": receipt["cases"][0]["decision"] = "pending"
    elif change == "duplicate": receipt["cases"].append(deepcopy(receipt["cases"][0]))
    elif change == "missing": receipt["cases"].pop()
    elif change == "declaration": receipt["independent_of_authors"] = False
    elif change == "hash": receipt["dataset_files"]["dev.gold.jsonl"] = "wrong"
    review.write_text(json.dumps(receipt))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match=match):
        state_release.release(directory, output, corpus, review, vocab)
    assert not output.exists()


def test_source_corruption_rejected_even_when_review_is_approved(prepared, tmp_path):
    directory, corpus, review, vocab = prepared
    (corpus / "train.txt").write_text("changed corpus bytes")
    with pytest.raises(ValueError, match="source bytes"):
        state_release.release(directory, tmp_path / "rejected", corpus, review, vocab)
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("prepared", ["mixed"], indirect=True)
def test_mixed_train_and_evaluation_layouts_cannot_be_released(prepared, tmp_path):
    directory, corpus, review, vocab = prepared
    with pytest.raises(ValueError, match="consistent Reader input layout"):
        state_release.release(directory, tmp_path / "rejected", corpus, review, vocab)
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("changed", ["dataset", "manifest", "source"])
def test_mutation_during_export_does_not_publish_partial_release(prepared, tmp_path, monkeypatch, changed):
    directory, corpus, review, vocab = prepared
    original = state_release.export_training
    def mutate(*args):
        result = original(*args)
        path = {"dataset": directory / "dev.gold.jsonl", "manifest": corpus / "manifest.jsonl",
                "source": corpus / "train.txt"}[changed]
        with path.open("a") as stream:
            stream.write("\n")
        return result
    monkeypatch.setattr(state_release, "export_training", mutate)
    with pytest.raises(ValueError, match="changed during"):
        state_release.release(directory, tmp_path / "rejected", corpus, review, vocab)
    assert not (tmp_path / "rejected").exists()
