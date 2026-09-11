"""Bind source provenance, independent review and split audit before token release.

Review names and independence are declarations, not authenticated identities.
This command verifies records and source bytes, never semantic correctness.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from .state_tokens import Vocabulary
from .statetune import SPLITS, audit, digest, export_training, rows, write_json


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def row_sha(row):
    return digest(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def dataset_files(directory):
    return {f"{split}.{kind}.jsonl": file_sha(directory / f"{split}.{kind}.jsonl")
            for split in SPLITS for kind in ("inputs", "gold")}


def paired_rows(directory):
    seen = set()
    for split in SPLITS:
        for item, gold in zip(rows(directory / f"{split}.inputs.jsonl"),
                              rows(directory / f"{split}.gold.jsonl"), strict=True):
            if (item["id"] != gold["id"] or item["split"] != split
                    or gold["split"] != split or item["id"] in seen):
                raise ValueError("review identity/split mismatch")
            seen.add(item["id"])
            yield item, gold


def review_template(directory, output):
    if output.exists():
        raise ValueError("review output already exists")
    pins = dataset_files(directory)
    cases = [{"id": item["id"], "input_sha256": row_sha(item), "gold_sha256": row_sha(gold),
              "decision": "pending", "notes": ""} for item, gold in paired_rows(directory)]
    if not cases or dataset_files(directory) != pins:
        raise ValueError("empty or changing dataset")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"schema": "rwkv_reader_review_v1", "reviewer": "",
                   "independent_of_authors": False, "dataset_files": pins, "cases": cases},
                  stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return {"cases": len(cases), "training_ready": False, "sha256": file_sha(output)}


def verify_review(directory, receipt, pins):
    if (receipt.get("schema") != "rwkv_reader_review_v1"
            or receipt.get("independent_of_authors") is not True
            or receipt.get("dataset_files") != pins):
        raise ValueError("independent review declaration or dataset binding missing")
    reviewer = receipt.get("reviewer", "").strip()
    if not reviewer:
        raise ValueError("independent reviewer required")
    entries = receipt["cases"]
    checked = {entry["id"]: entry for entry in entries}
    pairs = list(paired_rows(directory))
    if len(checked) != len(entries) or set(checked) != {item["id"] for item, _ in pairs}:
        raise ValueError("review coverage incomplete or duplicated")
    for item, gold in pairs:
        author = gold.get("reviewer", "").strip()
        if not author or author.casefold() == reviewer.casefold():
            raise ValueError("independent reviewer must differ from declared author")
        entry = checked[item["id"]]
        if (entry.get("decision") != "approve" or not entry.get("notes", "").strip()
                or entry.get("input_sha256") != row_sha(item)
                or entry.get("gold_sha256") != row_sha(gold)):
            raise ValueError("unapproved or changed reviewed case: " + item["id"])
    return {"reviewer": reviewer, "approved_cases": len(pairs),
            "independence_is_declared_not_authenticated": True,
            "semantic_correctness_verified_by_code": False}


def verify_sources(directory, corpus):
    manifest_path = corpus / "manifest.jsonl"
    before = file_sha(manifest_path)
    manifest = {}
    for entry in rows(manifest_path):
        key = (str(entry["id"]), str(entry["page_id"]), entry["text_sha256"])
        if key in manifest:
            raise ValueError("ambiguous corpus manifest identity")
        manifest[key] = entry
    checked = {}
    for item, _ in paired_rows(directory):
        source = item["source"]
        key = (str(source["document_id"]), str(source["page_id"]), digest(source["snippet"]))
        if key not in manifest:
            raise ValueError("source is not in the bound corpus manifest")
        entry = manifest[key]
        expected = {"title": entry["title"], "uri": entry["url"]}
        if any(source[k] != v for k, v in expected.items()) or any(
                source["metadata"][k] != entry[k] for k in ("license", "revision_url", "dataset_revision")):
            raise ValueError("source provenance metadata changed")
        path = (corpus / entry["text_path"]).resolve()
        if not path.is_relative_to(corpus.resolve()):
            raise ValueError("source path escapes corpus")
        if key not in checked:
            if path.read_bytes() != source["snippet"].encode("utf-8"):
                raise ValueError("source bytes differ from corpus")
            checked[key] = {"document_id": key[0], "page_id": key[1], "sha256": key[2],
                            "text_path": entry["text_path"]}
    if file_sha(manifest_path) != before:
        raise ValueError("corpus manifest changed during release")
    return {"manifest_sha256": before, "sources": list(checked.values())}


def verify_source_snapshot(corpus, provenance):
    if file_sha(corpus / "manifest.jsonl") != provenance["manifest_sha256"]:
        raise ValueError("corpus manifest changed during release")
    for source in provenance["sources"]:
        path = (corpus / source["text_path"]).resolve()
        if not path.is_relative_to(corpus.resolve()) or file_sha(path) != source["sha256"]:
            raise ValueError("corpus source changed during release")


def release(directory, output, corpus, review, vocab_path, max_tokens=4096):
    if output.exists():
        raise ValueError("release already exists")
    if not 256 <= max_tokens <= 4096:
        raise ValueError("current Reader runtime requires 256..4096 token budget")
    pins = dataset_files(directory)
    review_sha, vocab_sha = file_sha(review), file_sha(vocab_path)
    independent = verify_review(directory, json.loads(review.read_text()), pins)
    provenance = verify_sources(directory, corpus)
    if any(item.get("prompt_protocol") != "rwkv_g1j_no_think_v1"
           for item, _ in paired_rows(directory)):
        raise ValueError("new releases require the canonical Reader protocol")
    layouts = {item.get("input_layout", "original") for item, _ in paired_rows(directory)}
    if len(layouts) != 1 or not layouts <= {"original", "task_last"}:
        raise ValueError("new releases require one consistent Reader input layout")
    vocab = Vocabulary(vocab_path)
    report = audit(directory, vocab, max_tokens)
    for split, metrics in report["splits"].items():
        if not all(metrics.get(key, 0) for key in
                   ("positive", "negative", "positive_without_E1", "three_or_more_units")):
            raise ValueError(split + ": insufficient label/position/length coverage")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as name:
        staged = Path(name) / "release"
        staged.mkdir()
        exported = export_training(directory, staged / "train.tokens.jsonl", vocab, max_tokens)
        if (dataset_files(directory) != pins or file_sha(review) != review_sha
                or file_sha(vocab_path) != vocab_sha):
            raise ValueError("release inputs changed during validation")
        verify_source_snapshot(corpus, provenance)
        write_json(staged / "AUDIT.json", report)
        receipt = {"schema": "rwkv_reader_release_v1", "dataset_files": pins,
                   "review_sha256": review_sha, "review": independent,
                   "provenance": provenance, "vocabulary_sha256": vocab_sha,
                   "max_sequence_tokens": max_tokens, "max_generation_tokens": 32,
                   "prompt_protocol": "rwkv_g1j_no_think_v1", "export": exported,
                   "input_layout": next(iter(layouts)),
                   "files": {"train.tokens.jsonl": file_sha(staged / "train.tokens.jsonl"),
                             "AUDIT.json": file_sha(staged / "AUDIT.json")},
                   "training_data_admitted": True, "model_quality_validated": False,
                   "production_promoted": False}
        write_json(staged / "RELEASE.json", receipt)
        # Rename the fully checked directory only after every gate passes.
        if output.exists():
            raise ValueError("release appeared during validation")
        staged.rename(output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("review-template", "release"):
        p = sub.add_parser(name)
        p.add_argument("--input", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if name == "release":
            p.add_argument("--corpus", type=Path, required=True)
            p.add_argument("--review", type=Path, required=True)
            p.add_argument("--vocab", type=Path, required=True)
            p.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    try:
        result = (review_template(args.input, args.output) if args.command == "review-template"
                  else release(args.input, args.output, args.corpus, args.review, args.vocab, args.max_tokens))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as error:
        parser.exit(2, f"release rejected: {error}\n")


if __name__ == "__main__":
    main()
