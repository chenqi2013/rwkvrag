"""Prepare reviewable Writer train drafts using the production prompt and transport.

No inference, embeddings or automatic semantic labels. Targets are supplied by an
author; mechanical checks do not admit them for training. Existing outputs are
never overwritten. This entry does not read dev/heldout data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from llamaindex_retrieval.rwkvos_batch import render_batch_prompt
from llamaindex_retrieval.state_tokens import Vocabulary, encode_training
from llamaindex_retrieval.writer_prompt import writer_prompt_v2

VOCAB_SHA = "8324476023347dec2964625ccb2075c864d250a9c6d9a74f36daba628de8c008"
MAX_SEQUENCE = 8192
GENERATION = 2048


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def build(inputs, targets, vocab):
    if (not inputs or len({r["id"] for r in inputs}) != len(inputs)
            or any(r["split"] != "train" for r in inputs)
            or set(targets["answers"]) != {r["id"] for r in inputs}):
        raise ValueError("unique train-only inputs and exact authored target coverage required")
    first = {}
    for row in inputs:
        first.setdefault(row["source"]["document_id"], row["id"])
    if set(targets["empty_evidence_answers"]) != set(first.values()):
        raise ValueError("one authored empty-evidence variant per document required")
    prepared, tokens = [], []
    for row in inputs:
        source = row["source"]
        text = source["snippet"]
        if digest(text.encode()) != source["metadata"]["source_text_sha256"]:
            raise ValueError("source text hash mismatch")
        evidence, spans = [], []
        for i, unit in enumerate(row["units"], 1):
            start, end = unit["start"], unit["end"]
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                    or text[start:end] != unit["text"]
                    or digest(unit["text"].encode()) != unit["sha256"]):
                raise ValueError("unit text/span/hash mismatch")
            evidence.append({"label": f"资料 {i}",
                "id": f"{source['id']}@{start}:{end}:{unit['sha256'][:12]}",
                "title": source["title"], "uri": source["uri"], "text": unit["text"],
                "context_spans": source["metadata"].get("context_spans", []), "fields": []})
            spans.append({"label": f"资料 {i}", "unit_id": unit["id"],
                          "start": start, "end": end, "sha256": unit["sha256"]})
        variants = [("all_units", evidence, spans, targets["answers"][row["id"]])]
        if row["id"] in targets["empty_evidence_answers"]:
            variants.append(("empty_evidence", [], [], targets["empty_evidence_answers"][row["id"]]))
        for variant, material, coordinates, target in variants:
            if not isinstance(target, str) or not target.strip():
                raise ValueError("nonempty manually authored answer required")
            references = re.findall(r"\[资料([^\]]*)\]", target)
            if target.count("[资料") != len(references):
                raise ValueError("unclosed target citation")
            for reference in references:
                match = re.fullmatch(r"\s*([1-9][0-9]*)", reference)
                if match is None or int(match[1]) > len(material):
                    raise ValueError("target cites an invalid or absent source")
            task = json.dumps({"history": row["history"], "latest_question": row["question"]},
                              ensure_ascii=False)
            body = writer_prompt_v2(task, material, [])
            prompt, _ = render_batch_prompt([{"role": "user", "content": body}], "<think></think>")
            prompt += "\n"  # The canonical Writer transport's exact final LF.
            encoded = encode_training(prompt, target, vocab, MAX_SEQUENCE)
            if encoded["prompt_tokens"] + GENERATION - 1 > MAX_SEQUENCE:
                raise ValueError("complete prompt lacks reserved generation budget; never truncate")
            identity = row["id"].replace("reader_", "writer_", 1) + "_" + variant
            prepared.append({"id": identity, "parent_id": row["id"], "split": "train",
                "variant": variant, "question": row["question"], "history": row["history"],
                "fields": [], "evidence": material, "source_spans": coordinates,
                "origin": {k: source[k] for k in ("id", "document_id", "page_id", "uri")},
                "source_metadata": source["metadata"],
                "prompt": prompt, "prompt_sha256": digest(prompt.encode()),
                "target": target, "target_sha256": digest(target.encode()),
                "prompt_tokens": encoded["prompt_tokens"],
                "target_tokens_with_eos": encoded["target_tokens_with_eos"],
                "author": targets["author"]})
            tokens.append({"id": identity, "prompt_sha256": digest(prompt.encode()), **encoded})
    if len({row["id"] for row in prepared}) != len(prepared):
        raise ValueError("derived Writer identities collide")
    return prepared, tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-inputs", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs_raw, targets_raw = args.train_inputs.read_bytes(), args.targets.read_bytes()
    if digest(args.vocab.read_bytes()) != VOCAB_SHA:
        raise ValueError("canonical vocabulary mismatch")
    rows = [json.loads(line) for line in inputs_raw.splitlines() if line.strip()]
    drafts, tokens = build(rows, json.loads(targets_raw), Vocabulary(args.vocab))
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, values in (("train.drafts.jsonl", drafts), ("train.draft.tokens.jsonl", tokens)):
        raw = "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values).encode()
        (args.output / name).write_bytes(raw)
        hashes[name] = digest(raw)
    receipt = {"schema": "rwkv_writer_draft_v1", "training_data_admitted": False,
        "independent_writer_review_required": True, "quality_evaluated": False,
        "model": "rwkv7-g1j-2.9b-20260831-ctx16384", "model_checkpoint_sha256":
        "966f3420f833532aae3fb1fd6326533b08d43d23b7b03eaa2f0694a30b64a239",
        "source_train_sha256": digest(inputs_raw), "targets_sha256": digest(targets_raw),
        "vocabulary_sha256": VOCAB_SHA, "writer_prompt": "writer_prompt_v2",
        "transport_protocol": "rwkv_g1j_no_think_v1", "train_count": len(drafts),
        "documents": len({row["source"]["document_id"] for row in rows}),
        "all_units": len(rows), "empty_evidence": len(drafts) - len(rows),
        "history_rows": sum(bool(r["history"]) for r in drafts),
        "max_sequence_tokens": MAX_SEQUENCE, "max_generation_tokens": GENERATION,
        "max_prompt_tokens": max(r["prompt_tokens"] for r in tokens),
        "max_actual_sequence_tokens": max(len(r["input_ids"]) for r in tokens),
        "dev_or_heldout_content_read": False, "files": hashes,
        "limitations": ["8 single-document sources, no cross-document reasoning coverage",
            "short histories only", "all-units inputs are a Writer curriculum, not actual Reader outputs",
            "empty-evidence variants reuse their parent's train identity; no extra documents",
            "Reader semantic approval does not approve Writer targets",
            "old 4096-token Reader preflight and training entry are not applicable"]}
    (args.output / "DRAFT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
