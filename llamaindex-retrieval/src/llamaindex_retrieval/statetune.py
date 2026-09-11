"""Portable Reader dataset preparation, validation and masked-token export.

The model/author proposes semantics; this module checks provenance and syntax.
Drafts are never training labels. No embedding, GPU, API or eval-driven relabeling.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

from .rwkv_pipeline import evidence_units, parse_task_selections, task_resolver_prompt
from .rwkvos_batch import render_batch_prompt, render_reader_layout
from .schemas import ConversationMessage
from .state_tokens import Vocabulary, encode_training

SPLITS = ("train", "dev", "heldout")
GENERATION_TOKENS = 32


def digest(value):
    return sha256(value if isinstance(value, bytes) else value.encode("utf-8")).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path, values):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")


def units_for(text):
    return [{"id": f"E{i}", "start": unit.start, "end": unit.end,
             "text": unit.text, "sha256": digest(unit.text)}
            for i, unit in enumerate(evidence_units(0, text, 1200, 180), 1)]


def render(item):
    if (not isinstance(item["question"], str) or not item["question"].strip()
            or not isinstance(item["active_tasks"], list) or not item["active_tasks"]
            or any(not isinstance(t, str) or not t.strip() for t in item["active_tasks"])
            or not isinstance(item["history"], list)):
        raise ValueError("question, conversation history and explicit active tasks required")
    for message in item["history"]:
        ConversationMessage.model_validate(message, strict=True)
    task = json.dumps({"history": item["history"], "latest_question": item["question"]},
                      ensure_ascii=False)
    units = [SimpleNamespace(**unit) for unit in item["units"]]
    message = task_resolver_prompt(task, item["active_tasks"], SimpleNamespace(**item["source"]), units)
    prompt = render_batch_prompt([{"role": "user", "content": message}], "<think></think")[0]
    protocol = item.get("prompt_protocol", "batch_complete_v1")
    if protocol == "rwkv_g1j_no_think_v1":
        return render_reader_layout(prompt + "\n", item.get("input_layout", "original"))
    if protocol != "batch_complete_v1":
        raise ValueError("unknown dataset prompt protocol")
    if item.get("input_layout", "original") != "original":
        raise ValueError("Reader task_last layout requires canonical prompt protocol")
    return prompt


def validate_case(item, gold, vocab, max_tokens):
    if item["id"] != gold["id"] or item["split"] != gold["split"]:
        raise ValueError("input/gold identity mismatch")
    if item["split"] not in SPLITS:
        raise ValueError("unknown split")
    text = item["source"]["snippet"]
    if digest(text) != gold["bindings"]["source_text_sha256"]:
        raise ValueError("source hash mismatch")
    if digest(text) != item["source"]["metadata"]["source_text_sha256"]:
        raise ValueError("source differs from prepared corpus binding")
    actual = units_for(text)
    if not actual:
        raise ValueError("empty source")
    if [{k: unit[k] for k in actual[0]} for unit in item["units"]] != actual:
        raise ValueError("unit text, offsets or hashes differ from production slicing")
    prompt = render(item)
    if prompt != item["prompt"] or digest(prompt) != item["prompt_sha256"]:
        raise ValueError("prompt differs from current production protocol")
    if digest(prompt) != gold["bindings"]["prompt_sha256"]:
        raise ValueError("gold is bound to a different prompt")
    judgments = gold["unit_judgments"]
    if [j["unit_id"] for j in judgments] != [u["id"] for u in actual]:
        raise ValueError("every unit needs one explicit semantic judgment")
    selected = []
    for unit, judgment in zip(actual, judgments, strict=True):
        if (type(judgment["supported"]) is not bool or not judgment["reason"].strip()
                or judgment["unit_sha256"] != unit["sha256"]):
            raise ValueError("invalid unit judgment")
        if judgment["supported"] != bool(judgment["evidence"]):
            raise ValueError("positive judgments require source quotes")
        for support in judgment["evidence"]:
            a, b = support["source_span"]["start"], support["source_span"]["end"]
            if (type(a) is not int or type(b) is not int
                    or not unit["start"] <= a < b <= unit["end"]
                    or text[a:b] != support["quote"]
                    or digest(support["quote"]) != support["quote_sha256"]):
                raise ValueError("support quote is not the exact source slice")
        if judgment["supported"]:
            selected.append(unit["id"])
    target = ",".join(selected) if selected else "NONE"
    if target != gold["target"] or selected != gold["expected_unit_ids"]:
        raise ValueError("target disagrees with explicit judgments")
    parse_task_selections(target, len(actual))
    encoded = encode_training(prompt, target, vocab, max_tokens)
    if encoded["prompt_tokens"] != item["input_token_count"]:
        raise ValueError("prompt token count mismatch")
    # The final generated token needs no further forward pass; all earlier
    # generated tokens must fit beside the full, untruncated source prompt.
    if encoded["prompt_tokens"] + GENERATION_TOKENS - 1 > max_tokens:
        raise ValueError("prompt leaves insufficient room for 32-token natural generation")
    return encoded


def audit(directory, vocab, max_tokens=4096, splits=SPLITS):
    documents, pages, hashes, identities, summary = {}, {}, {}, set(), {}
    for split in splits:
        inputs, golds = rows(directory / f"{split}.inputs.jsonl"), rows(directory / f"{split}.gold.jsonl")
        if not inputs or len(inputs) != len(golds):
            raise ValueError(f"{split}: empty or unmatched inputs/gold")
        metrics, lengths = Counter(), []
        for item, gold in zip(inputs, golds, strict=True):
            if item["split"] != split or item["id"] in identities:
                raise ValueError("duplicate case identity or wrong split file")
            identities.add(item["id"])
            encoded = validate_case(item, gold, vocab, max_tokens)
            source = item["source"]
            for mapping, key in ((documents, str(source["document_id"])),
                                 (pages, str(source["page_id"])),
                                 (hashes, digest(source["snippet"]))):
                if key in mapping and mapping[key] != split:
                    raise ValueError("document or identical source leaked across splits")
                mapping[key] = split
            expected = gold["expected_unit_ids"]
            metrics["positive" if expected else "negative"] += 1
            if expected:
                metrics["positive_without_E1"] += "E1" not in expected
                metrics["positive_multiple_units"] += len(expected) > 1
            metrics["three_or_more_units"] += len(item["units"]) >= 3
            lengths.append(len(encoded["input_ids"]))
        summary[split] = {"samples": len(inputs), **metrics, "max_sequence_tokens": max(lengths)}
    return {"splits": summary, "checked_splits": list(splits),
            "cross_split_verified": len(splits) == len(SPLITS),
            "document_disjoint_in_checked_splits": True, "content_disjoint_in_checked_splits": True,
            "semantic_labels_verified_by_code": False,
            "scope": "integrity and protocol checks; source semantics require author review"}


def prepare(corpus, output, count, excluded, seed, min_units=3, *,
            vocab=None, max_tokens=4096, reserve_tokens=512, split_counts=None, input_layout="original"):
    """Choose complete sources; write unlabelled work orders, never invented gold."""
    if output.exists():
        raise ValueError("output already exists")
    if count < 1 or count > 256:
        raise ValueError("prepare budget is 1..256 documents")
    if type(min_units) is not int or not 1 <= min_units <= 5:
        raise ValueError("min_units must be 1..5")
    if input_layout not in {"original", "task_last"}:
        raise ValueError("unsupported Reader input layout")
    if split_counts is not None and (set(split_counts) != set(SPLITS)
            or any(type(n) is not int or n < 1 for n in split_counts.values())
            or sum(split_counts.values()) != count):
        raise ValueError("split counts must be positive train/dev/heldout counts summing to documents")
    if vocab is not None and not 0 < reserve_tokens < max_tokens:
        raise ValueError("authoring reserve must be positive and smaller than sequence budget")
    excluded_ids, excluded_hashes, excluded_document_ids = set(), set(), set()
    for directory in excluded:
        exclusions = directory / "EXCLUSIONS.json"
        if exclusions.exists():
            ledger = json.loads(exclusions.read_text())
            for key in ("excluded_page_ids", "excluded_document_ids"):
                values = ledger.get(key, [])
                if not isinstance(values, list) or any(type(v) not in (str, int) or not str(v) for v in values):
                    raise ValueError("exclusion identities must be lists of strings or integers")
            hashes = ledger.get("excluded_text_sha256", [])
            if not isinstance(hashes, list) or any(not isinstance(h, str) or len(h) != 64
                    or any(c not in "0123456789abcdef" for c in h) for h in hashes):
                raise ValueError("exclusion hashes must be a list of SHA256 values")
            excluded_ids.update(str(v) for v in ledger["excluded_page_ids"])
            excluded_hashes.update(hashes)
            excluded_document_ids.update(str(v) for v in ledger.get("excluded_document_ids", []))
        input_paths = [directory / f"{split}.inputs.jsonl" for split in SPLITS]
        if any(p.exists() for p in input_paths) and not all(p.exists() for p in input_paths):
            raise ValueError("excluded dataset has incomplete split inputs")
        if not any(p.exists() for p in input_paths) and not exclusions.exists():
            raise ValueError("exclusion directory needs a dataset or EXCLUSIONS.json ledger")
        for path in input_paths:
            for item in rows(path) if path.exists() else []:
                excluded_ids.add(str(item["source"]["page_id"]))
                excluded_hashes.add(digest(item["source"]["snippet"]))
                excluded_document_ids.add(str(item["source"]["document_id"]))
    manifest = rows(corpus / "manifest.jsonl")
    manifest.sort(key=lambda row: digest(f"{seed}|{row['text_sha256']}"))
    drafts, seen, seen_pages, seen_documents, over_budget = [], set(), set(), set(), 0
    selected_counts = Counter()
    for row in manifest:
        if (str(row["page_id"]) in excluded_ids or row["text_sha256"] in excluded_hashes
                or str(row["id"]) in excluded_document_ids
                or row["text_sha256"] in seen or str(row["page_id"]) in seen_pages
                or str(row["id"]) in seen_documents):
            continue
        path = (corpus / row["text_path"]).resolve()
        if not path.is_relative_to(corpus.resolve()):
            raise ValueError("source path escapes corpus")
        raw = path.read_bytes()
        if digest(raw) != row["text_sha256"]:
            raise ValueError("corpus manifest hash mismatch")
        text = raw.decode("utf-8")
        units = units_for(text)
        if len(text) > 6000 or not min_units <= len(units) <= 5:
            continue
        bucket = int(digest(seed + "|" + digest(raw)), 16) % 10
        split = "train" if bucket < 6 else "dev" if bucket < 8 else "heldout"
        if split_counts is not None and selected_counts[split] >= split_counts[split]:
            continue
        source = {"id": "reader:" + digest(raw), "document_id": row["id"],
                  "page_id": row["page_id"], "title": row["title"], "uri": row["url"],
                  "snippet": text, "metadata": {"source_text_sha256": digest(raw),
                  "license": row["license"], "revision_url": row["revision_url"],
                  "dataset_revision": row["dataset_revision"], "context_spans": []}}
        draft = {"id": f"reader-{row['page_id']}", "split": split, "source": source,
                       "prompt_protocol": "rwkv_g1j_no_think_v1",
                       "input_layout": input_layout,
                       "units": units, "reviewed": False, "question": "", "history": [],
                       "active_tasks": [], "unit_judgments": [],
                       "coverage_requests": ["E2-only", "E3-only", "multi-unit", "unsupported",
                                             "history correction", "partial answer support"]}
        if vocab is not None:
            probe = {**draft, "question": "待编写问题", "active_tasks": ["待编写问题"]}
            tokens = len(vocab.encode(render(probe)))
            if tokens + reserve_tokens > max_tokens:
                over_budget += 1
                continue
            draft["authoring_budget"] = {"provisional_prompt_tokens": tokens,
                                         "reserve_tokens": reserve_tokens,
                                         "max_sequence_tokens": max_tokens,
                                         "final_prompt_must_be_rechecked": True}
        seen.add(digest(raw))
        seen_pages.add(str(row["page_id"]))
        seen_documents.add(str(row["id"]))
        drafts.append(draft)
        selected_counts[split] += 1
        if len(drafts) == count:
            break
    if len(drafts) < count:
        raise ValueError(f"only {len(drafts)} eligible complete documents; requested {count}")
    output.mkdir(parents=True)
    for split in SPLITS:
        write_rows(output / f"{split}.drafts.jsonl", [d for d in drafts if d["split"] == split])
    write_json(output / "EXCLUSIONS.json", {"schema": "rwkv_reader_exclusions_v1",
               "excluded_page_ids": sorted(excluded_ids | seen_pages),
               "excluded_document_ids": sorted(excluded_document_ids | seen_documents),
               "excluded_text_sha256": sorted(excluded_hashes | seen)})
    write_json(output / "PREPARED.json", {"documents": count, "seed": seed,
               "counts": dict(Counter(d["split"] for d in drafts)), "training_ready": False,
               "requested_split_counts": split_counts,
               "excluded_documents": len(excluded_ids), "model_calls": 0,
               "token_screening_enabled": vocab is not None, "over_budget_skipped": over_budget})
    return {"documents": count, "training_ready": False}


def build(annotations, output, vocab, max_tokens):
    """Accept reviewed judgments explicitly; reject drafts and partial outputs."""
    if output.exists():
        raise ValueError("output already exists")
    items, golds = {s: [] for s in SPLITS}, {s: [] for s in SPLITS}
    for row in rows(annotations):
        if row.get("reviewed") is not True or not row.get("reviewer", "").strip():
            raise ValueError("unreviewed draft cannot become training data")
        split = row["split"]
        if split not in SPLITS:
            raise ValueError("invalid split")
        item = {key: row[key] for key in ("id", "split", "source", "units", "question", "history", "active_tasks")}
        # New builds use the canonical boundary. Missing protocol is interpreted
        # as legacy only when reading already-frozen datasets in render/audit.
        item["prompt_protocol"] = row.get("prompt_protocol", "rwkv_g1j_no_think_v1")
        item["input_layout"] = row.get("input_layout", "original")
        item["prompt"] = render(item)
        item["prompt_sha256"] = digest(item["prompt"])
        item["input_token_count"] = len(vocab.encode(item["prompt"]))
        selected = [j["unit_id"] for j in row["unit_judgments"] if j["supported"]]
        gold = {"id": row["id"], "split": split, "target": ",".join(selected) if selected else "NONE",
                "expected_unit_ids": selected, "unit_judgments": row["unit_judgments"],
                "reviewer": row["reviewer"], "bindings": {
                    "source_text_sha256": digest(row["source"]["snippet"]),
                    "prompt_sha256": item["prompt_sha256"]}}
        validate_case(item, gold, vocab, max_tokens)
        items[split].append(item); golds[split].append(gold)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        temp = Path(tmp) / "dataset"; temp.mkdir()
        for split in SPLITS:
            write_rows(temp / f"{split}.inputs.jsonl", items[split])
            write_rows(temp / f"{split}.gold.jsonl", golds[split])
        report = audit(temp, vocab, max_tokens)
        write_json(temp / "AUDIT.json", report)
        shutil.move(str(temp), output)
    return report


def export_training(directory, output, vocab, max_tokens):
    if output.exists():
        raise ValueError("export already exists")
    # Only train content is opened. Evaluation files are never exported to a trainer.
    report = audit(directory, vocab, max_tokens, splits=("train",))
    inputs, golds = rows(directory / "train.inputs.jsonl"), rows(directory / "train.gold.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = [{"id": item["id"], "prompt_sha256": item["prompt_sha256"],
                **validate_case(item, gold, vocab, max_tokens)}
               for item, gold in zip(inputs, golds, strict=True)]
    with output.open("x", encoding="utf-8") as stream:
        for item in encoded:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    return {**report, "exported": len(encoded), "sha256": digest(output.read_bytes()),
            "teacher_forcing": "logits(input_ids[:-1]) predict labels[1:]; ignore=-100; EOS=0",
            "evaluation_content_read": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    draft = sub.add_parser("prepare")
    draft.add_argument("--corpus", type=Path, required=True)
    draft.add_argument("--output", type=Path, required=True)
    draft.add_argument("--documents", type=int, default=24)
    draft.add_argument("--exclude-dataset", type=Path, action="append", default=[])
    draft.add_argument("--seed", default="rwkv-reader-v3")
    draft.add_argument("--min-units", type=int, default=3)
    draft.add_argument("--vocab", type=Path,
                       help="Screen full provisional prompts before authoring questions")
    draft.add_argument("--max-tokens", type=int, default=4096)
    draft.add_argument("--reserve-tokens", type=int, default=512)
    draft.add_argument("--split-counts", type=int, nargs=3, metavar=("TRAIN", "DEV", "HELDOUT"),
                       help="Document quotas, summing to --documents; keep each source's hash-assigned split")
    draft.add_argument("--input-layout", choices=("original", "task_last"), default="original")
    for command in ("audit", "build", "export"):
        p = sub.add_parser(command)
        p.add_argument("--vocab", type=Path, required=True)
        p.add_argument("--max-tokens", type=int, default=4096)
        p.add_argument("--input", type=Path, required=True)
        p.add_argument("--require-coverage", action="store_true",
                       help="Reject datasets lacking positive evidence beyond E1 or 3+ units")
        if command != "audit":
            p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if not 256 <= args.max_tokens <= 32768:
            raise ValueError("sequence budget must be 256..32768 tokens")
        if args.command == "prepare":
            result = prepare(args.corpus, args.output, args.documents, args.exclude_dataset,
                             args.seed, args.min_units,
                             vocab=Vocabulary(args.vocab) if args.vocab else None,
                             max_tokens=args.max_tokens, reserve_tokens=args.reserve_tokens,
                             split_counts=dict(zip(SPLITS, args.split_counts)) if args.split_counts else None,
                             input_layout=args.input_layout)
        else:
            vocab = Vocabulary(args.vocab)
            if args.require_coverage:
                if args.command == "build":
                    raise ValueError("use audit --require-coverage after building the reviewed dataset")
                checked = audit(args.input, vocab, args.max_tokens,
                                splits=("train",) if args.command == "export" else SPLITS)
                for split, metrics in checked["splits"].items():
                    if not (metrics.get("positive_without_E1", 0) and metrics.get("three_or_more_units", 0)
                            and metrics.get("positive", 0) and metrics.get("negative", 0)):
                        raise ValueError(f"{split}: insufficient label/position/length coverage for expansion")
            if args.command == "audit": result = audit(args.input, vocab, args.max_tokens)
            elif args.command == "build": result = build(args.input, args.output, vocab, args.max_tokens)
            else: result = export_training(args.input, args.output, vocab, args.max_tokens)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"dataset rejected: {error}\n")


if __name__ == "__main__":
    main()
