"""Balance V8 review IDs and audit tokenization without exporting training data."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from llamaindex_retrieval.state_tokens import Vocabulary, encode_training

from audit_candidates import audit, normalize

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "llamaindex-retrieval/statetune/retrieval-v1-20260923"
V8 = ROOT / "llamaindex-retrieval/statetune/retrieval-v8-20260924"
V9 = ROOT / "llamaindex-retrieval/statetune/retrieval-v9-20260924"
VOCAB = ROOT / "llamaindex-retrieval/statetune/assets/rwkv_vocab_v20230424.txt"


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(path, value):
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write("\n")


def finalize():
    assembly = json.loads((V8 / "ASSEMBLY-SUMMARY.json").read_text())
    if (assembly["training_started"] or assembly["admitted_training_rows"] or
            assembly["candidate_sha256"] != file_hash(V8 / "CANDIDATES.jsonl") or
            assembly["review_queue_sha256"] != file_hash(V8 / "REVIEW-QUEUE.jsonl")):
        raise ValueError("V8 candidate snapshot changed")
    all_rows = load_rows(V8 / "CANDIDATES.jsonl")
    previous = load_rows(V8 / "REVIEW-QUEUE.jsonl")
    by_id = {row["id"]: row for row in all_rows}
    if len(by_id) != len(all_rows):
        raise ValueError("duplicate V8 candidate ID")
    selected = {row["id"] for row in previous}
    if len(selected) != len(previous) or any(by_id[row["id"]] != row for row in previous):
        raise ValueError("V8 review queue differs from candidate rows")
    incidence = Counter(f for row in previous for f in row["source_families"])
    normalized = {normalize(row["question"]) for row in previous}
    eligible = [row for row in all_rows if row["id"] not in selected and
                row["split"] == "train" and row["role"] == "evidence" and
                row["kind"] == "missing" and row.get("status") == "insufficient" and
                normalize(row["question"]) not in normalized]
    additions = []
    while len(additions) < 41:
        if not eligible:
            raise ValueError("fewer than 41 unique missing-evidence drafts")
        choice = min(eligible, key=lambda row: (
            max(incidence[f] for f in row["source_families"]),
            sum(incidence[f] for f in row["source_families"]), row["id"]))
        additions.append(choice)
        selected.add(choice["id"])
        normalized.add(normalize(choice["question"]))
        for family in choice["source_families"]:
            incidence[family] += 1
        eligible = [row for row in eligible if row["id"] != choice["id"] and
                    normalize(row["question"]) not in normalized]
    review = previous + additions
    minimums = json.loads((V1 / "MINIMUMS.json").read_text())
    exclusions = json.loads((V1 / "EXCLUSIONS.json").read_text())
    report = audit(review, exclusions, minimums)
    if (report["admitted"] or report["train_rows"] != len(review) or
            report["largest_train_family_fraction"] > minimums["max_family_fraction"] or
            report["unique_question_fraction"] < minimums["min_unique_question_fraction"] or
            any(report["train_roles"].get(k, 0) < v for k, v in minimums["roles"].items())):
        raise ValueError("review queue mechanical audit failed unexpectedly")
    vocab = Vocabulary(VOCAB)
    token_lengths = []
    by_role = Counter()
    for row in review:
        encoded = encode_training(row["prompt"], row["target"], vocab, 8192)
        if (encoded["labels"][-1] != 0 or
                encoded["labels"][:encoded["prompt_tokens"]] != [-100] * encoded["prompt_tokens"] or
                encoded["input_ids"][-1] != 0):
            raise ValueError("target mask or EOS mismatch")
        token_lengths.append(len(encoded["input_ids"]))
        by_role[row["role"]] += 1
    issues = Counter("independent semantic review missing" if "independent review not recorded" in x
                     else x.split(":", 1)[-1].strip() for x in report["issues"])
    V9.mkdir(exist_ok=False)
    save(V9 / "SELECTED-IDS.json", {"schema": "retrieval_review_selection_v9",
         "candidate_file": str((V8 / "CANDIDATES.jsonl").relative_to(ROOT)),
         "candidate_sha256": file_hash(V8 / "CANDIDATES.jsonl"),
         "v8_queue_sha256": file_hash(V8 / "REVIEW-QUEUE.jsonl"),
         "selected_ids": [row["id"] for row in review],
         "added_missing_evidence_ids": [row["id"] for row in additions],
         "independent_review": False, "training_exported": False})
    save(V9 / "AUDIT.json", {"schema": "retrieval_review_audit_v9",
         "candidate_sha256": file_hash(V8 / "CANDIDATES.jsonl"),
         "selection_sha256": file_hash(V9 / "SELECTED-IDS.json"),
         "rows": len(review), "roles": dict(by_role),
         "kind_counts": report["train_kinds"], "train_families": report["train_families"],
         "largest_family_fraction": report["largest_train_family_fraction"],
         "unique_question_fraction": report["unique_question_fraction"],
         "max_tokens": max(token_lengths), "min_tokens": min(token_lengths),
         "over_8192": sum(length > 8192 for length in token_lengths),
         "mask_eos_checked": len(review), "audit_admitted": report["admitted"],
         "issue_counts": dict(issues), "issue_examples": report["issues"][:12],
         "independent_review": False, "admitted_training_rows": 0,
         "training_started": False})
    print(json.dumps({"rows": len(review), "roles": dict(by_role),
                      "kinds": report["train_kinds"],
                      "max_tokens": max(token_lengths),
                      "largest_family_fraction": report["largest_train_family_fraction"],
                      "unique_question_fraction": report["unique_question_fraction"],
                      "audit_admitted": report["admitted"]}, ensure_ascii=False))


if __name__ == "__main__":
    finalize()
