"""Recheck V9 source spans, labels, isolation and token masks after checkout."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from llamaindex_retrieval.retrieval_plan import RetrievalPlanV1
from llamaindex_retrieval.state_tokens import Vocabulary, encode_training

from audit_candidates import audit

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "llamaindex-retrieval/statetune/retrieval-v1-20260923"
V4 = ROOT / "llamaindex-retrieval/statetune/retrieval-v4-20260924"
V8 = ROOT / "llamaindex-retrieval/statetune/retrieval-v8-20260924"
V9 = ROOT / "llamaindex-retrieval/statetune/retrieval-v9-20260924"
VOCAB = ROOT / "llamaindex-retrieval/statetune/assets/rwkv_vocab_v20230424.txt"


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def verify():
    selection = json.loads((V9 / "SELECTED-IDS.json").read_text())
    saved = json.loads((V9 / "AUDIT.json").read_text())
    source_file = ROOT / selection["candidate_file"]
    if (file_hash(source_file) != selection["candidate_sha256"] or
            file_hash(V9 / "SELECTED-IDS.json") != saved["selection_sha256"] or
            selection["training_exported"] or saved["admitted_training_rows"]):
        raise ValueError("V9 data binding or status changed")
    all_rows = [json.loads(line) for line in source_file.read_text().splitlines() if line.strip()]
    by_id = {row["id"]: row for row in all_rows}
    ids = selection["selected_ids"]
    if len(by_id) != len(all_rows) or len(ids) != len(set(ids)) or any(key not in by_id for key in ids):
        raise ValueError("duplicate or missing candidate ID")
    selected = [by_id[key] for key in ids]
    source_manifest = json.loads((V4 / "SOURCES-CURATED.json").read_text())["sources"]
    by_family = {source["family"]: source for source in source_manifest}
    by_repo = {source["repo"]: source for source in source_manifest}
    source_bytes = {}
    for source in source_manifest:
        raw = (ROOT / source["readme_path"]).read_bytes()
        if sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError(f"README changed: {source['repo']}")
        source_bytes[source["repo"]] = raw
    vocab = Vocabulary(VOCAB)
    roles = Counter()
    max_tokens = 0
    for row in selected:
        if (sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"] or
                sha256(row["target"].encode()).hexdigest() != row["target_sha256"] or
                row["review"].get("accepted") is not False):
            raise ValueError(f"candidate hash or review status changed: {row['id']}")
        for family, source_hash in zip(row["source_families"], row["source_hashes"], strict=True):
            source = by_family[family]
            if source["sha256"] != source_hash or source["split"] != row["split"]:
                raise ValueError(f"source split or SHA changed: {row['id']}")
        target = json.loads(row["target"])
        if row["role"] == "plan":
            RetrievalPlanV1.model_validate(target, strict=True)
        elif row["role"] == "evidence":
            if target["status"] != row["status"]:
                raise ValueError("evidence label changed")
            expected = []
            for evidence in row["sources"]:
                source = by_repo[evidence["repo"]]
                if evidence["source_sha256"] != source["sha256"]:
                    raise ValueError("evidence source SHA changed")
                start = evidence["block_start_byte"]
                text = evidence["text"]
                raw = source_bytes[evidence["repo"]][start:start + len(text.encode())]
                if raw.decode("utf-8") != text or sha256(raw).hexdigest() != evidence["block_sha256"]:
                    raise ValueError("evidence block byte span changed")
                quote = evidence["quote"]
                if quote is not None:
                    if quote not in text:
                        raise ValueError("evidence quote not in source")
                    expected.append({"source_id": evidence["block_id"], "quote": quote})
            if target["quotes"] != expected or (
                    (target["status"] == "supported") != bool(expected)):
                raise ValueError("evidence target/quote mismatch")
        elif row["role"] == "followup":
            if row["execution_state"] == "no_unresolved_cells":
                if target != {"action": "stop", "query": None,
                              "reason_code": "no_unresolved_cells"}:
                    raise ValueError("stop state has search target")
            else:
                focus = next(cell for cell in row["cells"] if cell["id"] == row["focus_cell"])
                if (target["action"] != "search" or
                        target["reason_code"] != row["execution_state"] or
                        focus["object"] not in target["query"] or
                        target["query"] in row["attempted_queries"]):
                    raise ValueError("follow-up query does not match unresolved cell")
        else:
            raise ValueError("unknown role")
        encoded = encode_training(row["prompt"], row["target"], vocab, 8192)
        if (encoded["labels"][:encoded["prompt_tokens"]] != [-100] * encoded["prompt_tokens"] or
                encoded["labels"][-1] != 0 or encoded["input_ids"][-1] != 0):
            raise ValueError("token mask or EOS changed")
        max_tokens = max(max_tokens, len(encoded["input_ids"]))
        roles[row["role"]] += 1
    report = audit(selected, json.loads((V1 / "EXCLUSIONS.json").read_text()),
                   json.loads((V1 / "MINIMUMS.json").read_text()))
    if (report["admitted"] or len(selected) != saved["rows"] or
            dict(roles) != saved["roles"] or max_tokens != saved["max_tokens"] or
            report["train_kinds"] != saved["kind_counts"] or
            report["largest_train_family_fraction"] != saved["largest_family_fraction"]):
        raise ValueError("V9 saved audit differs from recomputation")
    print(json.dumps({"verified_rows": len(selected), "roles": dict(roles),
                      "max_tokens": max_tokens, "admitted_training_rows": 0},
                     ensure_ascii=False))


if __name__ == "__main__":
    verify()
