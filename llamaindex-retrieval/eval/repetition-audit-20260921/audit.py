"""Read-only repetition forensics; no generation, rescoring or answer edits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[3]


def digest(value):
    return hashlib.sha256(value).hexdigest()


def onset(text):
    """Locate a repeated line/block, ignoring list/citation counters for inspection.

    This is a locator for already-reviewed loops, not a semantic classifier.
    Offsets and quotes refer to unchanged raw text. Null means no match.
    """
    lines = text.splitlines(keepends=True)
    offsets = []; offset = 0
    for line in lines:
        offsets.append(offset); offset += len(line)
    normalized = [re.sub(r"\[资料[^\]]*\]", "[SOURCE]", re.sub(r"^\s*\d+[.、]\s*", "", line)).strip() for line in lines]
    for second in range(1, len(lines)):
        for width in range(1, min(12, second) + 1):
            first = second - width
            block = normalized[first:second]
            if second + 2 * width <= len(lines) and any(block) and block == normalized[second:second+width] == normalized[second+width:second+2*width]:
                return {"first_cycle_line": first+1, "repeat_begins_line": second+1,
                        "block_lines": width, "repeat_begins_character": offsets[second],
                        "original_first_block": "".join(lines[first:second]),
                        "original_repeated_block": "".join(lines[second:second+width])}
    return None


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    old_archive = ROOT / "artifacts/g1j72-20260920/raw-traces.tar.gz"
    new_archive = ROOT / "artifacts/model-size-paired-20260921/raw-run2.tar.gz"
    review_file = ROOT / "artifacts/model-size-paired-20260921/review1/REVIEW.json"
    inputs_file = ROOT / "llamaindex-retrieval/eval/model-size-paired-20260921/INPUTS.json"
    inputs = json.loads(inputs_file.read_text())
    reviews = json.loads(review_file.read_text())
    rows = {}
    with tarfile.open(new_archive) as archive:
        for member in archive.getmembers():
            if re.search(r"/(2\.9b|7\.2b)-round[12]/\d{4}\.json$", member.name):
                raw = archive.extractfile(member).read()
                row = json.loads(raw)
                assert digest(row["raw_text"].encode()) == row["raw_text_sha256"]
                rows[row["arm"], row["round"], row["ordinal"]] = row
    assert len(rows) == 752
    loops = []
    for review in reviews:
        if review["round"] != 1 or not review["review"]["repetition_confirmed"]:
            continue
        arm, ordinal = review["arm"], review["ordinal"]
        row = rows[arm, 1, ordinal]
        assert row["raw_text_sha256"] == review["raw_answer_sha256"]
        item = {"arm": arm, "ordinal": ordinal, "id": row["id"],
                "question": inputs[ordinal]["question"], "input_tokens": row["usage"]["prompt_tokens"],
                "output_tokens": len(row["output_token_ids"]), "finish_reason": row["finish_reason"],
                "output_contains_eos0": 0 in row["output_token_ids"],
                "round2_identical": row["raw_text"] == rows[arm, 2, ordinal]["raw_text"],
                "evidence_count": len(inputs[ordinal]["evidence"]),
                "raw_answer_sha256": row["raw_text_sha256"], "review_notes": review["review"]["notes"],
                "mechanical_onset_locator": onset(row["raw_text"])}
        loops.append(item)
    history = {}
    with tarfile.open(old_archive) as archive:
        for group in ["direct-29-zero", "direct-29-tuned", "direct-72-zero"]:
            counts = Counter(); length_ids = []; matched = 0; params = None
            for case in inputs[:28]:
                prefix = f'{group}/{case["case"]["id"]}/writer'
                request = json.load(archive.extractfile(prefix + ".started.json"))["request"]
                assert request["contents"][0] == case["prompt"]
                matched += 1
                params = {k: v for k, v in request.items() if k != "contents"}
                choice = json.load(archive.extractfile(prefix + ".completed.json"))["response"]["choices"][0]
                counts[choice["finish_reason"]] += 1
                if choice["finish_reason"] == "length":
                    length_ids.append(case["ordinal"])
            history[group] = {"byte_identical_prompts": matched, "finish_counts": dict(counts),
                              "length_ordinals": length_ids, "request_parameters": params}
    summary = {"scope": "Read-only post-hoc audit; normalized onset is diagnostic, not a new repetition score",
               "new_model_calls": 0, "old28": history, "models": {}}
    for arm in ["2.9b", "7.2b"]:
        selected = [r for r in loops if r["arm"] == arm]
        summary["models"][arm] = {"loops": len(selected),
            "input_token_range": [min(r["input_tokens"] for r in selected), max(r["input_tokens"] for r in selected)],
            "all_loops_at_2048_without_eos0": all(r["output_tokens"] == 2048 and not r["output_contains_eos0"] for r in selected),
            "all_loop_answers_identical_between_rounds": all(r["round2_identical"] for r in selected),
            "old28_finish": dict(Counter(rows[arm, 1, i]["finish_reason"] for i in range(28)))}
    for name, value in [("SUMMARY.json", summary), ("LOOP-LOCATIONS.json", loops)]:
        (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    for ordinal in [43, 97, 124]:
        (out/f"prompt-{ordinal:04d}.txt").write_text(inputs[ordinal]["prompt"])
    (out/"REQUEST-PARAMETERS.json").write_text(json.dumps({k:v for k,v in rows["7.2b",1,97]["request"].items() if k!="prompt"},ensure_ascii=False,indent=2)+"\n")
    (out/"BINDING.json").write_text(json.dumps({str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in [old_archive,new_archive,review_file,inputs_file,Path(__file__).resolve()]},indent=2)+"\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
