"""Freeze prompts and schedule using the model artifact's actual tokenizer/template."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import subprocess

from transformers import AutoTokenizer
from protocol import prompt

HERE = Path(__file__).resolve().parent


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--engine", type=Path, required=True)
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.metadata, local_files_only=True)
    cases = [json.loads(line) for line in (HERE.parent / "reader-label-replication-20260920/cases.jsonl").read_text().splitlines()]
    rendered = []
    for case in cases:
        body = prompt(case)
        text = tokenizer.apply_chat_template([{"role": "user", "content": body}],
            tokenize=False, add_generation_prompt=True, rwkv_generation_prompt="fake_think")
        assert text.endswith("<think></think")
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert len(ids) + 32 <= 4096
        assert tokenizer.decode(ids, skip_special_tokens=False) == text
        rendered.append({"case": case, "body": body, "prompt": text, "prompt_token_ids": ids,
                         "prompt_sha256": sha256(text.encode()).hexdigest()})
    save("inputs.json", rendered)
    conditions = [{"arm": "top1", "seed": s} for s in (11, 101)] + [{"arm": "fake", "seed": s} for s in (11, 29, 47, 71, 101)]
    shuffled = cases.copy()
    random.Random(20260920).shuffle(shuffled)
    schedule = []
    for i, case in enumerate(shuffled):
        rotated = conditions[i % 7:] + conditions[:i % 7]
        schedule.extend({"case_id": case["id"], **c} for c in rotated)
    save("schedule.json", schedule)
    files = sorted(args.engine.joinpath("vllm").rglob("*.py")) + sorted(args.engine.joinpath("vllm").glob("*.so"))
    save("ENGINE-SOURCE.json", {"local_root": str(args.engine),
        "git_head": subprocess.check_output(["git", "-C", str(args.engine), "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "-C", str(args.engine), "status", "--short"], text=True),
        "files": {str(p.relative_to(args.engine)): sha256(p.read_bytes()).hexdigest() for p in files},
        "model_metadata": {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(args.metadata.iterdir()) if p.is_file()}})
    print(json.dumps({"cases": len(cases), "calls": len(schedule), "engine_files": len(files),
                      "token_lengths": [min(len(c["prompt_token_ids"]) for c in rendered), max(len(c["prompt_token_ids"]) for c in rendered)]}))


if __name__ == "__main__":
    main()
