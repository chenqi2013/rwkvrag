"""Small transport checks on the dedicated 7.2B endpoint, not quality evaluation."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen

from transformers import AutoTokenizer

ROOT = Path("/home/chase/rwkvrag/data/services/g1j72-mainline-20260920")
MODEL_PATH = "/mnt/nas-model/g1j/hf/rwkv7-g1j-7.2b-20260831-ctx16384"
BASE = "http://127.0.0.1:18426"


def request(path, body=None):
    value = None if body is None else json.dumps(body).encode()
    with urlopen(Request(BASE + path, data=value, headers={"Content-Type": "application/json"}), timeout=180) as response:
        return json.loads(response.read())


def main():
    output = ROOT / "smoke"
    output.mkdir(exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    cases = json.loads((ROOT / "smoke-cases.json").read_text())
    models = request("/v1/models")
    (output / "models.json").write_text(json.dumps(models, ensure_ascii=False, indent=2))
    rows = []
    for case in cases:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": case["body"]}],
            tokenize=False, add_generation_prompt=True, rwkv_generation_prompt="fake_think")
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        assert tokenizer.decode(tokens, skip_special_tokens=False) == prompt
        assert len(tokens) + case["max_tokens"] <= 16384
        payload = {"model": "rwkvrag-g1j72-mainline", "prompt": tokens,
            "temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 11,
            "presence_penalty": 0.0, "frequency_penalty": 0.0, "penalty_decay": .996,
            "max_tokens": case["max_tokens"], "stop_token_ids": [0],
            "stop": ["✿", "\nUser:", "\n### User"], "skip_special_tokens": False,
            "return_token_ids": True}
        started = perf_counter()
        response = request("/v1/completions", payload)
        row = {"case": case, "prompt": prompt, "prompt_sha256": sha256(prompt.encode()).hexdigest(),
            "payload": payload, "response": response, "elapsed_ms": (perf_counter() - started) * 1000}
        (output / (case["id"] + ".json")).write_text(json.dumps(row, ensure_ascii=False, indent=2))
        choice = response["choices"][0]
        assert response["usage"]["prompt_tokens"] == len(tokens)
        assert choice["prompt_token_ids"] == tokens
        assert isinstance(choice["text"], str) and choice["text"]
        rows.append({"id": case["id"], "finish_reason": choice["finish_reason"],
                     "input_tokens": len(tokens), "elapsed_ms": row["elapsed_ms"]})
    summary = {"finished_at": datetime.now(timezone.utc).isoformat(), "scope": "transport smoke only; no RAG or general quality claim",
               "rows": rows, "raw_output_modified": False, "state": "zero"}
    (output / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
