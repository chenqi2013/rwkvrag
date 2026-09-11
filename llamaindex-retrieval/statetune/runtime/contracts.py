"""CPU-only contracts. No torch import, GPU access, network, or label I/O here."""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
from datetime import datetime, timezone


CHECKPOINT_SHA = "966f3420f833532aae3fb1fd6326533b08d43d23b7b03eaa2f0694a30b64a239"
SERVER_HOST = "rwkv-260304"
GPU_UUID = "GPU-5d61943c-0955-e221-92a8-318915f5a3a0"
MAX_GENERATION = 128
MAX_INPUT = 4096
STATE_KEYS = {f"blocks.{layer}.att.time_state" for layer in range(32)}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def checked_file(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Missing or invalid expected SHA256")
    if sha_file(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return Path(path)


def write_json(path, data):
    """Create once and fsync. Existing outputs are never resumed or overwritten."""
    path = Path(path)
    raw = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    sync_directory(path.parent)
    return {"path": path.name, "sha256": sha_bytes(raw), "bytes": len(raw)}


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def check_host_values(hostname, visible, policy):
    if hostname != SERVER_HOST or policy.get("verified_hostname") != SERVER_HOST:
        raise ValueError("Execution is restricted to the verified rwkv-8222 server")
    if visible != GPU_UUID or policy.get("gpu_uuid") != GPU_UUID:
        raise ValueError("CUDA_VISIBLE_DEVICES must bind only the verified GPU2 UUID")
    if policy.get("physical_gpu_index") != 2:
        raise ValueError("Physical GPU index must be 2")
    if policy.get("local_gpu_allowed") is not False or policy.get("other_server_gpus_allowed") is not False:
        raise ValueError("Policy must explicitly forbid local and other server GPUs")
    if policy.get("checkpoint_sha256") != CHECKPOINT_SHA:
        raise ValueError("Resource policy checkpoint mismatch")


def server_guard(config):
    """All checks run before any torch import; no local nvidia-smi is invoked."""
    policy = json.loads(checked_file(config["policy_path"], config["policy_sha256"]).read_text())
    check_host_values(socket.gethostname(), os.environ.get("CUDA_VISIBLE_DEVICES"), policy)
    result = subprocess.run([
        "nvidia-smi", "-i", "2", "--query-gpu=index,uuid,name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True)
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    if len(fields) != 5 or fields[0] != "2" or fields[1] != GPU_UUID:
        raise ValueError("Physical GPU2 no longer matches its recorded UUID")
    return {"hostname": socket.gethostname(), "physical_gpu": 2, "uuid": fields[1],
            "name": fields[2], "total_mib": int(fields[3]), "used_mib": int(fields[4]),
            "observed_at": now(), "is_reservation": False}


def load_parser(path, expected_sha):
    source = checked_file(path, expected_sha).read_text()
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == "parse_task_selections"]
    if len(matches) != 1:
        raise ValueError("Expected exactly one production selection parser")
    function = matches[0]
    if function.decorator_list:
        raise ValueError("Unexpected parser decorators")
    compiled = ast.Module(body=[function], type_ignores=[])
    namespace = {"re": re}
    exec(compile(compiled, str(path), "exec"), namespace)
    return namespace["parse_task_selections"], {
        "source_sha256": expected_sha,
        "function_ast_sha256": sha_bytes(ast.dump(function, include_attributes=False).encode()),
        "name": "parse_task_selections",
    }


def validate_state_metadata(metadata):
    if set(metadata) != STATE_KEYS:
        raise ValueError("State must contain exactly 32 canonical time_state tensors")
    for key, item in metadata.items():
        if item.get("shape") != [40, 64, 64] or item.get("dtype") != "torch.float32":
            raise ValueError(f"Wrong state shape or dtype: {key}")
        if item.get("finite") is not True:
            raise ValueError(f"Nonfinite state: {key}")


def output_record(token_ids, vocab, eos_observed, cap=MAX_GENERATION):
    """Never decode EOS as text or hide invalid UTF-8/cap termination."""
    if cap != MAX_GENERATION:
        raise ValueError("Generation cap is frozen at 128")
    if not token_ids or len(token_ids) > cap or any(type(token) is not int for token in token_ids):
        raise ValueError("Invalid generated token sequence")
    if eos_observed != (token_ids[-1] == 0) or 0 in token_ids[:-1]:
        raise ValueError("EOS receipt contradicts token sequence")
    if not eos_observed and len(token_ids) != cap:
        raise ValueError("Incomplete generation cannot become a completed record")
    output_ids = token_ids[:-1] if eos_observed else token_ids
    raw = b"".join(vocab[token] for token in output_ids)
    try:
        decoded, valid_utf8, utf8_error = raw.decode("utf-8", errors="strict"), True, None
    except UnicodeDecodeError as error:
        decoded, valid_utf8 = None, False
        utf8_error = {"start": error.start, "end": error.end, "reason": error.reason}
    return {
        "generated_token_ids_including_eos": token_ids,
        "output_token_ids": output_ids, "actual_eos_token": 0 if eos_observed else None,
        "eos_observed": eos_observed, "generated_steps": len(token_ids),
        "generation_cap": cap, "cap_reached_without_eos": not eos_observed,
        "raw_bytes_base64": base64.b64encode(raw).decode("ascii"),
        "raw_bytes_sha256": sha_bytes(raw), "raw_bytes_count": len(raw),
        "raw_text": decoded, "valid_utf8": valid_utf8, "utf8_error": utf8_error,
        "termination": "actual_eos_0" if eos_observed else "128_token_cap",
    }


def score_output(record, expected_ids, unit_count, parser):
    expected = {int(value.removeprefix("E")) for value in expected_ids}
    if any(value < 1 or value > unit_count for value in expected):
        raise ValueError("Gold contains an unknown unit")
    selected, error = [], None
    if not record["eos_observed"]:
        error = "no_actual_eos"
    elif not record["valid_utf8"]:
        error = "invalid_utf8"
    else:
        try:
            selected = parser(record["raw_text"], unit_count)
        except ValueError as exc:
            error = str(exc)
    valid = error is None
    chosen = set(selected) if valid else set()
    positive = bool(expected)
    return {
        "protocol_valid_with_eos": valid, "protocol_error": error,
        "selected_unit_ids": [f"E{value}" for value in selected] if valid else [],
        "expected_unit_ids": [f"E{value}" for value in sorted(expected)],
        "strict_source_correct": valid and chosen == expected,
        "positive": positive,
        "positive_any_support": positive and valid and bool(chosen & expected),
        "positive_all_support": positive and valid and expected <= chosen,
        "positive_exact_selection": positive and valid and chosen == expected,
        "negative_strict_rejection": not positive and valid and not chosen,
        "negative_valid_misselection": not positive and valid and bool(chosen),
        "negative_invalid": not positive and not valid,
        "unit_tp": len(chosen & expected), "unit_fp": len(chosen - expected),
        "unit_fn": len(expected - chosen),
    }


def aggregate(scores):
    keys = ["protocol_valid_with_eos", "strict_source_correct", "positive", "positive_any_support",
            "positive_all_support", "positive_exact_selection", "negative_strict_rejection",
            "negative_valid_misselection", "negative_invalid", "unit_tp", "unit_fp", "unit_fn"]
    result = {key: sum(int(item[key]) for item in scores) for key in keys}
    result["count"] = len(scores)
    result["negative"] = len(scores) - result["positive"]
    result["positive_any_support_recall"] = result["positive_any_support"] / result["positive"] if result["positive"] else None
    result["positive_exact_selection_recall"] = result["positive_exact_selection"] / result["positive"] if result["positive"] else None
    return result


def choose_candidate(metrics_by_step):
    if set(metrics_by_step) != {0, 8, 16}:
        raise ValueError("Selection requires exactly steps 0, 8, 16")
    return min(metrics_by_step, key=lambda step: (
        -metrics_by_step[step]["strict_source_correct"],
        metrics_by_step[step]["negative_valid_misselection"], step,
    ))


def validate_selection(receipt, held_inputs_sha, held_gold_sha, states):
    if receipt.get("kind") != "frozen_dev_candidate_selection_v1":
        raise ValueError("Heldout needs a completed dev selection receipt")
    if receipt.get("heldout_inputs_sha256") != held_inputs_sha or receipt.get("heldout_gold_sha256") != held_gold_sha:
        raise ValueError("Heldout hashes were not fixed before selection")
    if receipt.get("generation_cap") != MAX_GENERATION or receipt.get("selection_uses_heldout") is not False:
        raise ValueError("Selection receipt has an invalid protocol")
    picked = receipt.get("selected_step")
    metrics = {int(key): value for key, value in receipt["dev_metrics_by_step"].items()}
    if picked != choose_candidate(metrics):
        raise ValueError("Selected step does not follow the frozen selection rule")
    expected_steps = {0} if picked == 0 else {0, picked}
    if {state["step"] for state in states} != expected_steps or len(states) != len(expected_steps):
        raise ValueError("Heldout states must be exactly zero and the selected candidate")
    recorded = {state["step"]: state for state in receipt["states"]}
    for state in states:
        if state["step"] not in recorded or state["sha256"] != recorded[state["step"]]["sha256"]:
            raise ValueError("Heldout state differs from the selected dev state")


def validate_input(item, tokenizer, expected_split):
    if item.get("split") != expected_split or item.get("protocol") != "task-evidence-v1":
        raise ValueError("Wrong dataset split or protocol")
    prompt = item["prompt"]
    if sha_bytes(prompt.encode()) != item["prompt_sha256"]:
        raise ValueError("Prompt SHA mismatch")
    if item.get("assistant_prefill") != "<think></think>" or not prompt.endswith("<think></think>"):
        raise ValueError("The complete closed assistant prefill is required")
    if item.get("field_coverage_assessed") is not False or item.get("retrieval_performed") is not False:
        raise ValueError("This is a Reader-only pilot, not a retrieval or field-coverage evaluation")
    ids = tokenizer.encode(prompt)
    if len(ids) != item["input_token_count"] or not (0 < len(ids) <= MAX_INPUT):
        raise ValueError("Input token count mismatch or 4096-token budget exceeded")
    if tokenizer.decodeBytes(ids) != prompt.encode():
        raise ValueError("Input token roundtrip failed")
    if any(type(token) is not int or not 1 <= token < 65536 for token in ids):
        raise ValueError("Invalid input token IDs")
    if [unit["id"] for unit in item["units"]] != [f"E{i}" for i in range(1, len(item["units"]) + 1)]:
        raise ValueError("Evidence units are not consecutive")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", item["id"]):
        raise ValueError("Unsafe sample ID")
    return ids
