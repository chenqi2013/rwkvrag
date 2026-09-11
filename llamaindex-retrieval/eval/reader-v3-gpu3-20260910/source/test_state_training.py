from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from llamaindex_retrieval.state_training import (
    read_training_tokens, accumulation_groups, group_mean, longest_training_row,
)


def sample(identity="a", prompt_length=2):
    return {"id": identity, "input_ids": [1] * prompt_length + [2, 0],
            "labels": [-100] * prompt_length + [2, 0], "prompt_tokens": prompt_length,
            "target_tokens_with_eos": 2, "prompt_sha256": "a" * 64}


def write_export(path, rows):
    raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_21_rows_visit_every_sample_once_and_final_group_uses_own_size(tmp_path, monkeypatch):
    rows = [sample(str(i)) for i in range(21)]
    path = tmp_path / "train.jsonl"
    digest = write_export(path, rows)
    original = Path.read_bytes
    def only_train(p):
        assert p == path, "adapter attempted to open a different file"
        return original(p)
    monkeypatch.setattr(Path, "read_bytes", only_train)
    loaded = read_training_tokens(path, digest, 21)
    groups = accumulation_groups(loaded, seed=7)
    assert [len(g) for g in groups] == [2] * 10 + [1]
    assert sorted(r["id"] for g in groups for r in g) == sorted(r["id"] for r in rows)
    assert groups == accumulation_groups(loaded[::-1], seed=7)
    assert group_mean([6.0, 2.0]) == 4.0
    assert group_mean([6.0]) == 6.0


@pytest.mark.parametrize("mutation", ["mask", "bool", "eos", "internal_eos", "length", "boundary", "target_length", "duplicate"])
def test_invalid_training_export_rejected_before_model_import(tmp_path, mutation):
    row = sample()
    if mutation == "mask": row["labels"][0] = 1
    if mutation == "bool": row["input_ids"][0] = True
    if mutation == "eos": row["input_ids"][-1] = 3
    if mutation == "internal_eos": row["input_ids"][0] = 0
    if mutation == "length": row["labels"].pop()
    if mutation == "boundary": row["prompt_tokens"] = len(row["input_ids"]) - 1
    if mutation == "target_length": row["target_tokens_with_eos"] = 3
    rows = [row, deepcopy(row)] if mutation == "duplicate" else [row]
    path = tmp_path / "train.jsonl"
    digest = write_export(path, rows)
    with pytest.raises(ValueError):
        read_training_tokens(path, digest, len(rows))


def test_file_binding_count_length_and_longest_tie_break(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [sample("z", 4), sample("a", 4), sample("b", 2)]
    digest = write_export(path, rows)
    for pin, count, budget in [("0" * 64, 3, 4096), (digest, 2, 4096), (digest, 3, 5)]:
        with pytest.raises(ValueError):
            read_training_tokens(path, pin, count, max_tokens=budget)
    assert longest_training_row(rows)["id"] == "a"


def test_preflight_rejects_local_host_before_nvidia_smi(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "statetune/preflight_state.py"
    spec = importlib.util.spec_from_file_location("preflight_state_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.socket, "gethostname", lambda: "local-wsl")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", module.GPU)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("queried local GPU"))
    with pytest.raises(ValueError, match="only rwkv-8222"):
        module.host_guard()


@pytest.mark.parametrize("scenario", ["wrong_mask", "wrong_index", "busy"])
def test_gpu3_guard_rejects_wrong_device_and_occupied_card(monkeypatch, scenario):
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[1] / "statetune/preflight_state.py"
    spec = importlib.util.spec_from_file_location("gpu3_guard_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.socket,"gethostname",lambda:"rwkv-260304")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-5d61943c-0955-e221-92a8-318915f5a3a0" if scenario=="wrong_mask" else module.GPU)
    def query(args,**kwargs):
        if scenario == "wrong_mask": pytest.fail("must reject wrong mask before query")
        if "--query-gpu=index,uuid,memory.used" in args:
            return SimpleNamespace(stdout=f"{2 if scenario=='wrong_index' else 3}, {module.GPU}, 15")
        return SimpleNamespace(stdout=f"{module.GPU}, 1234")
    monkeypatch.setattr(module.subprocess,"run",query)
    with pytest.raises(ValueError): module.host_guard()
