"""Released evaluation coverage, source closure, and gold isolation checks."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from test_released_training import bundle, runner as trainer
from test_state_release import prepared

SCRIPTS = Path(__file__).resolve().parents[1] / "statetune"
spec = importlib.util.spec_from_file_location("evaluate_released_state", SCRIPTS / "evaluate_released_state.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


@pytest.fixture
def evaluation(bundle, prepared, tmp_path, monkeypatch):
    directory, corpus, review, vocab = prepared
    runtime = tmp_path / "official"
    (runtime / "rwkv").mkdir(parents=True)
    shutil.copyfile(vocab, runtime / "rwkv" / vocab.name)
    (runtime / "rwkv/model.py").write_text("# synthetic model closure\n")
    manifest = runtime / "RUNTIME-SOURCE.json"
    manifest.write_text(json.dumps({"installed_version": "0.8.30", "compile_cuda_extension": False,
        "files": {f.name: {"sha256": evaluator.sha(f)} for f in (runtime / "rwkv").iterdir()}}))
    pins = {"official/RUNTIME-SOURCE.json": evaluator.sha(manifest)}
    for name in evaluator.REQUIRED_CODE:
        relative = "llamaindex-retrieval/" + name
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists(): path.write_text("# synthetic frozen source\n")
        pins[relative] = evaluator.sha(path)
    zero = tmp_path / "zero.pth"
    zero.write_bytes(b"synthetic state; CPU validation checks hash only")
    monkeypatch.setattr(evaluator, "bound", trainer.bound)
    monkeypatch.setattr(evaluator, "OFFICIAL_MANIFEST_SHA", evaluator.sha(manifest))
    monkeypatch.setattr(evaluator, "ZERO_SHA", evaluator.sha(zero))
    receipt = json.loads((tmp_path / bundle["release"]).read_text())
    return {"schema": "rwkv_released_evaluation_v1", "wall_seconds": 3600,
            "deadline_utc": "2099-01-01T00:00:00+00:00", "official_runtime": "official",
            "bindings": pins, "release": bundle["release"], "release_sha256": bundle["release_sha256"],
            "checkpoint": "not-loaded-on-cpu.pth", "evaluation": {
                "split": "dev", "count": 2, "max_output_tokens": 32,
                "inputs": "dataset/dev.inputs.jsonl",
                "inputs_sha256": receipt["dataset_files"]["dev.inputs.jsonl"],
                "gold_sha256": receipt["dataset_files"]["dev.gold.jsonl"],
                "states": [{"name": "zero", "path": "zero.pth", "sha256": evaluator.sha(zero)}]}}


@pytest.mark.parametrize("prepared", ["task_last"], indirect=True)
def test_validate_split_without_any_gold_or_gpu_access(evaluation, tmp_path, monkeypatch):
    for path in (tmp_path / "dataset").glob("*.gold.jsonl"): path.unlink()
    monkeypatch.setattr(evaluator, "host_guard", lambda: pytest.fail("CPU check used GPU guard"))
    rows = evaluator.validate(evaluation)
    assert len(rows) == 2 and all(row["split"] == "dev" for row in rows)


@pytest.mark.parametrize("prepared", ["task_last"], indirect=True)
@pytest.mark.parametrize("change", ["count", "duplicate_state", "missing_zero", "inputs", "gold_hash",
                                   "partial_manifest", "missing_binding", "state", "heldout_count"])
def test_invalid_released_evaluation_rejected(evaluation, tmp_path, change):
    spec = evaluation["evaluation"]
    if change == "count": spec["count"] = 3
    elif change == "duplicate_state": spec["states"] *= 2
    elif change == "missing_zero": spec["states"][0]["name"] = "candidate"
    elif change == "inputs": (tmp_path / spec["inputs"]).write_text("{}\n")
    elif change == "gold_hash": spec["gold_sha256"] = "wrong"
    elif change == "partial_manifest":
        path = tmp_path / "official/RUNTIME-SOURCE.json"
        value = json.loads(path.read_text());value["files"].pop("model.py")
        path.write_text(json.dumps(value))
        evaluation["bindings"]["official/RUNTIME-SOURCE.json"] = evaluator.sha(path)
    elif change == "missing_binding": evaluation["bindings"].pop(next(iter(evaluation["bindings"])))
    elif change == "state": (tmp_path / "zero.pth").write_bytes(b"changed state")
    else: spec["split"] = "heldout"
    with pytest.raises(ValueError): evaluator.validate(evaluation)


@pytest.mark.parametrize("prepared", ["original"], indirect=True)
def test_evaluation_rejects_original_layout_release(evaluation):
    with pytest.raises(ValueError, match="unadmitted"):
        evaluator.validate(evaluation)
