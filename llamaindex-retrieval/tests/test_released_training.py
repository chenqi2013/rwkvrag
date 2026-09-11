"""Release-to-trainer isolation and preflight identity regression checks."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import sys

import pytest

from llamaindex_retrieval import state_release
from test_state_release import prepared

SCRIPTS = Path(__file__).resolve().parents[1] / "statetune"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("train_released_state", SCRIPTS / "train_released_state.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def bundle(prepared, tmp_path, monkeypatch):
    directory, corpus, review, vocab = prepared
    release = tmp_path / "release"
    state_release.release(directory, release, corpus, review, vocab)
    source = tmp_path / "runtime"
    source.mkdir()
    shutil.copyfile(vocab, source / vocab.name)
    (source / "SOURCE-MANIFEST.json").write_text(json.dumps({"files": {
        vocab.name: {"sha256": runner.sha(source / vocab.name)}}}))
    pins = {"runtime/SOURCE-MANIFEST.json": runner.sha(source / "SOURCE-MANIFEST.json")}
    for name in runner.REQUIRED_CODE:
        relative = "llamaindex-retrieval/" + name
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic frozen source\n")
        pins[relative] = runner.sha(path)
    def bound(relative):
        path = (tmp_path / relative).resolve()
        if not path.is_relative_to(tmp_path):
            raise ValueError("path outside project")
        return path
    monkeypatch.setattr(runner, "PROJECT", tmp_path)
    monkeypatch.setattr(runner, "bound", bound)
    # Synthetic tests have a synthetic closure; production pins the full real manifest.
    monkeypatch.setattr(runner, "PEFT_MANIFEST_SHA", pins["runtime/SOURCE-MANIFEST.json"])
    return {"schema": "rwkv_released_training_v1", "mode": "preflight",
            "packages": runner.PACKAGES, "wall_seconds": 1800,
            "deadline_utc": "2026-09-10T00:00:00+00:00",
            "runtime": {"peft_source": "runtime", "checkpoint": "not-read-on-cpu.pth"},
            "bindings": pins, "release": "release/RELEASE.json",
            "release_sha256": runner.sha(release / "RELEASE.json"),
            "training": {"epochs": 2, "learning_rate": 1e-4, "accumulation": 2, "seed": 20260909}}


@pytest.mark.parametrize("prepared", ["task_last"], indirect=True)
def test_cpu_validation_opens_no_evaluation_content_or_model(bundle, tmp_path, monkeypatch):
    shutil.rmtree(tmp_path / "dataset")
    shutil.rmtree(tmp_path / "corpus")
    def forbidden(*args, **kwargs):
        pytest.fail("CPU validation touched GPU/model")
    monkeypatch.setattr(runner, "host_guard", forbidden)
    monkeypatch.setattr(runner, "load_model", forbidden)
    rows, identity = runner.validate(bundle)
    assert len(rows) == identity["train_count"] == 2
    assert identity["longest_tokens"] == max(len(r["input_ids"]) for r in rows)


@pytest.mark.parametrize("prepared", ["task_last"], indirect=True)
@pytest.mark.parametrize("change", ["tokens", "receipt", "code", "missing_binding", "epoch_cap", "lr", "partial_manifest"])
def test_changed_release_or_execution_contract_is_rejected(bundle, tmp_path, change):
    if change == "tokens":
        (tmp_path / "release/train.tokens.jsonl").write_text("{}\n")
    elif change == "receipt":
        (tmp_path / "release/RELEASE.json").write_text("{}\n")
    elif change == "code":
        (tmp_path / next(iter(bundle["bindings"]))).write_text("changed")
    elif change == "missing_binding":
        bundle["bindings"].pop(next(iter(bundle["bindings"])))
    elif change == "epoch_cap":
        bundle["training"]["epochs"] = 7
    elif change == "partial_manifest":
        path = tmp_path / "runtime/SOURCE-MANIFEST.json"
        path.write_text(json.dumps({"files": {}}))
        bundle["bindings"]["runtime/SOURCE-MANIFEST.json"] = runner.sha(path)
    else:
        bundle["training"]["learning_rate"] = 1e-3
    with pytest.raises(ValueError):
        runner.validate(bundle)


@pytest.mark.parametrize("prepared", ["original"], indirect=True)
def test_original_layout_release_cannot_enter_task_last_training(bundle):
    with pytest.raises(ValueError, match="protocol"):
        runner.validate(bundle)


@pytest.mark.parametrize("prepared", ["task_last"], indirect=True)
@pytest.mark.parametrize("change", [None, "data", "updated", "base", "runtime"])
def test_training_requires_preflight_for_exact_release_and_runtime(bundle, tmp_path, change):
    _, identity = runner.validate(bundle)
    receipt = {"status": "PREFLIGHT_COMPLETE", "identity": deepcopy(identity),
               "optimizer_updates": 0, "backward_calls": 1, "base_tensors_unchanged": 1062}
    if change == "data": receipt["identity"]["train_sha256"] = "different release"
    elif change == "updated": receipt["optimizer_updates"] = 1
    elif change == "base": receipt["base_tensors_unchanged"] = 1061
    elif change == "runtime": receipt["identity"]["runtime"]["checkpoint"] = "another model"
    path = tmp_path / "PREFLIGHT.json"
    path.write_text(json.dumps(receipt))
    bundle.update(mode="train", preflight={"path": path.name, "sha256": runner.sha(path)})
    if change is None:
        assert len(runner.validate(bundle)[0]) == 2
    else:
        with pytest.raises(ValueError, match="matching zero-update"):
            runner.validate(bundle)


def test_real_autograd_preflight_and_partial_accumulation(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    model = torch.nn.Module()
    model.register_parameter("state", torch.nn.Parameter(torch.zeros(1)))
    rows = [{"id": str(i), "input_ids": [1, 2, 0], "target": float(i + 1)} for i in range(3)]
    monkeypatch.setattr(runner, "loss_for", lambda torch, model, row: (model.state - row["target"]).square().mean())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(runner, "save_state", lambda torch, state, output, step: {"step": step})
    state = [("state", model.state)]
    preflight = runner.execute(torch, model, state, rows, {"mode": "preflight"}, tmp_path)
    assert preflight["optimizer_updates"] == 0 and preflight["gradient_l2"] > 0
    assert model.state.item() == 0 and model.state.grad is None
    result = runner.execute(torch, model, state, rows, {"mode": "train", "training": {
        "epochs": 2, "learning_rate": 1e-4, "accumulation": 2, "seed": 20260909}}, tmp_path)
    assert result["optimizer_updates"] == 4 and result["backward_calls"] == 6
    records = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("step-*-DONE.json"))]
    for epoch in (1, 2):
        batch = [r for r in records if r["epoch"] == epoch]
        assert sorted(i for r in batch for i in r["ids"]) == ["0", "1", "2"]
        assert [r["loss_divisor"] for r in batch] == [2, 1]
    assert model.state.item() > 0 and model.state.grad is None


@pytest.mark.parametrize("prepared", ["task_last"], indirect=True)
def test_sigterm_records_failure_before_model_load(bundle, tmp_path, monkeypatch):
    bundle["deadline_utc"] = "2099-01-01T00:00:00+00:00"
    config = tmp_path / "config.json"
    config.write_text(json.dumps(bundle))
    monkeypatch.setattr(sys, "argv", ["train_released_state", "--config", str(config),
                        "--config-sha256", runner.sha(config), "--output", "stopped"])
    monkeypatch.setattr(runner, "host_guard", lambda: {"synthetic": True})
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name: runner.PACKAGES[name])
    def terminate(*unused):
        os.kill(os.getpid(), signal.SIGTERM)
    monkeypatch.setattr(runner, "load_model", terminate)
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(InterruptedError, match="SIGTERM"):
        runner.main()
    receipt = json.loads((tmp_path / "stopped/FAILED.json").read_text())
    assert receipt["confirmed_updates"] == receipt["attempted_updates"] == 0
    assert "SIGTERM" in receipt["error"]
    assert signal.getsignal(signal.SIGTERM) == previous
