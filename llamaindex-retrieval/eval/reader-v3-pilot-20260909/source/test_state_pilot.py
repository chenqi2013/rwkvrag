import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "statetune"
sys.path.insert(0, str(SCRIPTS))
from evaluate_state import generate
from score_state import score_case, select_state


def test_training_gate_accepts_metadata_certificate_and_rejects_wrong_preflight(tmp_path, monkeypatch):
    import copy
    import hashlib
    import json
    import train_state
    root = SCRIPTS.parents[1]
    config = json.loads((SCRIPTS.parent / "eval/reader-v3-pilot-20260909/PILOT.json").read_text())
    monkeypatch.setattr(train_state.importlib.metadata,"version",lambda n:config["packages"][n])
    monkeypatch.setattr(train_state,"bound",lambda p:root/p)
    train_state.check_config(config)
    for key,mutation in [
            ("semantic_review",lambda d:d.update(cases=[{"gold":"E1"}])),
            ("preflight_started",lambda d:d["config"].update(train_sha256="f"*64))]:
        changed = copy.deepcopy(config)
        source = root/config[key]["path"]
        value = json.loads(source.read_text())
        mutation(value)
        path = tmp_path/f"{key}.json"
        path.write_text(json.dumps(value))
        changed[key] = {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        with pytest.raises(ValueError):
            train_state.check_config(changed)


def test_full_prefix_eos_and_cap_are_not_repaired():
    prefixes = []
    tokens = iter([2, 0])
    def decide(prefix):
        prefixes.append(list(prefix))
        return next(tokens)
    raw = generate([1], {1:b"prompt", 2:b"E1"}, decide)
    assert prefixes == [[1], [1,2]]
    assert raw["eos_observed"] and raw["raw_text"] == "E1"
    capped = generate([1], {2:b"E1"}, lambda p: 2, cap=1)
    assert capped["cap_reached"] and not score_case(capped,["E1"],3)["correct"]
    with pytest.raises(ValueError, match="exceed context"):
        generate([1]*4096, {2:b"E1"}, lambda p: pytest.fail("model called"))


def test_scoring_uses_production_parser_and_rejects_malformed_bytes():
    output = {"eos_observed":True,"cap_reached":False,"utf8_valid":True,"raw_text":" E3,E2 "}
    assert score_case(output,["E2","E3"],3)["correct"]
    for text in ["E4", "E1 extra", "", "NONE E1"]:
        assert not score_case({**output,"raw_text":text},[],3)["valid"]
    assert not score_case({**output,"utf8_valid":False},["E2","E3"],3)["valid"]
    assert score_case({**output,"raw_text":"NONE"},[],3)["correct"]


def test_selection_tie_keeps_zero_and_syntax_gain_is_not_positive_gain():
    zero = {"step":0,"correct":4,"positive_correct":0,"negative_correct":4,"negative_false_positives":0}
    final = {**zero,"step":11}
    assert select_state([final,zero])["selected_step"] == 0
    assert not select_state([zero,{**final,"correct":5}])["heldout_eligible"]
    assert select_state([zero,{**final,"correct":5,"positive_correct":1}])["heldout_eligible"]
    assert not select_state([zero,{**final,"correct":5,"positive_correct":2,
                                  "negative_false_positives":1}])["heldout_eligible"]
    assert not select_state([zero,{**final,"correct":5,"positive_correct":5,
                                  "negative_correct":0}])["heldout_eligible"]


def test_training_accumulation_matches_reference_and_preserves_base(tmp_path):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from train_state import train_groups
    from state_training import accumulation_groups
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.state = torch.nn.Parameter(torch.tensor([0.05, -0.1, 0.2]))
            self.base = torch.nn.Parameter(torch.tensor([0.2, 0.3, 0.4]), requires_grad=False)
        def forward(self, ids):
            return ids[...,None].float()*self.state + self.base
    model, reference = Tiny(), Tiny()
    rows = [{"id":str(i), "input_ids":[i+1,1,0], "labels":[-100,1,0]} for i in range(3)]
    config = {"seed":7,"accumulation":2,"examples":3,"updates":2,"learning_rate":0.001,
              "betas":[0.9,0.999],"epsilon":1e-8,"weight_decay":0,"gradient_clip_l2":1.0}
    class CPU:
        cuda = SimpleNamespace(synchronize=lambda: None)
        def __getattr__(self, name): return getattr(torch,name)
        def tensor(self,*args,**kw):
            kw['device']='cpu'
            return torch.tensor(*args,**kw)
    records = train_groups(CPU(),model,[("state",model.state)],[("base",model.base)],rows,config,tmp_path)
    optimizer = torch.optim.AdamW([reference.state],lr=0.001,weight_decay=0)
    norms = []
    for group in accumulation_groups(rows,seed=7):
        optimizer.zero_grad(set_to_none=True)
        losses = [torch.nn.functional.cross_entropy(reference(torch.tensor([row["input_ids"][:-1]]))[0],
                        torch.tensor(row["labels"][1:])) for row in group]
        torch.stack(losses).mean().backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_([reference.state],1.0)))
        optimizer.step()
    assert [r["loss_divisor"] for r in records] == [2,1]
    assert [r["gradient_l2_before_clip"] for r in records] == pytest.approx(norms,rel=1e-6)
    assert torch.allclose(model.state,reference.state,atol=1e-7,rtol=1e-6)
    assert torch.equal(model.base,reference.base) and model.base.grad is None
