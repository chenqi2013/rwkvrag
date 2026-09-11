"""CPU-only actual-autograd check extracted from tests/test_state_pilot.py."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "statetune"))
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
