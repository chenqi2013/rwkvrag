import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('progression_train',Path(__file__).with_name('train.py'))
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)

@pytest.mark.parametrize('value',[1e-6,1e-5,1e-4])
def test_lr_range(value):t.check_lr(value)

@pytest.mark.parametrize('value',[True,0.,1e-7,1e-3,float('nan'),float('inf'),'1e-5'])
def test_lr_rejects_outside_and_nonfinite(value):
    with pytest.raises(ValueError):t.check_lr(value)

def test_only_complete_train_rows_with_eos_and_target_mask(tmp_path):
    p=tmp_path/'data.jsonl'
    row={'id':'a','split':'train','state_role':'writer','input_ids':[3,4,5,0],
         'labels':[-100,-100,5,0],'prompt_tokens':2}
    p.write_text(json.dumps(row)+'\n');assert len(t.load_rows(p,4))==1
    for field,value in [('split','holdout'),('labels',[3,4,5,0]),('input_ids',[3,4,5,6])]:
        p.write_text(json.dumps(dict(row,**{field:value}))+'\n')
        with pytest.raises(ValueError):t.load_rows(p,4)
    p.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError):t.load_rows(p,3)
