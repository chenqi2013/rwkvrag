import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('chunk_audit',Path(__file__).with_name('audit_chunked.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_decisions_require_complete_unique_targets():
    row={'index':0,'accept':True,'issues':[],'reason':'supported'}
    assert m.decisions({'reviews':[row]},1)[0]['accept']
    for rows,size in [([row],2),([row,row],2),([{**row,'accept':True,'issues':['wrong object']}],1),([{**row,'index':True}],1)]:
        with pytest.raises(ValueError):m.decisions({'reviews':rows},size)
