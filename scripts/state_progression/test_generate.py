import copy
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('progression_generator',Path(__file__).with_name('generate.py'))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def batch():
    items=[]
    for i in range(6):
        items.append({'id':str(i),'stage':['normal','normal','defect','progress','defect','progress'][i],
            'parent_id':str(i-1) if i in (3,5) else None,'available_sources':['S1'],
            'kind':'writer','question':'甲重多少？','history':[],
            'evidence':[{'source_id':'S1','quote':'甲重3kg。'}],
            'answer':'甲重3kg。[资料 1]','rationale':'原文','skill':'绑定对象'})
    return {'items':items},{'S1':{'id':'S1','title':'记录','text':'甲重3kg。'},
                            'S2':{'id':'S2','title':'补充','text':'乙重4kg。'}}

def test_future_source_is_not_visible():
    draft,sources=batch()
    draft['items'][3]['evidence']=[{'source_id':'S2','quote':'乙重4kg。'}]
    compiled,checks=module.compile_batch(draft,sources)
    assert 3 not in compiled and not checks[3]['pass']
    assert '乙重4kg' not in compiled[0]['prompt']

def test_parent_must_precede_and_be_accepted():
    draft,sources=batch()
    compiled,_=module.compile_batch(draft,sources)
    decisions={i:{'accept':i!=2} for i in range(6)}
    accepted=module.accepted_with_parents(draft['items'],compiled,decisions)
    assert {x['item_id'] for x in accepted}=={'0','1','4','5'}
    draft['items'][3]['parent_id']='5'
    compiled,_=module.compile_batch(draft,sources)
    assert 3 not in compiled

def test_no_single_stage_batch_or_duplicate_ids():
    draft,sources=batch()
    for item in draft['items']:item['stage']='normal'
    with pytest.raises(ValueError,match='Missing'):module.compile_batch(draft,sources)
    draft,sources=batch();draft['items'][1]['id']='0'
    with pytest.raises(ValueError,match='duplicate'):module.compile_batch(draft,sources)

def test_real_sources_are_exact_input():
    _,sources=batch()
    job={'material_mode':'real','sources':list(sources.values())}
    assert module.material_map(job,{'sources':[]})==sources
