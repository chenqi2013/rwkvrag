import asyncio
import json
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline, conversation
from llamaindex_retrieval.native_rwkv import NativeRWKVResult
from llamaindex_retrieval.schemas import SourceItem
from llamaindex_retrieval.funnel_writer_v2 import write_funnel


class Model:
    def __init__(self): self.calls=[]
    async def complete(self,messages,**kwargs):
        p=messages[0]['content'];self.calls.append(p)
        if p.startswith('整理用户'):
            value={'objects':['甲','乙'],'fields':['容量'],'requirements':[]}
        elif p.startswith('判断这份'):
            value={'object_ids':['O1']}
        elif p.startswith('只从这一份'):
            data=json.loads(p.split('\n',1)[1]);assert data['object']=='甲' and data['field']=='容量'
            assert 'objects' not in data and 'fields' not in data
            value={'value':'8升','quote':'甲容量8升'}
        elif p.startswith('只核验'):
            value={'status':'supported','value':'8升','fact_ids':['A1']} if '"object": "甲"' in p else {'status':'unknown','value':'未知','fact_ids':[]}
        elif p.startswith('按同一维度'):
            value={'status':'partial','summary':'甲8升乙未知','cell_ids':['O1:F1','O2:F1']}
        else:value='甲8升[资料 1]，乙未知。'
        raw=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False)
        return NativeRWKVResult('completed',raw,'stop',{'stage':kwargs['stage'],'raw_text':raw,'status':'completed','completion_attempted':True,'prompt_protocol':'g1j_plain','envelope':{'valid':True,'answer_span':{'start':0,'end':len(raw),'unit':'unicode_code_points'}}})


def test_atomic_calls_are_isolated_and_have_exact_provenance():
    model=Model();pipeline=RWKVPipeline(Settings(),None,model=model)
    source=SourceItem(id='s',document_id='d',source='test',title='甲',snippet='甲容量8升',score=1,metadata={})
    result=asyncio.run(write_funnel(pipeline,conversation('比较甲乙容量',[]),[source]))
    graph=result.trace['funnel']
    assert not graph['failures']
    assert len(graph['facts'])==1 and graph['facts'][0]['quote']==source.snippet
    assert graph['cells'][1]['status']=='unknown'
    assert result.raw_text=='甲8升[资料 1]，乙未知。'
    assert len(model.calls)==7
