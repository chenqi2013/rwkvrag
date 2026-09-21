import asyncio
import json
import pytest
from llamaindex_retrieval.funnel_writer import Facts, Cell, checked_facts, checked_cell
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline
from llamaindex_retrieval.native_rwkv import NativeRWKVResult
from llamaindex_retrieval.schemas import SourceItem, ConversationMessage


def source():
    return SourceItem(id='s1',document_id='d1',source='test',title='甲',snippet='甲容量8升，不支持离线。',score=1,metadata={})


def test_facts_require_literal_quotes_and_known_dimensions():
    base={'object_id':'O1','field_id':'F1','value':'8升','quote':'甲容量8升'}
    assert checked_facts(Facts(facts=[base]),{'O1':'甲'},{'F1':'容量'},source())[0]['start']==0
    with pytest.raises(ValueError):
        checked_facts(Facts(facts=[{**base,'quote':'甲容量9升'}]),{'O1':'甲'},{'F1':'容量'},source())
    with pytest.raises(ValueError):
        checked_facts(Facts(facts=[{**base,'object_id':'O2'}]),{'O1':'甲'},{'F1':'容量'},source())


def test_cells_cannot_borrow_foreign_facts_or_turn_no_evidence_into_supported():
    with pytest.raises(ValueError):checked_cell(Cell(status='supported',value='9升',fact_ids=['A2']),[{'id':'A1'}])
    with pytest.raises(ValueError):checked_cell(Cell(status='supported',value='9升',fact_ids=[]),[])
    assert checked_cell(Cell(status='unknown',value='没有依据',fact_ids=[]),[])['status']=='unknown'


class Model:
    def __init__(self):self.calls=[]
    async def complete(self,messages,**kwargs):
        prompt=messages[0]['content'];self.calls.append((prompt,kwargs))
        if prompt.startswith('整理用户'):
            value={'objects':['甲','乙'],'fields':['容量'],'requirements':[]}
        elif prompt.startswith('从这一份'):
            value={'facts':[{'object_id':'O1','field_id':'F1','value':'8升','quote':'甲容量8升'}]}
        elif prompt.startswith('只核验'):
            value={'status':'supported','value':'8升','fact_ids':['A1']} if '"object": "甲"' in prompt else {'status':'unknown','value':'乙容量未知','fact_ids':[]}
        elif prompt.startswith('按同一维度'):
            value={'status':'partial','summary':'甲8升，乙缺证据，不能比较大小','cell_ids':['O1:F1','O2:F1']}
        else:
            assert kwargs['stage']=='writer'
            assert '错误旧答案123' not in prompt
            assert '模型小结不是来源' in prompt
            value='甲8升[资料 1]；乙资料不足。'
        raw=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False)
        return NativeRWKVResult('completed',raw,'stop',{'stage':kwargs['stage'],'raw_text':raw,'status':'completed','messages':messages,'completion_attempted':True,'prompt_protocol':'g1j_plain','envelope':{'valid':True,'answer_span':{'start':0,'end':len(raw),'unit':'unicode_code_points'}}})


def test_full_funnel_dependencies_and_raw_answer_survive_pipeline():
    model=Model();pipeline=RWKVPipeline(Settings(native_writer_pipeline='funnel_v1'),None,model=model)
    response=asyncio.run(pipeline.ask_materials('比较甲乙容量',[source()],history=[ConversationMessage(role='assistant',content='错误旧答案123')]))
    assert response.answer=='甲8升[资料 1]；乙资料不足。'
    assert response.generation['raw_model_answer']==response.answer
    assert len(model.calls)==6
    assert len(response.generation['model_calls'])==6
    assert response.generation['citation_map']=={'1':'s1'}
    graph=response.retrieval['funnel']
    assert len(graph['cells'])==2 and not graph['failures']
    assert graph['field_summaries'][0]['status']=='partial'
    assert graph['facts'][0]['quote']==source().snippet[:5]
    json.dumps(response.model_dump())
