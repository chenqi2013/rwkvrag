import asyncio
import json
import httpx
from llamaindex_retrieval.structured_native import StructuredNativeRWKVClient


def test_concurrent_schemas_are_request_scoped_and_wire_matches_trace():
    seen=[]
    async def handle(request):
        data=json.loads(request.content)
        if request.url.path=='/tokenize':return httpx.Response(200,json={'count':2,'tokens':[0,2],'max_model_len':16384})
        await asyncio.sleep(.002)
        seen.append(data)
        return httpx.Response(200,json={'choices':[{'text':'<think></think>\n{}','finish_reason':'stop'}],'usage':{'prompt_tokens':2,'completion_tokens':4}})
    async def run():
        async with StructuredNativeRWKVClient(base_url='http://test/v1',model='test',transport=httpx.MockTransport(handle)) as client:
            return await asyncio.gather(*(client.complete([{'role':'user','content':str(i)}],structured_schema=schema) for i,schema in enumerate([{'type':'object'},{'type':'array'},None])))
    results=asyncio.run(run())
    assert len(seen)==3
    for result in results:
        payload=json.loads(__import__('base64').b64decode(result.trace['http'][-1]['request_body_base64']))
        assert payload.get('structured_outputs')==result.trace['parameters'].get('structured_outputs')
    assert 'structured_outputs' not in seen[-1]
    assert seen[0]['structured_outputs']!=seen[1]['structured_outputs']
