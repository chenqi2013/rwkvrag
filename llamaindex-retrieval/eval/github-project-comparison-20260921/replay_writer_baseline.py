import asyncio, hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"llamaindex-retrieval/src"))
from llamaindex_retrieval.native_rwkv import NativeRWKVClient
from llamaindex_retrieval.writer_decision_prompt import DECISION_INSTRUCTIONS

async def main():
    here=Path(__file__).resolve().parent
    for name,digest in json.loads((here/"PINS.json").read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for suite in ["github-project-comparison-20260921","live-comparison-20260921"]:
        directory=ROOT/"data/quality-runs"/suite
        out=directory/"baseline-writer-replay1";out.mkdir(exist_ok=False)
        cases=json.loads((ROOT/"llamaindex-retrieval/eval"/suite/"CASES.json").read_text())
        async with NativeRWKVClient(base_url="http://127.0.0.1:18426/v1",model="paired-eval",prompt_protocol="g1j_plain",timeout_seconds=180,max_concurrency=1) as client:
            for n,c in enumerate(cases):
                original=directory/"run1"/f"{n:02d}.json";record=json.loads(original.read_text())
                event=next((x for x in reversed(record.get("response",{}).get("generation",{}).get("model_calls",[])) if x.get("stage")=="writer"),None)
                row={"ordinal":n,"id":c["id"],"source_record_sha256":hashlib.sha256(original.read_bytes()).hexdigest()}
                tick=time.monotonic()
                if event is None:row.update(status="not_attempted",reason="upstream produced no Writer input")
                else:
                    messages=event["messages"]
                    assert len(messages)==1 and messages[0]["content"].endswith(DECISION_INSTRUCTIONS)
                    prompt=messages[0]["content"][:-len(DECISION_INSTRUCTIONS)]
                    result=await client.generate([{"role":"user","content":prompt}],max_tokens=2048,assistant_prefill="<think></think",temperature=1.0,top_p=1.0,top_k=1,seed=11,stage="writer",evidence_ids=event["evidence_ids"])
                    assert result.trace["parameters"]==event["parameters"]
                    row.update(status=result.status,raw_text=result.raw_text,trace=result.trace,source_event_sha256=hashlib.sha256(json.dumps(event,sort_keys=True,ensure_ascii=False).encode()).hexdigest())
                row["elapsed_s"]=time.monotonic()-tick
                with (out/f"{n:02d}.json").open("x") as f:json.dump(row,f,ensure_ascii=False,indent=2)
                print(suite,n,row["status"],flush=True)

if __name__=="__main__":asyncio.run(main())
