import base64,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data/quality-runs/writer-convergence-20260921"
RUN=DATA/"run2"

def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    summary=json.loads((RUN/"SUMMARY.json").read_text());assert summary["recorded"]==752 and summary["error"] is None
    inputs=json.loads((ROOT/"llamaindex-retrieval/eval/writer-convergence-20260921/INPUTS.json").read_text())
    cases=json.loads((ROOT/"llamaindex-retrieval/web/public/experiments/model-size-paired-20260921.json").read_text())["cases"]
    metrics={};records={};reviews={}
    for arm in ["baseline","decision"]:
        for rnd in [1,2]:
            key=f"{arm}-round{rnd}";rows=[];review=json.loads((DATA/"review1"/(key+".json")).read_text());assert len(review)==188
            for n in range(188):
                row=json.loads((RUN/key/f"{n:04d}.json").read_text());r=review[n]
                assert row["status"]=="recorded" and row["ordinal"]==n
                assert row["request"]["prompt"]==inputs[n][arm]
                assert sha(json.dumps(row["request"],ensure_ascii=False,separators=(",",":")).encode())==row["request_sha256"]
                body=base64.b64decode(row["response_body_base64"]);assert sha(body)==row["response_sha256"]
                answer=json.loads(body)["choices"][0];assert answer["text"]==row["raw_text"]
                assert sha(row["raw_text"].encode())==row["raw_text_sha256"]==r["raw_answer_sha256"]
                assert row["input_token_ids"]==answer["prompt_token_ids"] and row["output_token_ids"]==answer["token_ids"]
                rows.append(row)
            records[key]=rows;reviews[key]=review
            metrics[key]={"records":188,"length":sum(r["finish_reason"]=="length" for r in rows),**{k:sum(bool(r.get(k)) for r in review) for k in ["facts_complete","strict_pass","repetition_confirmed","runaway_copying","hard_condition_wrong_recommendation","empty_evidence_fabrication"]}}
    changes={}
    for arm in ["baseline","decision"]:
        for a,b in zip(records[arm+"-round1"],records[arm+"-round2"]):
            assert a["request"]==b["request"] and a["input_token_ids"]==b["input_token_ids"]
        changes[arm]=[a["ordinal"] for a,b in zip(records[arm+"-round1"],records[arm+"-round2"]) if (a["raw_text"],a["output_token_ids"],a["finish_reason"])!=(b["raw_text"],b["output_token_ids"],b["finish_reason"])]
    out={"metrics":metrics,"cross_round_changed_ordinals":changes,"all_752_wire_and_review_bindings_verified":True,"promotion":"failed","reason":"candidate repetition and ordinary regressions; hard-condition misrecommendations increased","reviewer":"task implementer, not independent"}
    (DATA/"review1/SUMMARY.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print(json.dumps(out,ensure_ascii=False))

if __name__=="__main__":main()
