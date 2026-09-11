import hashlib
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "statetune"
sys.path.insert(0, str(SCRIPTS))
import score_final_protocol as final


def write(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "bound", lambda p: Path(p))
    inputs, gold = tmp_path / "inputs.jsonl", tmp_path / "gold.jsonl"
    rows = [{"id":str(i),"split":"heldout","units":["unit"],"prompt_sha256":str(i)} for i in range(11)]
    inputs.write_text("\n".join(map(json.dumps,rows)))
    gold.write_text("\n".join(json.dumps({"id":str(i),"expected_unit_ids":["E1"] if i==0 else []}) for i in range(11)))
    states = []
    for name in ("zero", "epoch-11"):
        p = tmp_path / (name + ".pth");p.write_text(name)
        states.append({"name":name,"path":str(p),"sha256":final.sha(p)})
    configs = []
    for index in range(3):
        config = tmp_path / f"config-{index}.json"
        pin = write(config,{"bindings":{},"evaluation":{"split":"heldout","max_output_tokens":32,
            "inputs":str(inputs),"inputs_sha256":final.sha(inputs),"gold_sha256":final.sha(gold),"states":states}})
        run = tmp_path / f"run-{index}";run.mkdir();records=[]
        for state in states:
            for row in rows:
                text = "E1" if row["id"]=="0" and state["name"]=="epoch-11" else "NONE"
                p = run / f"record-{len(records)}.json"
                raw = {"state":state["name"],"id":row["id"],"state_sha256":state["sha256"],
                       "prompt_sha256":row["prompt_sha256"],"output":{"raw_text":text,
                       "utf8_valid":True,"eos_observed":True,"cap_reached":False}}
                records.append({"path":p.name,"sha256":write(p,raw)})
        write(run / "COMPLETED.json",{"status":"GENERATION_COMPLETE","gold_opened":False,
              "optimizer_updates":0,"split":"heldout","config_sha256":pin,"records":records})
        configs.append({"config":str(config),"sha256":pin,"run":run.name,"primary":index==2})
    protocol = tmp_path / "protocol.json"
    digest = write(protocol,{"schema":"rwkv_reader_final_heldout_v1","heldout_trial_count":1,
        "optimizer_updates":0,"total_generation_cap":66,"configs":configs,"score_after_all_raw_complete":True,
        "no_post_heldout_tuning":True,"candidate_state_sha256":states[1]["sha256"]})
    return protocol, digest, gold


@pytest.mark.parametrize("corruption", ["missing_run", "missing_case", "changed_raw", "missing_output", "wrong_output_type"])
def test_later_run_failure_prevents_any_gold_scoring(tmp_path, monkeypatch, corruption):
    protocol, digest, gold = bundle(tmp_path, monkeypatch)
    last = tmp_path / "run-2"
    if corruption == "missing_run": (last / "COMPLETED.json").unlink()
    elif corruption == "missing_case":
        receipt = json.loads((last / "COMPLETED.json").read_text());receipt["records"].pop()
        write(last / "COMPLETED.json", receipt)
    elif corruption == "changed_raw": (last / "record-0.json").write_text("changed")
    else:
        path = last / "record-0.json"
        record = json.loads(path.read_text())
        if corruption == "missing_output": del record["output"]
        else: record["output"]["eos_observed"] = "false"
        digest_raw = write(path, record)
        receipt = json.loads((last / "COMPLETED.json").read_text())
        receipt["records"][0]["sha256"] = digest_raw
        write(last / "COMPLETED.json", receipt)
    calls = []
    monkeypatch.setattr(final,"score",lambda *args:calls.append(args))
    with pytest.raises((ValueError,FileNotFoundError)):
        final.run(protocol,digest,tmp_path,gold,tmp_path / "result.json")
    assert calls == []
    assert not (tmp_path / "result.raw-ready.json").exists()


def test_all_raw_certificate_exists_before_first_score_and_candidate_is_fixed(tmp_path, monkeypatch):
    protocol,digest,gold = bundle(tmp_path, monkeypatch)
    actual_score = final.score
    calls = []
    def checked_score(*args):
        ready = json.loads((tmp_path / "result.raw-ready.json").read_text())
        assert ready["raw_records"] == 66 and not ready["gold_opened"]
        calls.append(args)
        return actual_score(*args)
    monkeypatch.setattr(final,"score",checked_score)
    result = final.run(protocol,digest,tmp_path,gold,tmp_path / "result.json")
    assert len(calls) == 3 and result["primary_gate_passed"]
    assert not result["candidate_reselected"] and not result["production_promoted"]


def test_complete_capped_output_is_scored_as_invalid_not_discarded(tmp_path, monkeypatch):
    protocol,digest,gold = bundle(tmp_path, monkeypatch)
    path = tmp_path / "run-2" / "record-11.json"
    record = json.loads(path.read_text())
    record["output"].update(eos_observed=False, cap_reached=True)
    raw_sha = write(path, record)
    receipt_path = path.parent / "COMPLETED.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["records"][11]["sha256"] = raw_sha
    write(receipt_path, receipt)
    result = final.run(protocol,digest,tmp_path,gold,tmp_path / "result.json")
    assert not result["primary_gate_passed"]
    assert result["runs"]["run-2"]["scores"][1]["invalid"] == 1
