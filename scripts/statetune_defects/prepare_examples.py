"""Join explicit implementer annotations to immutable inputs. No automatic semantic repair."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llamaindex-retrieval/src"))
from llamaindex_retrieval.typed_funnel_contract_v7 import Atomic, validate_atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    annotations = ROOT / "llamaindex-retrieval/statetune/defect-led-20260922/ANNOTATIONS.json"
    records = []
    for index, a in enumerate(json.loads(annotations.read_text())):
        if "writer_record" in a:
            source = ROOT / "data/quality-runs/writer-evidence-handoff-20260921/run1" / a["writer_record"]
            trace = json.loads(source.read_text())["trace"]
            target = a["target"]
        else:
            source = ROOT / "data/quality-runs/typed-funnel-v10-integration-20260921/run1" / f"{a['case']:03d}.json"
            calls = json.loads(source.read_text())["response"]["generation"]["model_calls"]
            trace = next(c for c in calls if c["call_id"] == a["call_id"])
            inp = json.loads(trace["messages"][0]["content"].split("\n")[-1])
            units = {k: SimpleNamespace(text=v) for k, v in inp["evidence"].items()}
            validate_atomic(Atomic.model_validate(a["target"]), inp["field"], units)
            target = json.dumps(a["target"], ensure_ascii=False)
        assert hashlib.sha256(trace["prompt"].encode()).hexdigest() == trace["prompt_sha256"]
        records.append({"id": f"diagnostic-{index:03d}", "defect": a["defect"],
            "source_record": str(source.relative_to(ROOT)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "call_id": trace["call_id"], "prompt": trace["prompt"], "prompt_sha256": trace["prompt_sha256"],
            "observed_output": trace["raw_text"], "target": target, "rationale": a["rationale"],
            "partition": "diagnostic_only", "review_status": "implementer_reviewed", "independent_review": False,
            "training_eligible": False, "unseen_evaluation_eligible": False,
            "annotation_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest()})
    with args.out.open("x") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print({"diagnostic_examples": len(records), "training_examples_exported": 0, "independently_reviewed": False})


if __name__ == "__main__":
    main()
