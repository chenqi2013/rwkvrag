"""Exercise actual source-local extraction, with no hand-selected evidence or retries."""
import argparse
import asyncio
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

from llamaindex_retrieval.atomic_evidence import AtomicEvidenceService, AtomicRequest
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.schemas import SourceItem


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--case-id", action="append")
    ap.add_argument("--fixtures", type=Path, default=Path(__file__).with_name("fixtures.jsonl"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).parent
    code = Path(__file__).parents[2] / "src/llamaindex_retrieval/atomic_evidence.py"
    (args.output / "BINDINGS.json").write_text(json.dumps({str(p): sha256(p.read_bytes()).hexdigest()
        for p in [args.fixtures, Path(__file__), code]}, indent=2))

    class Recorder:
        async def record_model_http(self, event, receipt):
            (args.output / (uuid4().hex + ".http.json")).write_text(json.dumps({"event": event, "receipt": receipt}, ensure_ascii=False, default=str))

    settings = Settings(_env_file=None, atomic_model_base_url="http://127.0.0.1:18425/v1",
        native_transport="rwkvos_batch", native_base_url="http://127.0.0.1:18423/v1",
        native_resolver_protocol="binary_query", native_task_source="queries",
        native_resolver_task_grouping="individual", rwkvos_reader_prompt_protocol="rwkv_g1j_no_think_v1",
        native_resolver_prefill="<think></think",
        rwkvos_binary_reader_state_id="reader-trace-450")
    svc = AtomicEvidenceService(settings, Recorder(), None)
    results = []
    try:
        for line in args.fixtures.read_text().splitlines():
            case = json.loads(line)
            if args.case_id and case["id"] not in args.case_id:
                continue
            sources = [SourceItem(id=f"s{i}", document_id=f"d{i}", source="synthetic",
                title=f"记录{i}", score=1, snippet=text, metadata={"knowledge_base_id": "eval", "source_sha256": sha256(text.encode()).hexdigest()})
                for i, text in enumerate(case["texts"], 1)]
            run = {"id": case["id"], "knowledge_base_id": "eval", "index_version": "fixed-material-v1",
                "claims": [], "sources": [], "calls": [], "issues": [], "coverage": {}}
            async with asyncio.timeout(180):
                await svc.extract(run, AtomicRequest(**case["request"]), sources)
            (args.output / (case["id"] + ".json")).write_text(json.dumps(run, ensure_ascii=False, indent=2))
            values = [c["statement_quote"] for c in run["claims"]]
            result = {"id": case["id"], "status": run["status"], "values": values,
                "kinds": [c["kind"] for c in run["claims"]], "issues": run["issues"],
                "expected_values": case["expected_values"], "manual_semantic_review_required": True}
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        await svc.aclose()
    (args.output / "SUMMARY.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
