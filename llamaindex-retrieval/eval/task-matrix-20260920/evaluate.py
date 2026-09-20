"""Frozen component evaluation. Writer semantics require explicit human review."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.model_client import model_client_class, model_client_options
from llamaindex_retrieval.reader_prompt import parse_binary_decision
from llamaindex_retrieval.rwkvos_batch import render_batch_prompt
from llamaindex_retrieval.rwkv_pipeline import structured_body
from llamaindex_retrieval.task_matrix import Assessment, Followup, Review, TaskPlan, parse_model


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "holdout"], default="validation")
    parser.add_argument("--stage", choices=["planner", "resolver", "writer"], required=True)
    parser.add_argument("--component", choices=["plan", "assessment", "review", "followup"], default="plan")
    parser.add_argument("--state-id")
    parser.add_argument("--base-url")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.settings.read_text())
    if args.base_url:
        config["native_base_url"] = args.base_url
    key = {"planner": "rwkvos_planner_state_id", "resolver": "rwkvos_binary_reader_state_id", "writer": "rwkvos_writer_state_id"}[args.stage]
    if args.state_id:
        config[key] = args.state_id
    settings = Settings(**config)
    model = model_client_class(settings)(**model_client_options(settings))
    path = args.data / f"{args.split}.{args.stage}.jsonl"
    manifest = json.loads((args.data / "MANIFEST.json").read_text())
    data_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if data_hash != manifest["files"][path.name]["sha256"]:
        raise ValueError("frozen evaluation data changed")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if args.stage == "planner":
        marker = {"plan": "根据完整对话", "assessment": "核对一个检索项目", "review": "检查答案", "followup": "仅为未解决项目"}[args.component]
        rows = [row for row in rows if row["prompt"].startswith("User: " + marker)]
    if args.stage == "writer":
        rows = [row for row in rows if "以下为逐项证据检查" in row["prompt"]]
    semaphore = asyncio.Semaphore(2)
    async def run(row):
        async with semaphore:
            suffix = "\n\nAssistant: <think></think>" + ("" if args.stage == "planner" else "\n")
            assert row["prompt"].startswith("User: ") and row["prompt"].endswith(suffix)
            messages = [{"role": "user", "content": row["prompt"][6:-len(suffix)]}]
            rendered, _ = render_batch_prompt(messages, "<think></think", "complete")
            assert rendered + ("" if args.stage == "planner" else "\n") == row["prompt"]
            result = await model.complete(messages, stage=args.stage, assistant_prefill="<think></think",
                max_tokens=32 if args.stage == "resolver" else 1024 if args.stage == "planner" else 2048)
            record = {"id": row["id"], "case_id": row["case_id"], "target": row["target"],
                      "status": result.status, "raw_text": result.raw_text, "trace": result.trace,
                      "passed": None}
            try:
                body = structured_body(result)
                record["answer"] = body
                if args.stage == "resolver":
                    predicted = parse_binary_decision(body)
                    expected = json.loads(row["target"])["answer"] == "YES"
                    record.update(passed=predicted == expected, expected=expected, predicted=predicted)
                elif args.stage == "planner":
                    if args.component == "plan":
                        actual = parse_model(body, TaskPlan)
                        expected = TaskPlan(**json.loads(row["target"]))
                        record["passed"] = (set(actual.objects) == set(expected.objects)
                            and set(actual.dimensions) == set(expected.dimensions)
                            and set(actual.conditions) == set(expected.conditions)
                            and actual.coverage == expected.coverage
                            and {(c.object, c.dimension) for c in actual.cells} == {(c.object, c.dimension) for c in expected.cells})
                        record["semantic_question_scope_manually_reviewed"] = False
                    elif args.component == "assessment":
                        actual = parse_model(body, Assessment)
                        expected = Assessment(**json.loads(row["target"]))
                        record["passed"] = actual.status == expected.status and set(actual.source_ids) == set(expected.source_ids)
                    elif args.component == "review":
                        actual = parse_model(body, Review)
                        expected = Review(**json.loads(row["target"]))
                        record.update(passed=actual.valid == expected.valid, expected=expected.valid, predicted=actual.valid)
                    else:
                        actual = parse_model(body, Followup)
                        expected = Followup(**json.loads(row["target"]))
                        context = json.loads(messages[0]["content"].split("\n", 1)[1])
                        previous = {(q["cell_id"], q["query"]) for q in context["previous_queries"]}
                        record["repeated_queries"] = [q.model_dump() for q in actual.queries if (q.cell_id, q.query) in previous]
                        record["passed"] = (actual.stop == expected.stop
                            and {q.cell_id for q in actual.queries} == {q.cell_id for q in expected.queries}
                            and not record["repeated_queries"])
            except (ValueError, TypeError) as error:
                record.update(passed=False, parse_error=str(error))
            (args.output / f"{row['id']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
            print(row["id"], result.status, record["passed"], flush=True)
            return record
    try:
        results = await asyncio.gather(*(run(row) for row in rows))
    finally:
        await model.aclose()
    summary = {"stage": args.stage, "component": args.component, "split": args.split, "state_id": config.get(key),
        "data_sha256": data_hash, "cases": len(results), "passed": None if args.stage == "writer" else sum(r["passed"] is True for r in results),
        "writer_semantics_scored": False, "full_rag_quality_claim": False,
        "false_positive": sum(r.get("expected") is False and r.get("predicted") is True for r in results),
        "false_negative": sum(r.get("expected") is True and r.get("predicted") is False for r in results),
        "format_or_transport_failures": sum("parse_error" in r for r in results)}
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
