"""Planner transfer diagnostic on previously exposed comparison questions.

Structural validity is not semantic success. Inspect saved plans against requests.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.model_client import model_client_class, model_client_options
from llamaindex_retrieval.rwkv_pipeline import conversation, structured_body
from llamaindex_retrieval.schemas import ConversationMessage
from llamaindex_retrieval.task_matrix import TaskPlan, format_repair_prompt, parse_model, plan_prompt


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.settings.read_text())
    config.update(native_base_url=args.base_url, rwkvos_planner_state_id=args.state_id)
    settings = Settings(**config)
    client = model_client_class(settings)(**model_client_options(settings))
    (args.output / "MANIFEST.json").write_text(json.dumps({
        "fixtures_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
        "state_id": args.state_id, "repair": args.repair, "blind": False, "semantic_scoring": "manual"}, indent=2))
    gate = asyncio.Semaphore(2)

    async def run(row):
        async with gate:
            task = conversation(row["question"], [ConversationMessage(**m) for m in row["history"]])
            prompt = plan_prompt(task, 8)
            result = await client.complete([{"role": "user", "content": prompt}],
                stage="planner", assistant_prefill="<think></think", max_tokens=1024)
            record = {"id": row["id"], "question": row["question"], "raw_text": result.raw_text,
                      "trace": result.trace, "semantic_passed": None}
            try:
                record["plan"] = parse_model(structured_body(result), TaskPlan).model_dump()
            except (ValueError, TypeError) as error:
                record["parse_error"] = str(error)
                if args.repair:
                    retry = await client.complete([{"role": "user", "content": format_repair_prompt(prompt, result.raw_text, error)}],
                        stage="planner", assistant_prefill="<think></think", max_tokens=1024)
                    record["repair_trace"] = retry.trace
                    record["repair_raw_text"] = retry.raw_text
                    try:
                        record["plan"] = parse_model(structured_body(retry), TaskPlan).model_dump()
                    except (ValueError, TypeError) as retry_error:
                        record["repair_parse_error"] = str(retry_error)
            (args.output / f"{row['id']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
            print(row["id"], "valid structure" if "plan" in record else "invalid", flush=True)

    try:
        await asyncio.gather(*(run(json.loads(line)) for line in args.fixtures.read_text().splitlines()))
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
