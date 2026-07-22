import argparse
import json
import sys
from pathlib import Path

import uvicorn

from .config import get_settings
from .evaluate import evaluate
from .ingest import ingest_finewiki, ingest_markdown
from .migrate_sqlite import migrate_from_sqlite


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="rwkvrag-retrieval")
    commands = root.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8090)

    ingest = commands.add_parser("ingest-finewiki")
    ingest.add_argument("--path", type=Path, required=True)
    ingest.add_argument("--source", default="finewiki-zh")
    ingest.add_argument("--title", action="append", default=[])
    ingest.add_argument("--limit", type=int, default=0)
    ingest.add_argument("--batch-size", type=int, default=16)
    ingest.add_argument("--recreate", action="store_true")

    markdown = commands.add_parser("ingest-markdown")
    markdown.add_argument("--path", type=Path, required=True)
    markdown.add_argument("--source", default="local-markdown")
    markdown.add_argument("--limit", type=int, default=0)
    markdown.add_argument("--batch-size", type=int, default=16)
    markdown.add_argument("--recreate", action="store_true")

    evaluation = commands.add_parser("eval")
    evaluation.add_argument("--url", default="http://127.0.0.1:8090")
    evaluation.add_argument("--cases", type=Path, default=Path("eval/cases.jsonl"))
    evaluation.add_argument("--top-k", type=int, default=5)

    migration = commands.add_parser("migrate-sqlite")
    migration.add_argument("--path", type=Path, default=None)
    migration.add_argument("--batch-size", type=int, default=500)
    migration.add_argument("--recreate", action="store_true")
    return root


def main() -> None:
    arguments = parser().parse_args()
    if arguments.command == "serve":
        uvicorn.run(
            "llamaindex_retrieval.api:app",
            host=arguments.host,
            port=arguments.port,
        )
        return
    if arguments.command == "ingest-finewiki":
        stats = ingest_finewiki(
            settings=get_settings(),
            path=arguments.path,
            source=arguments.source,
            titles=set(arguments.title) or None,
            limit=arguments.limit,
            batch_size=arguments.batch_size,
            recreate=arguments.recreate,
        )
        print(json.dumps(stats, ensure_ascii=False))
        return
    if arguments.command == "ingest-markdown":
        stats = ingest_markdown(
            settings=get_settings(),
            path=arguments.path,
            source=arguments.source,
            limit=arguments.limit,
            batch_size=arguments.batch_size,
            recreate=arguments.recreate,
        )
        print(json.dumps(stats, ensure_ascii=False))
        return
    if arguments.command == "eval":
        report = evaluate(arguments.url, arguments.cases, arguments.top_k)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["passed"] != report["total"]:
            raise SystemExit(1)
    if arguments.command == "migrate-sqlite":
        last_reported = 0

        def report_progress(processed: int) -> None:
            nonlocal last_reported
            if processed - last_reported >= 10_000:
                print(f"已迁移 {processed} 个切片", file=sys.stderr, flush=True)
                last_reported = processed

        stats = migrate_from_sqlite(
            get_settings(),
            path=arguments.path,
            batch_size=arguments.batch_size,
            recreate=arguments.recreate,
            progress_callback=report_progress,
        )
        print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
