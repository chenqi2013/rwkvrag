import asyncio
import logging
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet

from .config import Settings
from .ingest import ingest_finewiki, ingest_uploaded_documents, parquet_files
from .lexical_index import LexicalIndex
from .parsers import parse_uploaded_file
from .repository import MongoRepository, utc_now

logger = logging.getLogger(__name__)


class TaskManager:
    def __init__(
        self,
        settings: Settings,
        repository: MongoRepository,
        lexical_index: LexicalIndex,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.lexical_index = lexical_index
        self.semaphore = asyncio.Semaphore(settings.task_workers)
        self.tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        for job in await self.repository.recoverable_jobs():
            await self.repository.update_job(
                job["id"],
                {
                    "status": "pending",
                    "stage": "queued",
                    "message": "服务重启后恢复任务",
                    "error": None,
                },
            )
            self.submit(job["id"])

    async def shutdown(self) -> None:
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

    def submit(self, job_id: str) -> None:
        task = asyncio.create_task(self._run_job(job_id), name=f"admin-job-{job_id}")
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _run_job(self, job_id: str) -> None:
        async with self.semaphore:
            job = await self.repository.get_job(job_id)
            if job is None:
                return
            await self.repository.update_job(
                job_id,
                {
                    "status": "running",
                    "progress": 5,
                    "stage": "starting",
                    "message": "任务开始执行",
                    "started_at": utc_now(),
                    "completed_at": None,
                    "error": None,
                },
            )
            try:
                if job["kind"] in {"file_ingest", "file_reindex"}:
                    await self._run_file_job(job_id, job["payload"])
                elif job["kind"] == "finewiki_import":
                    await self._run_finewiki_job(job_id, job["payload"])
                else:
                    raise ValueError(f"不支持的任务类型：{job['kind']}")
            except asyncio.CancelledError:
                await self.repository.update_job(
                    job_id,
                    {
                        "status": "pending",
                        "stage": "queued",
                        "message": "服务停止，任务将在下次启动时继续",
                    },
                )
                raise
            except Exception as error:
                logger.exception("admin job failed: %s", job_id)
                await self.repository.update_job(
                    job_id,
                    {
                        "status": "failed",
                        "progress": 100,
                        "stage": "failed",
                        "message": "任务执行失败",
                        "error": str(error),
                        "completed_at": utc_now(),
                    },
                )
                file_id = job["payload"].get("file_id")
                if file_id:
                    await self.repository.update_file(
                        file_id,
                        {"status": "failed", "error": str(error)},
                    )

    async def _run_file_job(self, job_id: str, payload: dict[str, Any]) -> None:
        file_id = str(payload["file_id"])
        file_item = await self.repository.get_file(file_id)
        if file_item is None:
            raise FileNotFoundError(f"文件记录不存在：{file_id}")
        path = Path(str(payload["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"原始文件不存在：{path}")
        await self.repository.update_file(file_id, {"status": "processing", "error": None})
        await self.repository.update_job(
            job_id,
            {"progress": 15, "stage": "parsing", "message": "正在解析文档"},
        )
        documents = await asyncio.to_thread(
            parse_uploaded_file,
            path,
            file_id,
            str(file_item["knowledge_base_id"]),
        )
        await self.repository.update_job(
            job_id,
            {
                "progress": 30,
                "stage": "cleaning",
                "message": "正在清理旧切片",
                "documents_processed": 0,
                "nodes_processed": 0,
            },
        )
        await asyncio.to_thread(self.lexical_index.delete_by_field, "file_id", file_id)
        loop = asyncio.get_running_loop()
        total_documents = max(1, len(documents))
        progress_updates = []

        def progress(documents_processed: int, nodes_processed: int) -> None:
            percentage = min(95, 35 + int(60 * documents_processed / total_documents))
            progress_updates.append(
                asyncio.run_coroutine_threadsafe(
                    self.repository.update_job(
                        job_id,
                        {
                            "progress": percentage,
                            "stage": "indexing",
                            "message": "正在切片并建立 BM25 索引",
                            "documents_processed": documents_processed,
                            "nodes_processed": nodes_processed,
                        },
                    ),
                    loop,
                )
            )

        stats = await asyncio.to_thread(
            ingest_uploaded_documents,
            self.settings,
            documents,
            8,
            progress,
            self.lexical_index,
        )
        if progress_updates:
            await asyncio.gather(*(asyncio.wrap_future(update) for update in progress_updates))
        await self.repository.update_file(
            file_id,
            {
                "status": "ready",
                "node_count": stats["nodes"],
                "error": None,
                "last_job_id": job_id,
            },
        )
        await self._complete_job(job_id, stats)

    async def _run_finewiki_job(self, job_id: str, payload: dict[str, Any]) -> None:
        path = Path(str(payload["path"]))
        if not path.exists():
            raise FileNotFoundError(f"FineWiki 路径不存在：{path}")
        total_documents = await asyncio.to_thread(self._finewiki_total, path, payload)
        await self.repository.update_job(
            job_id,
            {
                "progress": 10,
                "stage": "reading",
                "message": f"正在读取 FineWiki，预计 {total_documents} 篇文档",
            },
        )
        loop = asyncio.get_running_loop()
        progress_updates = []

        def progress(documents_processed: int, nodes_processed: int) -> None:
            percentage = min(95, 10 + int(85 * documents_processed / max(1, total_documents)))
            progress_updates.append(
                asyncio.run_coroutine_threadsafe(
                    self.repository.update_job(
                        job_id,
                        {
                            "progress": percentage,
                            "stage": "indexing",
                            "message": "正在切片并建立 BM25 索引",
                            "documents_processed": documents_processed,
                            "nodes_processed": nodes_processed,
                        },
                    ),
                    loop,
                )
            )

        stats = await asyncio.to_thread(
            ingest_finewiki,
            self.settings,
            path,
            str(payload["source"]),
            set(payload.get("titles") or []) or None,
            int(payload.get("limit") or 0),
            int(payload.get("batch_size") or 8),
            bool(payload.get("recreate")),
            str(payload["knowledge_base_id"]),
            job_id,
            progress,
            self.lexical_index,
        )
        if progress_updates:
            await asyncio.gather(*(asyncio.wrap_future(update) for update in progress_updates))
        await self._complete_job(job_id, stats)

    @staticmethod
    def _finewiki_total(path: Path, payload: dict[str, Any]) -> int:
        titles = payload.get("titles") or []
        if titles:
            total = len(titles)
        else:
            total = sum(
                parquet.ParquetFile(file_path).metadata.num_rows
                for file_path in parquet_files(path)
            )
        limit = int(payload.get("limit") or 0)
        return min(total, limit) if limit else total

    async def _complete_job(self, job_id: str, stats: dict[str, int]) -> None:
        await self.repository.update_job(
            job_id,
            {
                "status": "completed",
                "progress": 100,
                "stage": "completed",
                "message": "任务执行完成",
                "documents_processed": stats["documents"],
                "nodes_processed": stats["nodes"],
                "error": None,
                "completed_at": utc_now(),
            },
        )
