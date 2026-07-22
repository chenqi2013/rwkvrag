import asyncio
import hashlib
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from .config import Settings
from .lexical_index import LexicalIndex
from .parsers import SUPPORTED_EXTENSIONS
from .repository import MongoRepository, RepositoryConflictError
from .schemas import FineWikiImportRequest
from .tasks import TaskManager


class AdminNotFoundError(RuntimeError):
    pass


class AdminConflictError(RuntimeError):
    pass


class AdminValidationError(ValueError):
    pass


class AdminService:
    def __init__(
        self,
        settings: Settings,
        repository: MongoRepository,
        tasks: TaskManager,
        lexical_index: LexicalIndex,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.tasks = tasks
        self.lexical_index = lexical_index

    async def upload_file(
        self,
        upload: UploadFile,
        knowledge_base_id: str,
    ) -> tuple[dict, dict]:
        if await self.repository.get_knowledge_base(knowledge_base_id) is None:
            raise AdminNotFoundError(f"知识库不存在：{knowledge_base_id}")
        filename = Path(upload.filename or "").name
        if not filename:
            raise AdminValidationError("文件名不能为空")
        extension = Path(filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise AdminValidationError(f"不支持 {extension or '无扩展名'}，支持：{supported}")
        file_id = uuid4().hex
        directory = self.settings.upload_dir / knowledge_base_id / file_id
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / filename
        digest = hashlib.sha256()
        size = 0
        try:
            with path.open("wb") as output:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.settings.max_upload_bytes:
                        raise AdminValidationError(
                            f"文件超过 {self.settings.max_upload_bytes // 1024 // 1024}MB 限制"
                        )
                    digest.update(chunk)
                    output.write(chunk)
            if size == 0:
                raise AdminValidationError("不能上传空文件")
            try:
                file_item = await self.repository.create_file(
                    {
                        "id": file_id,
                        "knowledge_base_id": knowledge_base_id,
                        "filename": filename,
                        "path": str(path),
                        "content_type": upload.content_type or "application/octet-stream",
                        "extension": extension,
                        "size": size,
                        "sha256": digest.hexdigest(),
                        "source": "uploaded-document",
                    }
                )
            except RepositoryConflictError as error:
                raise AdminConflictError(str(error)) from error
            job = await self.repository.create_job(
                "file_ingest",
                {
                    "file_id": file_id,
                    "knowledge_base_id": knowledge_base_id,
                    "path": str(path),
                },
            )
            file_item = await self.repository.update_file(
                file_id,
                {"last_job_id": job["id"]},
            )
            self.tasks.submit(job["id"])
            return file_item or {}, job
        except Exception:
            if not await self.repository.get_file(file_id):
                shutil.rmtree(directory, ignore_errors=True)
            raise
        finally:
            await upload.close()

    async def reindex_file(self, file_id: str) -> dict:
        file_item = await self.repository.get_file(file_id)
        if file_item is None:
            raise AdminNotFoundError(f"文件不存在：{file_id}")
        if file_item["status"] in {"pending", "processing", "deleting"}:
            raise AdminConflictError("文件当前正在执行任务，不能重复索引")
        job = await self.repository.create_job(
            "file_reindex",
            {
                "file_id": file_id,
                "knowledge_base_id": file_item["knowledge_base_id"],
                "path": file_item["path"],
            },
        )
        await self.repository.update_file(
            file_id,
            {"status": "pending", "last_job_id": job["id"], "error": None},
        )
        self.tasks.submit(job["id"])
        return job

    async def delete_file(self, file_id: str) -> None:
        file_item = await self.repository.get_file(file_id)
        if file_item is None:
            raise AdminNotFoundError(f"文件不存在：{file_id}")
        if file_item["status"] in {"pending", "processing"}:
            raise AdminConflictError("文件正在处理，完成后才能删除")
        await self.repository.update_file(file_id, {"status": "deleting"})
        await asyncio.to_thread(self.lexical_index.delete_by_field, "file_id", file_id)
        path = Path(file_item["path"])
        await asyncio.to_thread(shutil.rmtree, path.parent, True)
        await self.repository.delete_file(file_id)

    async def create_finewiki_job(self, request: FineWikiImportRequest) -> dict:
        if await self.repository.get_knowledge_base(request.knowledge_base_id) is None:
            raise AdminNotFoundError(f"知识库不存在：{request.knowledge_base_id}")
        path = Path(request.path).expanduser().resolve()
        if not path.exists():
            raise AdminValidationError(f"FineWiki 路径不存在：{path}")
        payload = request.model_dump()
        payload["path"] = str(path)
        job = await self.repository.create_job("finewiki_import", payload)
        self.tasks.submit(job["id"])
        return job

    async def delete_knowledge_base(self, knowledge_base_id: str) -> None:
        if knowledge_base_id == "default":
            raise AdminConflictError("默认知识库不能删除")
        knowledge_base = await self.repository.get_knowledge_base(knowledge_base_id)
        if knowledge_base is None:
            raise AdminNotFoundError(f"知识库不存在：{knowledge_base_id}")
        running_jobs = [
            job
            for job in await self.repository.list_jobs(status="running")
            if job.get("payload", {}).get("knowledge_base_id") == knowledge_base_id
        ]
        if running_jobs:
            raise AdminConflictError("知识库仍有运行中的任务")
        await asyncio.to_thread(
            self.lexical_index.delete_by_field,
            "knowledge_base_id",
            knowledge_base_id,
        )
        directory = self.settings.upload_dir / knowledge_base_id
        await asyncio.to_thread(shutil.rmtree, directory, True)
        await self.repository.delete_files_for_knowledge_base(knowledge_base_id)
        await self.repository.delete_jobs_for_knowledge_base(knowledge_base_id)
        await self.repository.delete_knowledge_base(knowledge_base_id)

    async def health(self) -> dict:
        mongo_result, lexical_result = await asyncio.gather(
            self.repository.health(),
            asyncio.to_thread(self.lexical_index.health),
            return_exceptions=True,
        )
        mongodb = self._health_result(mongo_result)
        lexical = self._health_result(lexical_result)
        status = "ok" if all(item.get("ok") for item in [mongodb, lexical]) else "degraded"
        return {
            "status": status,
            "mongodb": mongodb,
            "lexical": lexical,
        }

    @staticmethod
    def _health_result(result: object) -> dict:
        if isinstance(result, Exception):
            return {"ok": False, "error": str(result)}
        return dict(result) if isinstance(result, dict) else {"ok": False}
