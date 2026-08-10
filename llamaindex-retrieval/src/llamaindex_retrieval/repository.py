import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pymongo import ASCENDING, DESCENDING, AsyncMongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError


def utc_now() -> datetime:
    return datetime.now(UTC)


class RepositoryConflictError(RuntimeError):
    pass


class MongoRepository:
    def __init__(self, url: str, database: str) -> None:
        self.client = AsyncMongoClient(url, tz_aware=True)
        self.database = self.client[database]
        self.knowledge_bases = self.database["knowledge_bases"]
        self.files = self.database["files"]
        self.jobs = self.database["jobs"]
        self.search_tests = self.database["search_tests"]
        self.search_test_runs = self.database["search_test_runs"]

    async def connect(self) -> None:
        await self.client.admin.command("ping")
        await self.knowledge_bases.create_index("id", unique=True)
        await self.knowledge_bases.create_index("name", unique=True)
        await self.files.create_index("id", unique=True)
        await self.files.create_index(
            [("knowledge_base_id", ASCENDING), ("created_at", DESCENDING)]
        )
        await self.files.create_index(
            [("knowledge_base_id", ASCENDING), ("sha256", ASCENDING)], unique=True
        )
        await self.jobs.create_index("id", unique=True)
        await self.jobs.create_index([("status", ASCENDING), ("created_at", ASCENDING)])
        await self.search_tests.create_index("id", unique=True)
        await self.search_tests.create_index("key", unique=True)
        await self.search_tests.create_index([("updated_at", DESCENDING)])
        await self.search_test_runs.create_index("id", unique=True)
        await self.search_test_runs.create_index(
            [("test_id", ASCENDING), ("run_number", ASCENDING)], unique=True
        )
        await self.search_test_runs.create_index(
            [("test_id", ASCENDING), ("created_at", DESCENDING)]
        )
        now = utc_now()
        await self.knowledge_bases.update_one(
            {"id": "default"},
            {
                "$setOnInsert": {
                    "id": "default",
                    "name": "默认知识库",
                    "description": "默认的 FineWiki 和本地文档知识库",
                    "created_at": now,
                    "updated_at": now,
                }
            },
            upsert=True,
        )

    async def close(self) -> None:
        await self.client.close()

    async def health(self) -> dict[str, Any]:
        started = utc_now()
        result = await self.client.admin.command("ping")
        elapsed = (utc_now() - started).total_seconds() * 1000
        return {"ok": result.get("ok") == 1, "latency_ms": round(elapsed, 2)}

    async def create_knowledge_base(self, name: str, description: str) -> dict[str, Any]:
        now = utc_now()
        item = {
            "id": uuid4().hex,
            "name": name.strip(),
            "description": description.strip(),
            "created_at": now,
            "updated_at": now,
        }
        try:
            await self.knowledge_bases.insert_one(item)
        except DuplicateKeyError as error:
            raise RepositoryConflictError(f"知识库名称已存在：{name}") from error
        return {**item, "file_count": 0}

    async def list_knowledge_bases(self) -> list[dict[str, Any]]:
        cursor = self.knowledge_bases.find({}, {"_id": 0}).sort("created_at", ASCENDING)
        items = await cursor.to_list(length=None)
        for item in items:
            item["file_count"] = await self.files.count_documents(
                {"knowledge_base_id": item["id"]}
            )
        return items

    async def get_knowledge_base(self, knowledge_base_id: str) -> dict[str, Any] | None:
        item = await self.knowledge_bases.find_one(
            {"id": knowledge_base_id}, {"_id": 0}
        )
        if item is not None:
            item["file_count"] = await self.files.count_documents(
                {"knowledge_base_id": knowledge_base_id}
            )
        return item

    async def update_knowledge_base(
        self,
        knowledge_base_id: str,
        values: dict[str, Any],
    ) -> dict[str, Any] | None:
        updates = {key: value.strip() if isinstance(value, str) else value for key, value in values.items()}
        updates["updated_at"] = utc_now()
        try:
            await self.knowledge_bases.update_one({"id": knowledge_base_id}, {"$set": updates})
        except DuplicateKeyError as error:
            raise RepositoryConflictError("知识库名称已存在") from error
        return await self.get_knowledge_base(knowledge_base_id)

    async def delete_knowledge_base(self, knowledge_base_id: str) -> bool:
        result = await self.knowledge_bases.delete_one({"id": knowledge_base_id})
        return result.deleted_count == 1

    async def create_file(self, item: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        record = {
            **item,
            "status": "pending",
            "node_count": 0,
            "error": None,
            "last_job_id": None,
            "created_at": now,
            "updated_at": now,
        }
        try:
            await self.files.insert_one(record)
        except DuplicateKeyError as error:
            raise RepositoryConflictError("同一知识库中已经上传过相同内容的文件") from error
        return record

    async def get_file(self, file_id: str) -> dict[str, Any] | None:
        return await self.files.find_one({"id": file_id}, {"_id": 0})

    async def list_files(
        self,
        knowledge_base_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        query = {"knowledge_base_id": knowledge_base_id} if knowledge_base_id else {}
        cursor = self.files.find(query, {"_id": 0}).sort("created_at", DESCENDING).limit(limit)
        return await cursor.to_list(length=limit)

    async def update_file(self, file_id: str, values: dict[str, Any]) -> dict[str, Any] | None:
        updates = {**values, "updated_at": utc_now()}
        await self.files.update_one({"id": file_id}, {"$set": updates})
        return await self.get_file(file_id)

    async def delete_file(self, file_id: str) -> bool:
        result = await self.files.delete_one({"id": file_id})
        return result.deleted_count == 1

    async def create_job(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        item = {
            "id": uuid4().hex,
            "kind": kind,
            "status": "pending",
            "progress": 0,
            "stage": "queued",
            "message": "任务已进入队列",
            "documents_processed": 0,
            "nodes_processed": 0,
            "error": None,
            "payload": payload,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
        await self.jobs.insert_one(item)
        return item

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        return await self.jobs.find_one({"id": job_id}, {"_id": 0})

    async def list_jobs(
        self,
        status: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        query = {"status": status} if status else {}
        cursor = self.jobs.find(query, {"_id": 0}).sort("created_at", DESCENDING).limit(limit)
        return await cursor.to_list(length=limit)

    async def update_job(self, job_id: str, values: dict[str, Any]) -> dict[str, Any] | None:
        updates = {**values, "updated_at": utc_now()}
        await self.jobs.update_one({"id": job_id}, {"$set": updates})
        return await self.get_job(job_id)

    async def recoverable_jobs(self) -> list[dict[str, Any]]:
        cursor = self.jobs.find(
            {"status": {"$in": ["pending", "running"]}}, {"_id": 0}
        ).sort("created_at", ASCENDING)
        return await cursor.to_list(length=None)

    async def delete_files_for_knowledge_base(self, knowledge_base_id: str) -> None:
        await self.files.delete_many({"knowledge_base_id": knowledge_base_id})

    async def delete_jobs_for_knowledge_base(self, knowledge_base_id: str) -> None:
        await self.jobs.delete_many({"payload.knowledge_base_id": knowledge_base_id})

    async def record_search_test_run(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
        *,
        test_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Append an immutable execution result to a question's test history."""
        now = utc_now()
        if test_id:
            test = await self.search_tests.find_one_and_update(
                {"id": test_id},
                {
                    "$set": {"request": request, "updated_at": now},
                    "$inc": {"run_count": 1},
                },
                return_document=ReturnDocument.AFTER,
            )
        else:
            question = str(request.get("question") or "").strip()
            knowledge_base_id = request.get("knowledge_base_id")
            key = self._search_test_key(question, knowledge_base_id)
            test = await self.search_tests.find_one_and_update(
                {"key": key},
                {
	                    "$setOnInsert": {
	                        "id": uuid4().hex,
	                        "key": key,
	                        "question": question,
	                        "knowledge_base_id": knowledge_base_id,
	                        "created_at": now,
	                    },
                    "$set": {"request": request, "updated_at": now},
                    "$inc": {"run_count": 1},
                },
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
        if test is None:
            return None

        run = {
            "id": uuid4().hex,
            "test_id": test["id"],
            "run_number": int(test["run_count"]),
            "request": request,
            "response": response,
            "created_at": now,
        }
        await self.search_test_runs.insert_one(run)
        await self.search_tests.update_one(
            {"id": test["id"]}, {"$set": {"latest_run_id": run["id"]}}
        )
        return run

    async def list_search_tests(self, limit: int = 100) -> list[dict[str, Any]]:
        cursor = self.search_tests.find({}, {"_id": 0}).sort("updated_at", DESCENDING).limit(limit)
        tests = await cursor.to_list(length=limit)
        for test in tests:
            latest_run_id = test.get("latest_run_id")
            test["latest_run"] = (
                await self.search_test_runs.find_one({"id": latest_run_id}, {"_id": 0})
                if latest_run_id
                else None
            )
        return tests

    async def get_search_test(self, test_id: str) -> dict[str, Any] | None:
        test = await self.search_tests.find_one({"id": test_id}, {"_id": 0})
        if test is None:
            return None
        cursor = self.search_test_runs.find({"test_id": test_id}, {"_id": 0}).sort(
            "run_number", ASCENDING
        )
        test["runs"] = await cursor.to_list(length=None)
        return test

    @staticmethod
    def _search_test_key(question: str, knowledge_base_id: object) -> str:
        value = json.dumps(
            {"question": question, "knowledge_base_id": knowledge_base_id},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
