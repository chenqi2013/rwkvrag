"""Persist only server-observed provider snapshots, never answers or client materials."""
import asyncio
import hashlib
import io
import json
from urllib.parse import urlsplit

from fastapi import UploadFile

from .admin_service import AdminConflictError, AdminNotFoundError, AdminValidationError


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def valid_url(url):
    try:
        parsed = urlsplit(url)
        return parsed.scheme in {"https", "http"} and bool(parsed.hostname) and not (
            parsed.username or parsed.password)
    except (ValueError, TypeError):
        return False


def validate_provenance(provenance, source_sha):
    required = ("snapshot_id", "url", "title", "retrieved_at", "provider", "sha256", "content_status")
    if (not isinstance(provenance, dict) or any(not isinstance(provenance.get(k), str) for k in required)
            or not valid_url(provenance["url"]) or provenance["content_status"] != "fetched"
            or provenance["sha256"] != source_sha or not provenance["snapshot_id"]):
        raise AdminValidationError("网页来源绑定与保存内容不一致")


class WebSnapshotService:
    def __init__(self, repository, admin):
        self.repo, self.admin = repository, admin
        # Cross-process collisions are also protected by deterministic file identity.
        self._save_lock = asyncio.Lock()

    async def record(self, snapshot):
        if (not isinstance(snapshot.get("text"), str) or not snapshot["text"].strip()
                or text_sha(snapshot["text"]) != snapshot.get("sha256")
                or not valid_url(snapshot.get("url"))
                or snapshot.get("content_status") not in {"fetched", "snippet_only"}):
            raise AdminValidationError("检索供应商快照不完整或哈希不符")
        return await self.repo.record_web_snapshot(snapshot)

    async def preview(self, identity):
        record = await self.repo.get_web_snapshot(identity)
        if record is None:
            raise AdminNotFoundError("没有服务器检索快照；客户端材料不能作为网页收据")
        if (record.get("id") != identity or not isinstance(record.get("text"), str)
                or text_sha(record["text"]) != record.get("sha256")
                or not valid_url(record.get("url"))):
            raise AdminConflictError("服务器网页快照完整性检查失败")
        result = {k: record.get(k) for k in (
            "id", "title", "url", "text", "sha256", "retrieved_at", "content_status", "provider")}
        allowed = record.get("content_status") == "fetched" and bool(record["text"].strip())
        result["save_allowed"] = allowed
        if not allowed:
            result["reason"] = "仅有搜索摘要或无正文；不能作为网页正文保存到知识库"
        elif len(record["text"].encode("utf-8")) > self.admin.settings.max_upload_bytes:
            result.update(save_allowed=False, reason="网页正文超过上传大小限制")
        return result

    async def save(self, identity, knowledge_base_id, expected_sha256, confirmed):
        if confirmed is not True:
            raise AdminValidationError("需要确认保存网页原文")
        snapshot = await self.preview(identity)
        if expected_sha256 != snapshot["sha256"]:
            raise AdminConflictError("预览内容哈希已不一致，请重新查看快照")
        if not snapshot["save_allowed"]:
            raise AdminValidationError(snapshot["reason"])
        if await self.repo.get_knowledge_base(knowledge_base_id) is None:
            raise AdminNotFoundError("目标知识库不存在")
        # Keep URL exact: removing query parameters could conflate different pages.
        file_id = text_sha(json.dumps([knowledge_base_id, snapshot["url"], snapshot["sha256"]],
                                     ensure_ascii=False, separators=(",", ":")))
        job_id = "web-ingest-" + file_id
        provenance = {"snapshot_id": identity, **{k: snapshot[k] for k in (
            "title", "url", "sha256", "retrieved_at", "content_status", "provider")}}
        validate_provenance(provenance, snapshot["sha256"])
        async with self._save_lock:
            item = await self.repo.get_file(file_id)
            if item is not None:
                bound = item.get("web_provenance") or {}
                if (item.get("knowledge_base_id") != knowledge_base_id
                        or item.get("sha256") != snapshot["sha256"]
                        or bound.get("url") != snapshot["url"] or bound.get("sha256") != snapshot["sha256"]):
                    raise AdminConflictError("已保存文件曾被修订，不能覆盖或重新绑定原网页")
                current_job_id = item.get("last_job_id") or job_id
                job = await self.repo.get_job(current_job_id)
                if job is None:
                    # Resume only a file whose first upload stopped before job creation.
                    if item.get("last_job_id") or item.get("status") != "pending":
                        raise AdminConflictError("已保存文件的入库任务记录缺失，需先检查状态")
                    job = await self.repo.create_job("file_ingest", {
                        "file_id": file_id, "knowledge_base_id": knowledge_base_id,
                        "path": item["path"]}, job_id=job_id)
                if job["status"] == "pending":
                    await self.repo.update_file(file_id, {"last_job_id": current_job_id})
                    self.admin.tasks.submit(current_job_id)
                return {"file_id": file_id, "job_id": current_job_id, "status": job["status"], "existing": True}
            upload = UploadFile(filename="web-" + file_id[:16] + ".md",
                                file=io.BytesIO(snapshot["text"].encode("utf-8")))
            try:
                item, job = await self.admin.upload_file(upload, knowledge_base_id,
                    web_provenance=provenance, file_identity=file_id)
            except FileExistsError as error:
                await upload.close()
                raise AdminConflictError("相同网页正在保存或有未完成的保存，请稍后重试并检查任务") from error
            return {"file_id": item["id"], "job_id": job["id"], "status": job["status"], "existing": False}
