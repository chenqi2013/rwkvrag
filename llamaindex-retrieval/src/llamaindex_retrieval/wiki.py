"""Source-bound Wiki drafts. Generated text is never promoted to source evidence."""
import asyncio
import hashlib
import json
import logging
from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import uuid4

from .admin_service import AdminConflictError, AdminNotFoundError, AdminValidationError
from .citation_audit import audit_citations
from .repository import utc_now
from .schemas import SourceItem

logger = logging.getLogger(__name__)

PROMPT = """请将资料整理为知识条目正文：逐项列出其中明确记载的事实、参数、条件、操作要求和限制。
保留数值、单位、版本和否定。每条都引用对应的[资料 N]。只写有原文依据的内容，不要仅复述标题。"""


class WikiService:
    def __init__(self, settings, repository, index, pipeline, tasks):
        self.settings, self.repo, self.index = settings, repository, index
        self.pipeline, self.tasks = pipeline, tasks

    async def enqueue(self, file_id, *, automatic=False):
        item = await self.repo.get_file(file_id)
        if item is None:
            raise AdminNotFoundError("来源文件不存在")
        revision = item.get("last_indexed_revision")
        if item["status"] != "ready" or not revision:
            raise AdminConflictError("来源尚未成功入库或缺少版本记录，请先完成入库")
        if self.pipeline is None:
            raise AdminValidationError("Wiki 生成需要配置 RWKV 问答链路")
        binding = {"file_id": file_id, "revision": revision,
                   "index_version": item["last_indexed_index_version"],
                   "title": item["filename"], "knowledge_base_id": item["knowledge_base_id"],
                   "wiki_prompt": PROMPT, "wiki_prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest()}
        if await self._freshness({"binding": binding}) != "current":
            raise AdminConflictError("原文与活动索引版本不一致或不可确认，请先完成重建")
        identity = hashlib.sha256(json.dumps({"file": file_id,
            "parsed": revision["parsed_snapshot_sha256"], "prompt": PROMPT}, sort_keys=True).encode()).hexdigest()
        job_id = "wiki-" + (identity if automatic else uuid4().hex)
        job = await self.repo.create_job("wiki_generate", binding, job_id=job_id)
        if job["status"] == "pending":
            self.tasks.submit(job["id"])
        return job

    async def backfill(self):
        if not self.settings.wiki_auto_generate or self.pipeline is None:
            return
        async for item in self.repo.files.find({"status": "ready", "last_indexed_revision": {"$exists": True}}, {"id": 1}):
            try:
                await self.enqueue(item["id"], automatic=True)
            except (AdminConflictError, AdminNotFoundError) as error:
                # Rollback or concurrent deletion must not prevent API startup.
                logger.warning("Skipping Wiki backfill for file %s: %s", item["id"], error)

    def _materials(self, binding):
        revision = binding["revision"]
        uri = urlparse(revision["parsed_snapshot_uri"])
        path = Path(unquote(uri.path)).resolve()
        if uri.scheme != "file" or uri.netloc or not path.is_relative_to((self.settings.upload_dir / ".revisions").resolve()):
            raise ValueError("解析快照不在受管理的来源目录内")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != revision["parsed_snapshot_sha256"]:
            raise ValueError("解析快照哈希不符，停止 Wiki 生成")
        parsed = json.loads(data)
        if (parsed["file_id"] != binding["file_id"] or parsed["knowledge_base_id"] != binding["knowledge_base_id"]
                or parsed["source_sha256"] != revision["source_sha256"]):
            raise ValueError("解析快照来源绑定不符")
        documents = parsed["documents"]
        if not documents or sum(len(d["text"]) for d in documents) > self.settings.wiki_max_source_characters:
            raise ValueError("原文为空或超过 Wiki 单页材料上限；未截断原文，请拆分后生成")
        return [SourceItem(id=d["id_"], document_id=d["id_"], source="uploaded-document",
            title=str(d["metadata"].get("title") or binding["title"]),
            uri=d["metadata"].get("uri"), score=1.0, snippet=d["text"],
            metadata={**d["metadata"], **revision}) for d in documents]

    async def generate(self, job):
        # Atomic claim prevents duplicate scheduling of the same Wiki job.
        if not await self.repo.claim_wiki_job(job["id"]):
            return
        binding = job["payload"]
        existing = await self.repo.get_wiki_version(job["id"])
        if existing:
            await self.repo.update_job(job["id"], {"status": "completed" if existing["status"] == "draft" else "failed",
                "stage": "completed" if existing["status"] == "draft" else "failed", "progress": 100,
                "completed_at": utc_now()})
            return
        try:
            materials = await asyncio.to_thread(self._materials, binding)
            response = await self.pipeline.ask_materials(binding.get("wiki_prompt", PROMPT), materials)
            payload = response.model_dump(mode="json")
            bounds = payload["generation"].get("answer_span")
            raw = payload["answer"]
            body = raw[bounds[0]:bounds[1]] if bounds else ""
            audit = audit_citations(body, payload["sources"])
            valid = (payload["generation"].get("status") == "completed" and bool(body.strip())
                     and bool(audit["label_ids"]) and not audit["unknown_label_ids"] and not audit["invalid_labels"])
            page = {"id": job["id"], "page_id": binding["file_id"],
                    "title": binding["title"], "knowledge_base_id": binding["knowledge_base_id"],
                    "created_at": job["created_at"], "generated_at": utc_now(), "binding": binding,
                    "status": "draft" if valid else "generation_failed", "body": body,
                    "citation_audit": audit, "semantic_reviewed": False,
                    "response": payload}
            await self.repo.save_wiki_version(page)
            await self.repo.update_job(job["id"], {"status": "completed" if valid else "failed",
                "stage": "completed" if valid else "failed", "progress": 100,
                "message": "Wiki 草稿已生成，尚未人工审核" if valid else "Wiki 生成未通过执行或引用格式检查",
                "error": None if valid else f"生成状态：{payload['generation'].get('status')}；原始响应及引用检查已保留，可从 Wiki 历史查看", "completed_at": utc_now()})
        except asyncio.CancelledError:
            await self.repo.update_job(job["id"], {"status": "pending", "stage": "queued"})
            raise
        except Exception as error:
            await self.repo.update_job(job["id"], {"status": "failed", "stage": "failed", "progress": 100,
                "error": str(error), "message": "Wiki 生成失败，来源文档未受影响", "completed_at": utc_now()})

    async def _freshness(self, page):
        binding = page["binding"]
        item = await self.repo.get_file(binding["file_id"])
        if not item:
            return "source_deleted"
        if item.get("sha256") != binding["revision"]["source_sha256"]:
            return "source_changed"
        try:
            current = await asyncio.to_thread(self.index.versions.current)
            sources = await asyncio.to_thread(self.index.versions.source_revisions, current, binding["file_id"])
            if not sources["chunks"] or sources["untracked_chunks"]:
                return "source_missing_or_untracked"
            expected = binding["revision"]["parsed_snapshot_sha256"]
            if {r["parsed_snapshot_sha256"] for r in sources["revisions"]} != {expected}:
                return "source_changed"
            return "current"
        except Exception:
            return "unknown"

    async def list_pages(self, knowledge_base_id=None):
        query = {"knowledge_base_id": knowledge_base_id} if knowledge_base_id else {}
        # Latest attempt is shown; failed output never masquerades as a valid draft.
        cursor = await self.repo.wiki_versions.aggregate([
            {"$match": query}, {"$sort": {"created_at": -1, "id": -1}},
            {"$group": {"_id": "$page_id", "page": {"$first": "$$ROOT"}}},
            {"$replaceRoot": {"newRoot": "$page"}}, {"$sort": {"created_at": -1}},
            {"$limit": 100}, {"$project": {"_id": 0, "response": 0, "body": 0}}])
        pages = await cursor.to_list(length=100)
        for page in pages:
            page["freshness"] = await self._freshness(page)
        return pages

    async def detail(self, identity):
        page = await self.repo.get_wiki_version(identity)
        if not page:
            raise AdminNotFoundError("Wiki 版本不存在")
        page["freshness"] = await self._freshness(page)
        return page
