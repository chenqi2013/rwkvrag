"""Staged index publication. Old generations are retained; no automatic GC.

The lock index deliberately has no expiry: after a process crash, an operator
must confirm the writer stopped before removing the lock. Guessing expiry could
let a paused writer publish an obsolete snapshot over a newer generation.
"""
from contextlib import contextmanager
from copy import copy
from functools import wraps
from threading import local
from uuid import uuid4

from opensearchpy import RequestError


def serialized_write(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.versions.write_lock():
            return method(self, *args, **kwargs)
    return wrapped


class IndexVersions:
    def __init__(self, index):
        self.index = index
        self.client = index.client
        self.base = index.settings.opensearch_index
        self.alias = self.base + "-active"
        self.lock_name = self.base + "-write-lock"
        self.local = local()

    @contextmanager
    def write_lock(self):
        if getattr(self.local, "held", False):
            yield
            return
        # Atomic index creation serializes all application instances, not just
        # this Python process. A conflict fails the job without touching data.
        try:
            self.client.indices.create(index=self.lock_name, body={
                "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                "mappings": {"_meta": {"owner": uuid4().hex}},
            })
        except RequestError as error:
            if error.error == "resource_already_exists_exception":
                raise RuntimeError("索引已有写入任务或待恢复的写锁，请稍后重试；不要自动删除写锁") from error
            raise
        self.local.held = True
        try:
            yield
        finally:
            self.local.held = False
            self.client.indices.delete(index=self.lock_name)

    def initialize(self):
        if self.client.indices.exists_alias(name=self.alias):
            self.current()
            return
        with self.write_lock():
            if self.client.indices.exists_alias(name=self.alias):
                self.current()
                return
            # Adopt the existing physical index without renaming or deleting it.
            if not self.client.indices.exists(index=self.base):
                self.client.indices.create(index=self.base, body=self.index.index_definition())
            self.client.indices.update_aliases(body={"actions": [
                {"add": {"index": self.base, "alias": self.alias, "is_write_index": True}}
            ]})

    def current(self):
        aliases = self.client.indices.get_alias(name=self.alias)
        if len(aliases) != 1:
            raise RuntimeError("检索入口必须且只能指向一个索引版本")
        name, info = next(iter(aliases.items()))
        options = info["aliases"][self.alias]
        if options.get("filter") or any(k in options for k in ("routing", "index_routing", "search_routing")):
            raise RuntimeError("版本入口不支持过滤或路由别名")
        return name

    @contextmanager
    def replacement(self, *, file_id=None, source_revision=None):
        with self.write_lock():
            previous = self.current()
            target = self.base + "-v-" + uuid4().hex
            definition = self.index.index_definition()
            definition["mappings"]["_meta"] = {
                "rwkvrag_build": {"previous_index": previous, "file_id": file_id,
                                  "source_revision": source_revision}}
            self.client.indices.create(index=target, body=definition)
            staged = copy(self.index)
            staged.index_name = target
            # A copy shares this lock; re-entrant calls run in the same thread.
            try:
                if file_id is not None:
                    query = {"bool": {"must_not": [{"term": {"file_id": file_id}}]}}
                    self.client.indices.refresh(index=previous)
                    expected = self.client.count(index=previous, body={"query": query})["count"]
                    result = self.client.reindex(body={
                        "source": {"index": previous, "query": query},
                        "dest": {"index": target, "op_type": "create"},
                    }, wait_for_completion=True, refresh=True,
                        request_timeout=self.index.settings.opensearch_bulk_timeout)
                    if (result.get("failures") or result.get("timed_out")
                            or result.get("created") != expected
                            or self.client.count(index=target)["count"] != expected):
                        raise RuntimeError("复制已有知识失败，新版本未发布")
                yield staged
                staged.refresh()
                if self._count(target) == 0:
                    raise ValueError("拒绝发布空索引，已有知识保持可用")
                if self.current() != previous:
                    raise RuntimeError("活动索引已改变，新版本未发布")
                self._remember_active(previous)
                self.client.indices.update_aliases(body={"actions": [
                    {"remove": {"index": previous, "alias": self.alias, "must_exist": True}},
                    {"add": {"index": target, "alias": self.alias, "is_write_index": True}},
                ]})
            except BaseException:
                # Publication timeouts may have committed. Retain the staging
                # index too: never guess and delete a possibly active version.
                raise

    def _metadata(self, name):
        return self.client.indices.get_mapping(index=name)[name]["mappings"].get("_meta", {})

    def _remember_active(self, name):
        metadata = self._metadata(name)
        metadata["rwkvrag_version"] = {"base": self.base, "verified_active": True}
        self.client.indices.put_mapping(index=name, body={"_meta": metadata})

    def _owned_name(self, name):
        import re
        return name == self.base or re.fullmatch(re.escape(self.base) + r"-v-[0-9a-f]{32}", name) is not None

    def _count(self, name):
        result = self.client.count(index=name)
        if result.get("_shards", {}).get("failed", 0):
            raise RuntimeError("索引存在不可读分片，不能确认版本完整性")
        return int(result["count"])

    def history(self):
        """Read only; staged indexes are visible but never rollback candidates."""
        active = self.current()
        indexes = self.client.indices.get(index=self.base + "*", allow_no_indices=True)
        items = []
        for name, info in sorted(indexes.items()):
            if not self._owned_name(name):
                continue
            marker = info.get("mappings", {}).get("_meta", {}).get("rwkvrag_version", {})
            verified = marker.get("base") == self.base and marker.get("verified_active") is True
            items.append({"index": name, "active": name == active,
                          "status": "active" if name == active else "retained" if verified else "unverified",
                          "rollback_eligible": verified and name != active,
                          "created_at_ms": info.get("settings", {}).get("index", {}).get("creation_date")})
        return {"alias": self.alias, "active": active, "scope": "entire_search_index_only",
                "items": items}

    def rollback(self, target, *, expected_current):
        """Switch only to a previously active generation, with stale-view rejection.

        This intentionally does not restore MongoDB records or uploaded files.
        """
        if not self._owned_name(target):
            raise ValueError("回滚目标不属于当前索引；必须使用版本列表中的完整物理索引名")
        with self.write_lock():
            previous = self.current()
            if previous != expected_current:
                raise RuntimeError("当前版本已改变，请重新查看版本列表")
            if target == previous:
                raise ValueError("目标已经是当前版本")
            marker = self._metadata(target).get("rwkvrag_version", {})
            if marker.get("base") != self.base or marker.get("verified_active") is not True:
                raise ValueError("目标没有成功活动版本记录，不能回滚到暂存或未验证索引")
            if self._count(target) == 0:
                raise ValueError("拒绝回滚到空索引")
            self._remember_active(previous)
            self.client.indices.update_aliases(body={"actions": [
                {"remove": {"index": previous, "alias": self.alias, "must_exist": True}},
                {"add": {"index": target, "alias": self.alias, "is_write_index": True}},
            ]})
            return {"previous": previous, "active": target, "scope": "entire_search_index_only",
                    "original_files_restored": False, "database_records_restored": False}


    def source_revisions(self, version, file_id):
        from opensearchpy import helpers
        if not self._owned_name(version):
            raise ValueError("版本不属于当前配置索引")
        marker = self._metadata(version).get("rwkvrag_version", {})
        if version != self.current() and not (
                marker.get("base") == self.base and marker.get("verified_active") is True):
            raise ValueError("不能将未验证的暂存版本作为已发布来源")
        revisions = {}
        untracked = 0
        chunks = 0
        keys = ("source_revision_id", "source_sha256", "source_snapshot_uri",
                "parsed_snapshot_sha256", "parsed_snapshot_uri")
        for hit in helpers.scan(self.client, index=version, query={
                "query": {"term": {"file_id": file_id}}, "_source": ["metadata"]}):
            chunks += 1
            metadata = hit.get("_source", {}).get("metadata", {})
            if not all(metadata.get(key) for key in keys):
                untracked += 1
                continue
            item = {key: metadata[key] for key in keys}
            revisions[(item["source_revision_id"], item["parsed_snapshot_sha256"])] = item
        return {"index_version": version, "file_id": file_id, "chunks": chunks,
                "untracked_chunks": untracked, "revisions": list(revisions.values()),
                "snapshot_integrity_checked": False}
