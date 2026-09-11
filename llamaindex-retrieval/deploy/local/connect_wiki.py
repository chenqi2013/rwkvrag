"""Attach the frozen 5000-page Wiki index to the local UI without rechunking it."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from opensearchpy import OpenSearch, helpers
from pymongo import MongoClient, UpdateOne


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.settings.read_text())
    source_index = "rwkvrag-bm250820-wiki-5000-20260908"
    target_index = config["opensearch_index"]
    kb_id = "wiki-5000-20260908"
    kb_name = "FineWiki 中文百科（5000篇）"
    if source_index == target_index:
        raise ValueError("Source and UI indexes must be different")
    search = OpenSearch(config["opensearch_url"], timeout=300, max_retries=0)
    mongo = MongoClient(config["mongo_url"])
    db = mongo[config["mongo_database"]]
    try:
        source_uuid = search.indices.get_settings(index=source_index)[source_index]["settings"]["index"]["uuid"]
        manifest = [json.loads(line) for line in (args.corpus / "manifest.jsonl").read_text().splitlines()]
        pages = {str(row["page_id"]): row for row in manifest}
        assert len(pages) == len(manifest) == 5000
        texts = {}
        for page, row in pages.items():
            raw = (args.corpus / row["text_path"]).read_bytes()
            assert sha(raw) == row["text_sha256"]
            texts[page] = raw.decode("utf-8")

        def snapshot(index):
            return {hit["_id"]: hit["_source"] for hit in helpers.scan(search,
                index=index, query={"query": {"match_all": {}}}, size=500, scroll="5m")}

        sources = snapshot(source_index)
        before = snapshot(target_index)
        assert len(sources) == 45960
        documents = {}
        counts = Counter()
        for node_id, node in sources.items():
            meta = node["metadata"]
            page = str(meta["page_id"])
            row = pages[page]
            span = meta["source_span"]
            assert node["knowledge_base_id"] == meta["knowledge_base_id"] == kb_id
            assert node["file_id"] == node["document_id"] == meta["file_id"]
            assert node["text"] == texts[page][span["start"]:span["end"]]
            assert sha(node["text"].encode()) == span["sha256"]
            assert meta["source_text_sha256"] == row["text_sha256"]
            if node_id in before:
                assert before[node_id] == node, "Existing UI node differs; refusing overwrite"
            file_id = node["file_id"]
            if file_id in documents:
                assert documents[file_id]["page_id"] == row["page_id"]
            documents[file_id] = row
            counts[file_id] += 1
        assert len(documents) == 5000 and len({r["text_sha256"] for r in documents.values()}) == 5000
        existing_kb = db.knowledge_bases.find_one({"id": kb_id})
        assert existing_kb is None or existing_kb["name"] == kb_name
        assert db.knowledge_bases.find_one({"name": kb_name, "id": {"$ne": kb_id}}) is None
        assert db.files.count_documents({"knowledge_base_id": kb_id, "id": {"$nin": list(documents)}}) == 0
        for file_id, row in documents.items():
            existing = db.files.find_one({"id": file_id})
            assert existing is None or (existing["knowledge_base_id"] == kb_id and existing["sha256"] == row["text_sha256"])

        now = datetime.now(timezone.utc)
        records = []
        for file_id, row in documents.items():
            directory = Path(config["upload_dir"]) / kb_id / file_id
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "document.md"
            if path.exists():
                assert sha(path.read_bytes()) == row["text_sha256"]
            else:
                shutil.copyfile(args.corpus / row["text_path"], path)
            (directory / "PROVENANCE.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
            records.append(dict(id=file_id, knowledge_base_id=kb_id, filename=row["title"] + ".md",
                path=str(path.resolve()), content_type="text/markdown", extension=".md",
                size=row["text_bytes"], sha256=row["text_sha256"], source="finewiki-zh",
                status="ready", node_count=counts[file_id], error=None, last_job_id=None,
                created_at=now, updated_at=now, origin=row))
        print("Validated5000 source files and45960 exact source spans; copying index.", flush=True)
        result = search.reindex(body={"source": {"index": source_index},
            "dest": {"index": target_index, "op_type": "create"}, "conflicts": "proceed"},
            params={"refresh": "true", "wait_for_completion": "true"}, request_timeout=300)
        assert not result.get("failures") and not result.get("timed_out")
        after = snapshot(target_index)
        assert all(after.get(key) == value for key, value in sources.items())
        assert all(after.get(key) == value for key, value in before.items())
        assert len(after) == len(set(before) | set(sources))
        assert search.count(index=source_index)["count"] == 45960
        assert search.indices.get_settings(index=source_index)[source_index]["settings"]["index"]["uuid"] == source_uuid
        db.files.bulk_write([UpdateOne({"id": r["id"]}, {"$setOnInsert": r}, upsert=True)
                             for r in records], ordered=True)
        db.knowledge_bases.update_one({"id": kb_id}, {"$setOnInsert": dict(
            id=kb_id, name=kb_name, description="此前真实RAG评测使用的5000篇FineWiki中文文章；原文、来源及45960个原文块保持一致。Wikipedia contributors / CC BY-SA4.0。",
            created_at=now, updated_at=now)}, upsert=True)
        assert db.files.count_documents({"knowledge_base_id": kb_id}) == 5000
        report = dict(knowledge_base_id=kb_id, knowledge_base_name=kb_name, source_index=source_index,
            source_uuid=source_uuid, target_index=target_index, documents=5000, wiki_chunks=45960,
            previous_target_chunks=len(before), target_chunks=len(after), source_spans_verified=45960,
            all_copied_nodes_equal=True, previous_nodes_unchanged=True, reindex=result,
            corpus_manifest_sha256=sha((args.corpus / "manifest.jsonl").read_bytes()),
            source_content_manifest_sha256=sha(canonical({k: sha(canonical(v)) for k, v in sources.items()})))
        (args.output / "CONNECTED.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k:v for k,v in report.items() if k != "reindex"}, ensure_ascii=False), flush=True)
    finally:
        search.close()
        mongo.close()


if __name__ == "__main__":
    main()
