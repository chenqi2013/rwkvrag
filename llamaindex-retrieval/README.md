# RWKVRAG LlamaIndex Retrieval

这是一个带 Web 管理后台的经典证据检索服务。LlamaIndex 负责文档解析和切片，SQLite FTS5 提供 BM25 全文检索，并结合标题、关键词和标签加权。MongoDB 保存知识库、文件和任务状态；服务只返回资料片段和来源，不生成最终答案。

## 架构

```text
管理后台 -> MongoDB 任务 -> Markdown / PDF / DOCX / FineWiki -> LlamaIndex SentenceSplitter
文本切片 -> 中文规范化/分词 -> SQLite FTS5 BM25（正文、标题、标签）
前端问题 -> BM25 + 关键词扩展 + 标题加权 -> /v1/search
```

## 初始化

要求 Python 3.11-3.13、MongoDB。Qdrant 仅用于旧 collection 的管理页面和一次性只读迁移，不参与当前检索，也不需要 Embedding 服务：

```bash
brew services start mongodb-community

cd llamaindex-retrieval
uv sync --python 3.13 --extra dev
cp .env.example .env

cd web
npm install
npm run build
```

## 导入

导入会直接建立本地 SQLite BM25 索引，不创建或更新 Qdrant collection：

```bash
uv run rwkvrag-retrieval ingest-finewiki \
  --path ../data/deploy-demo/finewiki-sample/train-00000-of-00026.parquet \
  --limit 100 \
  --batch-size 16 \
  --recreate
```

导入完整文件：

```bash
uv run rwkvrag-retrieval ingest-finewiki \
  --path ../data/deploy-demo/finewiki-sample/train-00000-of-00026.parquet \
  --batch-size 16 \
  --recreate
```

`--recreate` 会清空并重建本地 BM25 索引，请勿在并发导入时使用。

如果需要从旧本地 Qdrant collection 迁移已有切片，可执行只读迁移命令。该命令只调用 Qdrant scroll，不会写入、删除 collection 或修改 alias：

```bash
uv run rwkvrag-retrieval rebuild-lexical \
  --collection rwkvrag-knowledge-current \
  --batch-size 512
```

导入本地 Markdown 文件或目录：

```bash
uv run rwkvrag-retrieval ingest-markdown \
  --path ../data/deploy-demo/docs \
  --source local-markdown \
  --batch-size 16
```

也可以通过本地 API 导入：

```bash
curl -sS -X POST http://127.0.0.1:8090/v1/import/markdown \
  -H 'Content-Type: application/json' \
  -d '{"path":"/Volumes/mark/rwkvrag/data/deploy-demo/docs","source":"local-markdown"}'
```

## API

```bash
uv run rwkvrag-retrieval serve --host 127.0.0.1 --port 8090
```

管理后台：

```text
http://127.0.0.1:8090/admin/
```

后台支持：

- Markdown、MDX、文本型 PDF、DOCX 上传、删除和重新索引；
- 结构化问答 Markdown 自动按问题拆分，检索时返回完整答案并跳过待回答项；
- 知识库 CRUD 和检索隔离；
- FineWiki 异步导入、进度和错误记录；
- FineWiki 服务器端 Parquet 文件/目录浏览选择；
- 切片正文与 metadata 查看；
- MongoDB 和 BM25 索引健康检查；
- 在线检索测试。

扫描版 PDF 需要先通过 OCR 生成文字层。旧 `.doc` 文件应先转换为 `.docx`。

macOS 常驻运行：

```bash
uv venv ~/.local/share/rwkvrag-llamaindex/.venv --python 3.13
uv pip install \
  --python ~/.local/share/rwkvrag-llamaindex/.venv/bin/python \
  .
cp deploy/com.rwkvrag.llamaindex-retrieval.plist \
  ~/Library/LaunchAgents/com.rwkvrag.llamaindex-retrieval.plist
launchctl bootstrap gui/$(id -u) \
  ~/Library/LaunchAgents/com.rwkvrag.llamaindex-retrieval.plist
```

```bash
curl -sS -X POST http://127.0.0.1:8090/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"question":"中国人口最多的省份是哪个省","knowledge_base_id":"default","top_k":5}'
```

返回结构：

```json
{
  "results": [
    {
      "id": "node-id",
      "document_id": "document-id",
      "source": "finewiki-zh",
      "title": "中国男女比例失调",
      "uri": "https://zh.wikipedia.org/wiki/中国男女比例失调",
      "score": 0.82,
      "snippet": "...",
      "metadata": {}
    }
  ],
  "retrieval": {
    "algorithm": "BM25",
    "mode": "bm25+keyword",
    "keyword_fields": ["body", "title", "tags"],
    "candidate_k": 40,
    "top_k": 5
  }
}
```

运行检索改写回归集：

```bash
uv run rwkvrag-retrieval eval --url http://127.0.0.1:8090 --top-k 5
```

## 质量原则

- BM25 对正文、标题和 metadata 中的 question/tags/keywords 等字段检索；标题匹配会额外加权。
- 中文查询先进行繁简归一化、搜索引擎分词和停用词过滤，并对常见问法做有限关键词扩展。
- 检索服务只返回证据，不生成答案；前端生成模型可以基于返回的证据组织最终回答。
- `top_k` 按文档返回，默认每篇文档只保留得分最高的一个证据块。
- 低于最高分 55% 的候选默认不返回，避免为了凑满 `top_k` 混入弱相关资料。
- 数据中没有明确数字时，API 返回最相关资料；前端生成模型负责判断资料是否足够。
- `eval/cases.jsonl` 保存必须持续通过的改写检索用例。
