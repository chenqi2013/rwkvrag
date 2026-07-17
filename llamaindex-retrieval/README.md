# RWKVRAG LlamaIndex Retrieval

这是一个带 Web 管理后台的证据检索服务，使用 LlamaIndex 编排 Qwen3-Embedding-8B、Qdrant dense/sparse 混合检索和可选 reranker。MongoDB 保存知识库、文件和任务状态；服务只返回资料片段和来源，不生成最终答案。

## 架构

```text
管理后台 -> MongoDB 任务 -> Markdown / PDF / DOCX / FineWiki -> LlamaIndex IngestionPipeline
LlamaIndex -> Qwen3 4096 维 + 中文 sparse -> Qdrant collection alias
前端问题 -> LlamaIndex Hybrid Retriever -> 可选 BGE reranker -> /v1/search
```

## 初始化

要求 Python 3.11-3.13、MongoDB、Qdrant 和一个 OpenAI 兼容的 Embedding 服务。本机开发使用 Ollama：

```bash
brew services start ollama
brew services start mongodb-community
ollama pull qwen3-embedding:8b

curl -sS -X POST http://127.0.0.1:11434/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-embedding:8b","input":["测试中文向量"]}' \
  | jq '.data[0].embedding | length'

cd llamaindex-retrieval
uv sync --python 3.13 --extra dev
cp .env.example .env

cd web
npm install
npm run build
```

启用 `BAAI/bge-reranker-v2-m3` 时安装额外依赖：

```bash
uv sync --python 3.13 --extra dev --extra rerank
```

## 导入

在线服务默认通过 `rwkvrag-knowledge-current` alias 访问当前 collection。批量入库时必须指定新的物理 collection，例如：

```bash
RWKVRAG_QDRANT_COLLECTION=rwkvrag-qwen3-finewiki-v2 \
uv run rwkvrag-retrieval ingest-finewiki \
  --path ../data/deploy-demo/finewiki-sample/train-00000-of-00026.parquet \
  --limit 100 \
  --batch-size 16 \
  --recreate
```

验证完成后在 GPU 服务器导入整个 `train-00000-of-00026.parquet`：

```bash
RWKVRAG_QDRANT_COLLECTION=rwkvrag-qwen3-finewiki-v2 \
uv run rwkvrag-retrieval ingest-finewiki \
  --path ../data/deploy-demo/finewiki-sample/train-00000-of-00026.parquet \
  --batch-size 16 \
  --recreate
```

在线 alias 禁止执行 `--recreate`。迁移到 GPU 服务器时，将 `RWKVRAG_EMBEDDING_BASE_URL` 改为该服务器的 OpenAI 兼容 `/v1` 地址，并保持模型名和 4096 维配置一致。新 collection 验证完成后，在管理后台把 `rwkvrag-knowledge-current` alias 原子切换到新 collection。

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
- 知识库 CRUD 和检索隔离；
- FineWiki 异步导入、进度和错误记录；
- 切片正文与 metadata 查看；
- Qdrant collection、snapshot 备份/恢复和 alias 切换；
- MongoDB、Qdrant、Qwen3 Embedding 健康检查；
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
    "embedding_model": "qwen3-embedding:8b",
    "embedding_dimensions": 4096,
    "mode": "hybrid",
    "reranked": false,
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

- Qwen3-Embedding-8B 处理“光棍男性 / 未婚男性 / 婚姻挤压”这类中文语义改写。
- 中文单字和双字 sparse 向量保留专有名词、数字和精确关键词召回。
- Hybrid 检索默认使用 80% dense 语义分数和 20% sparse 关键词分数，避免纯关键词频次压过直接语义证据。
- reranker 只对候选证据排序，不生成答案。
- `top_k` 按文档返回，默认每篇文档只保留得分最高的一个证据块。
- 低于最高分 55% 的候选默认不返回，避免为了凑满 `top_k` 混入弱相关资料。
- 数据中没有明确数字时，API 返回最相关资料；前端生成模型负责判断资料是否足够。
- `eval/cases.jsonl` 保存必须持续通过的改写检索用例。
