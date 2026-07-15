# rwkvrag

`rwkvrag` 是一个 Go 写的轻量 RAG 服务，支持：

- 导入 [HuggingFaceFW/finewiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) 中文子集 `zh` 的 parquet 文件。
- 导入公司内部 Markdown / MDX 文档。
- 对外提供 HTTP API，用户问答时调用 `/v1/ask`。
- 默认使用本地 hash embedding，方便离线试跑；生产环境可切到 OpenAI 兼容的 embedding 和 chat/RWKV 服务。

## 快速开始

```bash
go test ./...
go run ./cmd/rwkvrag serve --addr :8080
```

健康检查：

```bash
curl http://localhost:8080/health
```

## 导入公司 Markdown

```bash
go run ./cmd/rwkvrag import-markdown \
  --path ./company-docs \
  --source company
```

也可以通过 API 导入：

```bash
curl -X POST http://localhost:8080/v1/import/markdown \
  -H 'Content-Type: application/json' \
  -d '{"path":"./company-docs","source":"company"}'
```

## 导入 FineWiki 中文数据集

FineWiki 的中文配置名是 `zh`，文件位于 Hugging Face 仓库的 `data/zhwiki/*`。
中文子集约 5GB，当前由 5 个 parquet 文件组成，首次测试建议只下载 1 个文件或限制导入条数。

下载中文 parquet：

```bash
go run ./cmd/rwkvrag download-finewiki \
  --config zh \
  --out data/finewiki/zh \
  --max-files 1
```

导入中文 parquet：

```bash
go run ./cmd/rwkvrag import-finewiki \
  --path data/finewiki/zh \
  --language zh \
  --limit 10000
```

也可以通过 API 导入：

```bash
curl -X POST http://localhost:8080/v1/import/finewiki \
  -H 'Content-Type: application/json' \
  -d '{"path":"data/finewiki/zh","language":"zh","limit":10000}'
```

## 问答 API

```bash
curl -X POST http://localhost:8080/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"RWKV 是什么？","top_k":5}'
```

响应格式：

```json
{
  "answer": "回答内容",
  "sources": [
    {
      "id": "chunk id",
      "document_id": "document id",
      "source": "company",
      "title": "文档标题",
      "uri": "文件路径或 URL",
      "score": 0.82,
      "snippet": "命中的片段"
    }
  ]
}
```

仅检索不生成：

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"question":"报销流程","top_k":3}'
```

## 接入 embedding 和 LLM

默认配置不依赖外部服务：

- `RWKVRAG_EMBEDDING_PROVIDER=hash`
- `RWKVRAG_LLM_PROVIDER=extractive`

这种模式可以导入、检索和返回相关片段摘要，但不会生成自然语言答案。生产建议接 OpenAI 兼容服务：

```bash
export RWKVRAG_EMBEDDING_PROVIDER=openai
export RWKVRAG_EMBEDDING_BASE_URL=http://localhost:8000/v1
export RWKVRAG_EMBEDDING_API_KEY=your-key
export RWKVRAG_EMBEDDING_MODEL=your-embedding-model

export RWKVRAG_LLM_PROVIDER=openai
export RWKVRAG_LLM_BASE_URL=http://localhost:8000/v1
export RWKVRAG_LLM_API_KEY=your-key
export RWKVRAG_LLM_MODEL=your-chat-model

go run ./cmd/rwkvrag serve --addr :8080
```

如果公司 RWKV 服务兼容 OpenAI 的 `/v1/embeddings` 和 `/v1/chat/completions`，直接把 `*_BASE_URL` 指向该服务即可。

## 常用配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `RWKVRAG_ADDR` | `:8080` | HTTP 监听地址 |
| `RWKVRAG_STORE_PROVIDER` | `jsonl` | 存储后端；生产/全量数据使用 `qdrant` |
| `RWKVRAG_DB` | `data/index.jsonl` | `jsonl` 后端的本地索引文件 |
| `RWKVRAG_QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant HTTP 地址 |
| `RWKVRAG_QDRANT_COLLECTION` | `rwkvrag` | Qdrant collection 名称 |
| `RWKVRAG_CHUNK_SIZE` | `1200` | 文档切块大小，按 rune 计 |
| `RWKVRAG_CHUNK_OVERLAP` | `180` | 文档切块重叠 |
| `RWKVRAG_CORS_ORIGIN` | `*` | CORS origin |

## API 列表

- `GET /health`
- `GET /v1/stats`
- `POST /v1/search`
- `POST /v1/ask`
- `POST /v1/import/markdown`
- `POST /v1/import/finewiki`

## 说明

默认 `jsonl` 后端只适合小数据快速验证。全量 FineWiki 使用 Qdrant 的 dense HNSW + sparse IDF 混合索引，并通过 RRF 和本地词项分数重排；设置 `RWKVRAG_STORE_PROVIDER=qdrant` 后，导入、检索和问答 API 会自动使用配置的 collection。相同文档重复导入会覆盖相同 point，不会产生重复内容。
