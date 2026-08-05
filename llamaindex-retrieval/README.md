# RWKVRAG LlamaIndex Retrieval

这是一个带 Web 管理后台的知识库问答服务。LlamaIndex 负责文档解析和切片，OpenSearch 提供可分片的 BM25 全文检索，并结合标题、关键词和标签加权。MongoDB 保存知识库、文件和任务状态；RWKV 可基于检索证据生成最终答案。

## 架构

```text
管理后台 -> MongoDB 任务 -> Markdown / PDF / DOCX / FineWiki -> LlamaIndex SentenceSplitter
文本切片 -> 中文规范化/Jieba 分词 -> OpenSearch BM25（正文、标题、标签）
前端问题 -> OpenSearch BM25 + 关键词扩展 + 标题加权 -> 证据 -> RWKV -> /v1/ask
```

## 初始化

要求 Python 3.11-3.13、MongoDB 和 OpenSearch，不需要 Embedding 服务：

```bash
brew services start mongodb-community
brew services start opensearch

cd llamaindex-retrieval
uv sync --python 3.13 --extra dev
cp .env.example .env

cd web
npm install
npm run build
```

## 导入

导入会直接写入 OpenSearch BM25 索引：

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

`--recreate` 会删除并重建当前 OpenSearch 索引，请勿在并发导入时使用。

从旧 SQLite BM25 索引迁移现有切片：

```bash
uv run rwkvrag-retrieval migrate-sqlite \
  --path ../data/lexical/bm25.sqlite3 \
  --batch-size 500 \
  --recreate
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
- MongoDB 和 OpenSearch BM25 索引健康检查；
- 在线检索测试。

答案生成使用 `RWKVRAG_GENERATION_*` 配置。请在实际运行环境中设置
`RWKVRAG_GENERATION_PASSWORD`，不要将密码写入 Git。

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

需要最终答案时调用 `/v1/ask`。它先执行同一套检索，再将命中资料作为唯一证据传给
RWKV；若资料不足，模型被要求返回“根据检索到的资料，无法确定。”：

```bash
curl -sS -X POST http://127.0.0.1:8090/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"中国的首都是哪个城市","knowledge_base_id":"default","top_k":5}'
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
    "algorithm": "OpenSearch BM25",
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

- OpenSearch BM25 对正文、标题和 metadata 中的 question/tags/keywords 等字段检索；标题匹配会额外加权。
- 中文查询先进行繁简归一化、搜索引擎分词和停用词过滤，并对常见问法做有限关键词扩展。
- 地铁线路等编号会统一中文数字和阿拉伯数字写法，例如“一号线”和“1号线”使用同一组检索词。
- `/v1/search` 只返回证据；`/v1/ask` 返回 RWKV 基于证据生成的答案及同一批来源。
- `top_k` 按文档返回，默认每篇文档只保留得分最高的一个证据块。
- “有哪些、列表、全部、站点”等多证据问题默认允许同一文档返回最多 3 个高相关切片。
- 低于最高分 55% 的候选默认不返回，避免为了凑满 `top_k` 混入弱相关资料。
- 数据中没有明确数字时，`/v1/ask` 会要求模型明确说明资料不足，不能自行补全答案。
- `eval/cases.jsonl` 保存必须持续通过的改写检索用例。
