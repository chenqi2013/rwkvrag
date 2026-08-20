# RWKVRAG LlamaIndex Retrieval

这是一个带 Web 管理后台的知识库问答服务。LlamaIndex 负责文档解析和切片，OpenSearch 提供可分片的 BM25 全文检索，并结合标题、关键词和标签加权。MongoDB 保存知识库、文件和任务状态；RWKV 可基于检索证据生成最终答案。

## 架构

```text
管理后台 -> MongoDB 任务 -> Markdown / PDF / DOCX / FineWiki -> LlamaIndex SentenceSplitter
文本切片 -> 别名/结构提取 -> 中文规范化/Jieba 分词 -> OpenSearch BM25F 多字段索引
前端问题 -> 多查询规划 -> 标题/别名/章节/正文召回 -> 页面级加权融合 -> 证据 -> RWKV -> /v1/ask
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

标题、正文、章节、结构字段之外，新导入的数据还会建立页面别名和实体中文二元词字段。
服务升级后旧索引仍可查询，但需要重新执行导入或重新索引，才能获得别名与二元词召回能力。

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

检索查询默认由语言模型生成 3～6 组 BM25 关键词，并保留原问题作为独立检索路由；
OpenSearch 并行执行后使用加权 RRF 融合。模型查询规划超时、不可用或 JSON 协议错误时，
自动回退到确定性查询规划，不中断检索。可通过 `RWKVRAG_MODEL_QUERY_PLANNING_ENABLED`
关闭该能力；规划结果默认缓存 600 秒。

`/v1/ask` 在首次证据门禁未通过、完整列表缺少结构或原因问题缺少上下文时启用受控的主动检索：
RWKV 生成新的关键词，后端调用只读 `bm25_search` 工具补充证据。默认补搜 1 轮、
每轮最多 3 条查询；不开放 Bash、`rg`、写入或网络工具。
执行轨迹会返回在 `retrieval.active_retrieval` 中。可通过
`RWKVRAG_ACTIVE_RETRIEVAL_ENABLED` 关闭该能力。

`RWKVRAG_GENERATION_TOTAL_TIMEOUT` 默认 `45` 秒，用于限制 SSE 生成流的总等待时间；
达到时限时，服务返回已接收的内容，避免前端持续等待。

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

- OpenSearch BM25F 分别检索正文、标题、页面别名、章节和结构化标签，并按字段重要性加权。
- 页面别名来自文档 metadata、带限定词的标题以及正文开头的“又称、简称、原名”等表达。
- 查询同时执行宽松关键词、严格全词、短语和实体中文二元词召回，再使用加权 RRF 合并到页面级候选。
- 页面重排同时检查标题/别名主体、关系词覆盖、关键词距离和列表/表格等内容类型。
- 中文查询先进行繁简归一化、搜索引擎分词和停用词过滤，并对常见问法做有限关键词扩展。
- 地铁线路等编号会统一中文数字和阿拉伯数字写法，例如“一号线”和“1号线”使用同一组检索词。
- `/v1/search` 只返回证据；`/v1/ask` 返回 RWKV 基于证据生成的答案及同一批来源。
- `/v1/ask` 可在证据不足时由 RWKV 主动追加受控 BM25 查询，达到轮次上限或证据充分后停止。
- `top_k` 按文档返回，默认每篇文档只保留得分最高的一个证据块。
- “有哪些、列表、全部、站点”等多证据问题默认允许同一文档返回最多 3 个高相关切片。
- 低于最高分 55% 的候选默认不返回，避免为了凑满 `top_k` 混入弱相关资料。
- 数据中没有明确数字时，`/v1/ask` 会要求模型明确说明资料不足，不能自行补全答案。
- `eval/cases.jsonl` 保存必须持续通过的改写检索用例。
