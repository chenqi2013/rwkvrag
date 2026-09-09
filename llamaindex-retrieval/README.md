# RWKV 原生检索问答服务

从 `bm250820@2bbc406` 重建。`rwkv_pipeline.py` 编排检索、原文选择和作答，`model_client.py` 选择原生或外部 batch 传输，`verbatim_chunking.py` 保留原文。约束见 [ARCHITECTURE_RULES.md](ARCHITECTURE_RULES.md)。

当前主模型 **RWKV7 G1j 2.9B**，外部 API 适配和完整 Wiki 开发基线已完成，但质量尚未达标：8 题 / 29 事实为 0 正确、25 遗漏、4 错误，正式完整 0/8。固定材料 Writer 为 19/28。详见 [2.9B 评测](eval/rwkvos-29b-20260909/README.md) 和 [接口/state 合同](../docs/rwkvos-api.md)。此前 13.3B 是独立历史条件。

## 依赖与启动

| 组件 | 作用 | 资源 |
| --- | --- | --- |
| OpenSearch | BM25 原文块检索 | CPU、内存、磁盘 |
| MongoDB | 知识库、文件、任务、问答及 trace | CPU、内存、磁盘 |
| RWKV API | 规划、逐来源阅读、作答 | 当前外部 2.9B，备用本机 GPU |
| FastAPI | 编排、导入、管理接口 | CPU |

不需要 Qdrant 或 embedding 服务。以下启动完整应用还需自行配置 OpenSearch 和 MongoDB；本轮本机基线只验证了索引与检索链路，并非管理服务部署验收。本机已跑通官方 2.9B 短输入推理及 state 梯度检查，尚未部署为备用 API，也没有自动故障切换。本机资源不足时可按最新授权使用 `rwkv-8222` 的 GPU2，仍使用 2.9B；本轮尚未启用服务器。最新质量与兼容性结果见 [后续验证](eval/rwkvos-reader-followup-20260909/README.md)。

```bash
uv sync --frozen --extra dev
cp .env.example .env
# 在本机 .env 填写 CF Access 认证、数据库地址和数据目录
uv run uvicorn llamaindex_retrieval.api:app --host 127.0.0.1 --port 8080
```

示例明确启用 `RAG_PIPELINE=rwkv` 和 `NATIVE_TRANSPORT=rwkvos_batch`，采用已测零 state、complete 闭合前缀、EOS 停止、queries/tasks 原文选择、80 总来源配置。它是可复核的开发基线，没有通过质量验收。没有 .env 时保留 `existing` 及 `native` 兼容默认，不能用省略配置代替选择部署方案。

## 主要参数

表中参数均加 `RWKVRAG_` 前缀。软件默认保留回归兼容；示例配置选择本轮外部基线。

| 参数 | 软件默认 | 意义 |
| --- | --- | --- |
| `NATIVE_TRANSPORT` | `native` | 外部选 `rwkvos_batch` |
| `NATIVE_BASE_URL` | 本机 18421/v1 | 外部示例 `https://api-3b.rwkvos.com/v1` |
| `NATIVE_MODEL` | `rwkv7-g1j-2.9b-20260831-ctx16384` | 服务身份校验 |
| `RWKVOS_CF_ACCESS_CLIENT_ID` / `RWKVOS_CF_ACCESS_CLIENT_SECRET` | 空 | 只在本机配置，不进入 trace |
| `RWKVOS_STATE_ID` | 省略 | 仅使用有效且模型兼容的 state |
| `RWKVOS_BATCH_SIZE` / `RWKVOS_BATCH_WAIT_MS` | 8 / 5 | 同参数调用合批 |
| `NATIVE_MAX_CONCURRENCY` | 32 | 进程内所有阶段共享项数 |
| `RWKVOS_PREFILL_MODE` | `complete` | `continuation` 省略前缀末尾 >，未晋级候选 |
| `RWKVOS_STOP_TOKENS` | 省略 | 示例 `[0]`；与空数组不同 |
| `RWKVOS_COUNT_INPUT_TOKENS` | false | 可选逐项完整 prompt 服务计数 |
| `RWKVOS_INPUT_TOKEN_LIMIT` | 无 | 显式应用输入上限，须同时开启计数 |
| `NATIVE_CONTEXT_WINDOW_TOKENS` | 16384 | native 与服务限制取小；外部不将它当服务硬上限 |
| `NATIVE_PLANNER_PREFILL` / `NATIVE_RESOLVER_PREFILL` / `NATIVE_WRITER_PREFILL` | `<think` | 示例各为 `<think></think` |
| `NATIVE_PLAN_PROTOCOL` | `queries_fields` | `shared_tasks` 为独立实验，未晋级 |
| `NATIVE_RESOLVER_PROTOCOL` | `fields` | 示例 `task_units` 只选择原文编号 |
| `NATIVE_TASK_SOURCE` | `fields` | 示例 `queries` 同时交给 Reader 和 Writer |
| `NATIVE_CANDIDATE_ORDER` | `rrf` | 示例按查询轮转，保留 RRF 分数和所有候选 |
| `NATIVE_RESOLVER_SOURCES` | 24 | 示例 80，总来源预算，无每文档配额 |
| `NATIVE_PLANNER_MAX_TOKENS` / `NATIVE_RESOLVER_MAX_TOKENS` | 1024 / 1024 | 阶段输出预算 |
| `GENERATION_MAX_TOKENS` | 2048 | Writer 输出预算 |
| `NATIVE_INGEST_CHUNK_CHARACTERS` / `NATIVE_INGEST_OVERLAP_CHARACTERS` | 2400 / 180 | 逐字软窗口与重叠 |
| `NATIVE_TIMEOUT_SECONDS` | 180 | 示例 600；计数开启时涵盖排队、计数和生成 |

`candidate_k` 是每条检索式的候选量，总来源上限不是调用次数上限。超长原文可分多次读取。每次 Reader 的约 6,000 原文字符没有包括历史、任务、父级上下文和模板，不能当总 token 上限。外部计数默认关闭；启用后计数失败或超过显式应用上限均显式返回，不裁切输入。没有动态 state 续读，详细机制与 8K/16K/32K 结果见 [超长上下文](../docs/long-context.md)。

计数开启时，总期限从调用入口开始，包含等槽和批次排队；失败收据的有界落盘收尾单独计时。计数关闭时保留原外部批次 HTTP 时限。

新链路不用旧 `MAX_CHUNKS_PER_DOCUMENT`、`RELATIVE_SCORE_THRESHOLD` 或语义规则门禁。多 worker 各有独立并发限制，部署时需分配总容量。

## API 与 trace

`POST /v1/ask` 从完整知识库检索后作答：

```json
{
  "question": "更正，现在只说明 A 的发布时间和兼容系统。",
  "history": [
    {"role": "user", "content": "先比较 A 和 B。"},
    {"role": "assistant", "content": "请说明要比较的字段。"}
  ],
  "knowledge_base_id": "wiki-5000-20260908",
  "candidate_k": 80
}
```

`POST /v1/material-ask` 只运行固定材料 Writer，输入 question、history、materials（1–20 份 SourceItem）。`POST /v1/search` 只运行 BM25；展示 top_k 不限制 Writer 返回的引用来源。管理、导入和任务接口复用基点实现，新切块应使用新索引。

`answer` 保留原始模型文本。`generation.answer_span` 标出正文的 Unicode 起止位置；不能把它当事实审核。没有模型文本时 answer 为空、raw_model_answer 为 null。

- `completed` 表示传输返回完成；外部 `termination_verified=false`，其 stop 不能证明自然结束。
- `length`、`budget_exceeded`、`token_count_failed`、`timeout` 等保留错误与实际已有输出。
- `planner_failed`、`retrieval_failed` 明确指出停止阶段。
- `resolver_partial_failure` 表示部分读取或解析失败，即便 Writer 返回也不标整链路成功。

`generation.model_calls` 保存 prompt、字节、SHA、输入来源及阶段耗时；共享 batch 按唯一 ID 统计，计数 HTTP 单列。`retrieval.candidates`、读取来源和预算排除项均可查；sources 与 citation_map 覆盖 Writer 全部资料。片段、原文块及父级上下文携带 Unicode 坐标和 SHA。

原始输出不增删、不补引用、不静默重试。citation_audit 只检查标签，semantic_support_verified 保持 false。模型自己生成的“证据”列表不能替代真实输入来源。

## 验证

```bash
uv run pytest -q tests/test_native_rwkv.py tests/test_rwkvos_batch.py tests/test_model_transport.py tests/test_rwkv_pipeline.py tests/test_native_integration.py tests/test_verbatim_chunking.py tests/test_ingest.py
```

本轮完整回归与原始日志见 [2.9B 归档](eval/rwkvos-29b-20260909/README.md)，基点 114 项失败单列比较。前端测试与构建也单独验证。真实模型开发题、源码快照、失败输出和引用语义审阅全部保留；软件测试不证明模型答案正确。旧 13.3B 记录见 [历史报告](eval/bm250820-followup-20260908/README.md)，基点部署说明见 [存档](../docs/previous-python-guide.md)。
