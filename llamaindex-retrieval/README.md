# RWKV 原生检索问答服务

从 `bm250820@2bbc406` 重建。`rwkv_pipeline.py` 编排检索、原文选择和作答，`model_client.py` 选择传输，`verbatim_chunking.py` 保留原文。约束见 [ARCHITECTURE_RULES.md](ARCHITECTURE_RULES.md)。

通过 `POST /v1/ask` 调用 **RWKV7 G1j 2.9B + OpenSearch BM25 + MongoDB**，不使用embedding。现有前端用于人工试用，入口为 <http://127.0.0.1:18440/admin/#/search>；API交互文档在 `/docs`。文件、知识库、历史和trace由后端接口管理。

本轮2000条数据与六组state训练、评测已完成，见[StateTune经验](../docs/statetune-experience.md)及[最新结果](eval/trace-eval-20260911/RESULTS.md)。Reader和Writer部分能力有改善，Planner出现格式退步；不同题组结果不能合成全场景准确率。试用配置按阶段使用Reader450与Writer300，Planner保持零state；这个组合是试用候选，尚未证明在完整RAG中最优。应用配置以本机配置文件为准。

实验原始输出、旧方案、模型与源码依赖按[归档说明](../docs/artifacts.md)恢复。2026-09-11测试后保留比较推理服务，未恢复旧问答模型服务；不要将历史部署说明当成当前加载状态。

## 依赖与启动

| 组件 | 作用 | 资源 |
| --- | --- | --- |
| OpenSearch | BM25 原文块检索 | CPU、内存、磁盘 |
| MongoDB | 知识库、文件、任务、问答及 trace | CPU、内存、磁盘 |
| RWKV API | 规划、逐来源阅读、作答 | 当前本地部署使用 rwkv-8222 GPU3 的2.9B |
| FastAPI | 编排、导入、管理接口 | CPU |

不需要Qdrant或embedding服务。当前WSL环境已经安装MongoDB、OpenSearch、API及独立SSH隧道，使用[本地部署说明](deploy/local/README.md)。以下为其他环境的通用配置和启动步骤。

```bash
uv sync --frozen --extra dev
cp .env.example .env
# 在本机 .env 填写 CF Access 认证、数据库地址和数据目录
uv run uvicorn llamaindex_retrieval.api:app --host 127.0.0.1 --port 8080
```

示例明确启用 `RAG_PIPELINE=rwkv` 和 `NATIVE_TRANSPORT=rwkvos_batch`，采用已测零 state、complete 闭合前缀、EOS 停止、queries/tasks 原文选择、80 总来源配置。它是可复核的开发基线，没有通过质量验收。没有 .env 时也默认走 `rwkv`，传输仍默认 `native`；`existing` 只可显式选择用于旧回归。实际服务地址仍须配置。

## 主要参数

表中参数均加 `RWKVRAG_` 前缀。软件默认启用 RWKV 管线；示例传输配置选择已测外部基线。

| 参数 | 软件默认 | 意义 |
| --- | --- | --- |
| `RAG_PIPELINE` | `rwkv` | `existing` 仅用于旧回归 |
| `NATIVE_TRANSPORT` | `native` | 外部选 `rwkvos_batch` |
| `NATIVE_BASE_URL` | 本机 18421/v1 | 外部示例 `https://api-3b.rwkvos.com/v1` |
| `NATIVE_MODEL` | `rwkv7-g1j-2.9b-20260831-ctx16384` | 服务身份校验 |
| `RWKVOS_CF_ACCESS_CLIENT_ID` / `RWKVOS_CF_ACCESS_CLIENT_SECRET` | 空 | 只在本机配置，不进入 trace |
| `RWKVOS_STATE_ID` | 省略 | 仅使用有效且模型兼容的 state |
| `RWKVOS_WRITER_STATE_ID` | 省略 | 只覆盖Writer；需要canonical evidence-first协议 |
| `RWKVOS_BINARY_READER_STATE_ID` | 省略 | 只供binary_query Reader使用，与旧编号Reader state分开 |
| `RWKVOS_READER_STATE_ID` | 省略 | 可仅覆盖 Reader；省略时继承全局 state，零对照须两者均为空 |
| `RWKVOS_READER_PROMPT_PROTOCOL` | `batch_complete_v1` | 显式 canonical 候选为 `rwkv_g1j_no_think_v1` |
| `RWKVOS_READER_INPUT_LAYOUT` | `original` | `task_last` 与数据渲染共用实现，仅可搭配 canonical Reader |
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

`generation.model_calls` 保存本项 prompt、原始输出、SHA、输入来源及阶段耗时。多项共享 batch 的完整 HTTP 字节只进入私有 `model_receipts` GridFS bucket，每批每事件一次；公开 trace 使用 `payload_scope=single_item_projection` 和 `private_receipt_id`，不会带出其他调用的内容。单项请求仍可保留自身原始字节。共享 batch 按唯一 ID 统计，计数 HTTP 单列。`retrieval.candidates`、读取来源和预算排除项均可查；sources 与 citation_map 覆盖 Writer 全部资料。片段、原文块及父级上下文携带 Unicode 坐标和 SHA。

原始输出不增删、不补引用、不静默重试。citation_audit 只检查标签，semantic_support_verified 保持 false。模型自己生成的“证据”列表不能替代真实输入来源。

超过 8 MiB 的问答/请求完整内容保存到 `rag_payloads` GridFS bucket，Mongo 集合保存摘要和引用，历史读取自动还原。原始回答不会为了入库被删节。数据库备份必须包含这两个 bucket 的 `files` 和 `chunks` 集合；它们没有公开读取 API，访问由数据库权限控制。

## StateTune数据与训练

本轮从真实trace的17个纠错种子生成2000条数据，已完成六组训练与对照。[数据管线](statetune/README.md)提供入口、格式和正式训练包；旧草稿及阶段运行记录按[附件说明](../docs/artifacts.md)恢复。

`rwkvrag-state-data`提供 `prepare / build / audit / export`，`rwkvrag-state-release`管理独立复核后的训练发布。数据导出、格式通过和loss下降均不能代替实际问答测试。

## 验证

```bash
uv run pytest -q tests/test_native_rwkv.py tests/test_rwkvos_batch.py tests/test_model_transport.py tests/test_rwkv_pipeline.py tests/test_native_integration.py tests/test_verbatim_chunking.py tests/test_ingest.py
```

发布验证见 [VALIDATION.json](../artifacts/statetune-20260911/VALIDATION.json)。完整回归仍有114项历史失败；没有删除这些测试或将其改成跳过。软件测试不证明模型答案正确。历史日志与失败集对照保存在实验附件中。
