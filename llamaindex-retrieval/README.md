# RWKV 原生检索问答服务

基于 `bm250820@2bbc406`。新链路由 `rwkv_pipeline.py` 编排，`native_rwkv.py` 负责推理传输，`verbatim_chunking.py` 保留导入原文。约束见 [ARCHITECTURE_RULES.md](ARCHITECTURE_RULES.md)。

## 依赖与启动

| 组件 | 作用 | 计算资源 |
| --- | --- | --- |
| OpenSearch | BM25 块检索 | CPU、内存、磁盘 |
| MongoDB | 知识库、文件、任务、问答及 trace | CPU、内存、磁盘 |
| RWKV 原生服务 | 规划、逐来源读取、作答 | GPU |
| FastAPI | 编排、导入、管理和 API | CPU |

不需要 Qdrant 或 embedding 模型。RWKV 服务须支持本项目验证的原生协议：`/tokenize` 返回完整 token 列表、数量和上下文上限；`/v1/completions` 支持原生 prompt、单 BOS、停止符及 usage。普通 chat 接口不能替代此契约。

```bash
uv sync --frozen --extra dev
cp .env.example .env
# 编辑实际地址、模型名和数据目录
uv run uvicorn llamaindex_retrieval.api:app --host 127.0.0.1 --port 8080
```

## 新链路参数

| 参数（前缀 `RWKVRAG_`） | 默认值 | 意义 |
| --- | --- | --- |
| `RAG_PIPELINE` | `existing`；示例为 `rwkv` | 明确选择链路 |
| `NATIVE_CONTEXT_WINDOW_TOKENS` | 16384 | 与服务上限取较小值 |
| `NATIVE_MAX_CONCURRENCY` | 32 | 进程内所有模型阶段共享 |
| `GENERATION_MAX_TOKENS` | 2048 | Writer 输出预算 |
| `NATIVE_PLANNER_MAX_TOKENS` / `NATIVE_RESOLVER_MAX_TOKENS` | 1024 / 1024 | 包括模型思考 |
| `NATIVE_RESOLVER_SOURCES` | 24 | 总读取来源预算，无每文档配额 |
| `NATIVE_PLANNER_PREFILL` / `NATIVE_RESOLVER_PREFILL` | `<think` / `<think` | 可选 `<think></think`，逐阶段实验开关 |
| `NATIVE_PLAN_PROTOCOL` | `queries_fields` | `shared_tasks` 实验使用单一子问题列表贯穿检索和阅读 |
| `NATIVE_RESOLVER_PROTOCOL` | `fields` | `task_units` 实验只选支持当前任务的原文编号，不推断字段覆盖 |
| `NATIVE_TASK_SOURCE` | `fields` | 可选 `queries`，固定将该列表同时交给 Reader 和 Writer；原计划与完整历史保留 |
| `NATIVE_CANDIDATE_ORDER` | `rrf` | 可选 `query_round_robin`，按各查询排序队列轮流取块，仍保留 RRF 分数和全部候选 |
| `NATIVE_INGEST_CHUNK_CHARACTERS` | 2400 | 正文软窗口，结构块可更长 |
| `NATIVE_INGEST_OVERLAP_CHARACTERS` | 180 | 连续正文重叠 |
| `NATIVE_TIMEOUT_SECONDS` | 180 | 获取并发槽后 tokenize 与生成合计时限 |

`NATIVE_RESOLVER_SOURCES` 是总来源上限，不是生成调用次数上限；一个超长来源可能按既定窗口分批阅读。`candidate_k` 是每条检索式的候选量。新链路不用旧 `MAX_CHUNKS_PER_DOCUMENT`、`RELATIVE_SCORE_THRESHOLD` 或语义规则门禁。超长输入显式记录 `budget_exceeded`，不会裁切。多进程部署时各 worker 有自己的并发限制，应按 GPU 总容量分配。

若要试用实测的 v5 开发条件，在 `.env` 中替换以下值（仍不是通过质量验收的生产配置）：

```dotenv
RWKVRAG_NATIVE_PLANNER_PREFILL="<think></think"
RWKVRAG_NATIVE_RESOLVER_PREFILL="<think></think"
RWKVRAG_NATIVE_PLAN_PROTOCOL=queries_fields
RWKVRAG_NATIVE_RESOLVER_PROTOCOL=task_units
RWKVRAG_NATIVE_TASK_SOURCE=queries
RWKVRAG_NATIVE_CANDIDATE_ORDER=query_round_robin
RWKVRAG_NATIVE_RESOLVER_SOURCES=24
RWKVRAG_NATIVE_TIMEOUT_SECONDS=600
```

其余参数沿用示例配置，Writer 保持开放 prefill 和 2,048 输出预算。v6 只把总来源改为 80；一轮观察中正确事实从 15/29 到 16/29，每题中位耗时从 39.96 到 98.88 秒，且一题规划输出也变化，不能将差异完全归因于来源预算。评测还固定了 4 题并发、模型/引擎及索引快照；仅修改 `.env` 不保证逐字重现。实际条件、原始调用及审阅见 [后续实验报告](eval/bm250820-followup-20260908/README.md)。

## API

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

`POST /v1/material-ask` 是固定材料 Writer 入口；传入 `question`、`history`、`materials: SourceItem[]`（1–20 份），不调用检索或 Resolver。

`POST /v1/search` 提供 BM25 检索。`top_k` 控制搜索展示条数，不限制 `/v1/ask` 返回的 Writer 引用来源。

管理页面、文件导入与任务接口复用基点实现。新切块方式应使用**新索引**，不把原文块和旧结构重写块混入同一评测索引。

## 输出与 trace

`answer` 是原始 `choice.text`，包括思考封套，不能当成已审核正文。`generation.answer_span` 给出正文在原始字符串中的 Unicode 起止位置。尚无模型文本时 `answer=""`、`raw_model_answer=null`；错误看 `generation.status`。

- `completed`：模型自然完成，不等于事实和引用都正确。
- `length`、`budget_exceeded`、`timeout` 等：保留错误与已有原始输出。
- `planner_failed`、`retrieval_failed`：链路停在相应阶段，已完成调用可检查。
- `resolver_partial_failure`：部分读取/解析失败，即便 Writer 完成也不标整链路成功。

`generation.model_calls` 保存精确 prompt、请求/响应字节、SHA、输入来源、状态、usage 和时间。`retrieval.candidates` 保存完整候选，另列入选来源、预算排除项。`sources` 与 `citation_map` 对应 Writer 全部资料。

片段 `span_start/span_end` 相对于索引块；导入 `source_span` 相对于整篇提取正文，均为 Unicode 字符坐标。父级标题/表头上下文也携带原文位置与 SHA。`citation_audit` 仅扫描正文的字面标签，不证明语义支持，`semantic_support_verified` 固定为 false。

没有应用层答案修复、静默重试或 RWKV state 缓存。

## 验证

```bash
uv run pytest -q tests/test_native_rwkv.py tests/test_rwkv_pipeline.py tests/test_native_integration.py tests/test_verbatim_chunking.py tests/test_ingest.py
```

这是实验开发链路，目前 Wiki 端到端尚未达标：最新 v6 在 8 道公开开发题上为 16/29 项事实正确，正式事实与引用完整通过 1/8，额外接受明确可读的其他引用格式时为 2/8。默认保留早期协议用于回归；上述 v5 配置是已测开发条件。初始流程见 [Wiki 实验记录](eval/native-wiki/README.md)，后续分阶段检查、完整复测与失败见 [最新实验报告](eval/bm250820-followup-20260908/README.md)。

最终全量测试为 472 通过 / 114 失败，失败 ID 与纯净基线完全一致，原有 114 项失败不标记通过。真实模型 smoke 用法和结果见 [验证记录](eval/native-smoke/README.md)。旧部署说明存档于 [基点文档](../docs/previous-python-guide.md)。
