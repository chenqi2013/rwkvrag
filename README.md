# RWKVRAG — RWKV 原生 RAG

本分支从 [chenqi2013/rwkvrag:bm250820](https://github.com/chenqi2013/rwkvrag/tree/bm250820) 的 `2bbc406125e7e030eda98b11993dc13fc4534ca4` 开始修改，分支名为 `chase/rwkv-native-rag-rebuild`。

Python 主链路使用 **OpenSearch BM25 + MongoDB + RWKV 原生推理服务**。OpenSearch 负责搜索，MongoDB 保存知识库、导入任务和问答记录；两者使用 CPU、内存和磁盘。RWKV 推理使用 GPU。新链路不需要 Qdrant 或 embedding 服务。

```mermaid
flowchart LR
  D[Wiki / 上传文档] --> C[逐字切块与来源记录]
  C --> B[OpenSearch BM25]
  Q[最新问题与历史] --> P[RWKV 规划]
  P --> B
  B --> R[RWKV 并发读取各来源]
  R --> E[原文选择与来源校验]
  E --> W[RWKV 汇总作答]
  W --> A[原始输出与完整 trace]
  A --> M[MongoDB 问答记录]
```

## 当前实现

- 新链路独立于旧规则服务，复用 `bm250820` 的 OpenSearch、管理接口和导入能力。
- 原生 `User✿/Bot✿<think` 模板及 `/tokenize`、`/v1/completions`；输入加输出按真实 tokenizer 检查。所有模型阶段共享并发限制，默认 32。
- 不限制每文档两块，不按文档去重。BM25 返回完整块，支持 RRF 排序或按查询轮转取候选；总候选和总读取来源预算独立配置，未读候选在 trace 中可见。
- Resolver 并发读取原文，支持按字段绑定证据或按当前任务选择原文编号。可将同一份模型生成的查询列表传给检索、阅读和作答，完整历史保留；代码只解析和验证编号，Writer 仅接收入选原文及其父级上下文。
- 来源、整篇文本 SHA、块位置、选中片段位置与 SHA 可追溯。FineWiki 导入修复跨批次 ID 碰撞，保留页 ID、版本和语言。
- RWKV 输出原样返回，包括思考封套和长度耗尽时的部分输出。`answer_span` 标出正文范围；不会补引用、删句、改答案或静默重试。
- 问答支持历史；全部 Writer 来源返回，引用表不会因展示 `top_k` 被截断。历史摘要区分运行完成、部分失败和失败，不再把空答案超时误记为已回答。

## 使用

详见 [Python 服务说明](llamaindex-retrieval/README.md)。填写示例配置中的实际服务地址后启动：

```bash
cd llamaindex-retrieval
uv sync --frozen --extra dev
cp .env.example .env
uv run uvicorn llamaindex_retrieval.api:app --host 127.0.0.1 --port 8080
```

示例配置启用 `RWKVRAG_RAG_PIPELINE=rwkv`。没有 `.env` 时保留 `existing` 兼容模式，便于旧接口回归；部署时请明确选择。

## 验证与边界

先验证固定材料 Writer，再检查 Resolver、检索和 Wiki 端到端。开发 smoke 的完成率、事实正确率、引用支持率分别报告，不是生产准确率或新盲测成绩。

纯净 `bm250820` 为 **176 通过 / 114 失败**；最终本地回归为 **472 通过 / 114 失败**，失败测试 ID 与基点完全相同，没有新增失败，也没有将原有失败标记通过。前端 19 项测试及构建通过。完整原日志、失败集合和源码 SHA 见 [后续实验归档](llamaindex-retrieval/eval/bm250820-followup-20260908/README.md)。

在 `rwkv-8222` 上，使用 RWKV7 13.3B、16K 上下文和实际峰值 32 并发，对固定 **5,000 篇 Wiki、45,960 个逐字块**运行完整链路。以下是相同的 8 道已公开开发题、29 项必要事实，由 Codex 逐项检查正文和实际引用，并非人工盲测：

| 条件 | 正确事实 | 事实完整且正式引用通过 | 每题中位耗时 |
| --- | ---: | ---: | ---: |
| v2：规划闭合 prefill，字段读取 | 2/29 | 0/8 | 77.84 秒 |
| v3：闭合读取与任务原文选择 | 12/29 | 0/8 | 36.03 秒 |
| v4：检索、阅读、作答共用 queries | 13/29 | 0/8 | 39.04 秒 |
| v5：按查询轮转，总读取 24 来源 | 15/29 | 0/8 | 39.96 秒 |
| v6：总读取增至 80 来源 | 16/29 | 1/8 | 98.88 秒 |

v6 若额外接受明确可读但不符合正式语法的引用格式，完整通过为 2/8；该口径单列，不替换正式分数。唯一正式通过的列车题同时出现模型规划和检索候选变化，不能将通过隔离归因于增加来源。两题耗尽 2,048 输出预算；输入均未超 16K。来源数量增加没有解决引用错配、已提供证据仍遗漏、部分题目选不到证据的问题。

当前仍是**质量尚未达标的实验开发分支**。实际测试覆盖完整检索到回答流程，尚未完成新 API 的生产部署验收。固定材料 Writer 基线为 27/28 个事实正确；更换 prefill、资料标签或增加引用提醒的候选均未满足晋级条件，Writer 保持原方案。默认配置保留早期协议用于回归，表中各实验参数及可复核原始 trace 见 [后续实验报告](llamaindex-retrieval/eval/bm250820-followup-20260908/README.md)；试用 v5 的配置见 [服务说明](llamaindex-retrieval/README.md)。

字段规划、证据选择和最终引用分别评估；标签存在或 SHA 一致不代表语义正确。应用层跨文档 RWKV state 缓存尚未实现，原生服务的前缀缓存不等于该能力。

旧分支实验说明，增加输出长度、取消分数阈值、加强提示词均不能直接推出质量提升。本分支保留开发测试与失败输出，后续改动需重新测量。

## 其他代码

早期 Go 原型说明移至 [Go 文档](docs/go-prototype.md)。`bm250820` 原有 Python 文档存档于 [旧服务说明](docs/previous-python-guide.md)。新的 RWKV 专属开发入口是上述 Python 链路。
