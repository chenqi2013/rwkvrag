# 按问题并发阅读原文

`question_documents_v1` 是材料问答的显式候选策略，默认关闭。它让 RWKV 按同一个原问题并发阅读每份传入材料，再将模型选中的完整原文交给独立 Writer。它不创建子问题，也不经过摘要归并。该策略不调用检索索引、关键词评分或向量服务。

本接口是已测研究策略的应用接入；应用接入后的完整模型质量仍需验证，不能将研究结果或工程单测当作生产准确率承诺。

## 配置与入口

在兼容原生 JSON Schema 解码的 RWKV 服务上显式设置：

```dotenv
RWKVRAG_RAG_PIPELINE=rwkv
RWKVRAG_GENERATION_OUTPUT_MODE=immutable
RWKVRAG_NATIVE_TRANSPORT=native
RWKVRAG_NATIVE_COMPLETION_PROTOCOL=g1j_plain
RWKVRAG_NATIVE_REQUIRE_MODEL_IDENTITY=true
RWKVRAG_NATIVE_WRITER_PIPELINE=single
RWKVRAG_NATIVE_PLANNER_PREFILL=<think></think
RWKVRAG_NATIVE_RESOLVER_PREFILL=<think></think
RWKVRAG_NATIVE_WRITER_PREFILL=<think></think
RWKVRAG_NATIVE_MATERIAL_READING=question_documents_v1
RWKVRAG_NATIVE_DOCUMENT_READ_CONCURRENCY=32
```

另行配置 `NATIVE_BASE_URL`、`NATIVE_MODEL`、认证、模型实际上下文容量及需要的 Reader/Writer [State 引用](NATIVE_STATE_ROUTING.md)。该选项不会加载或分发模型与 State，也不会提高推理服务的批处理容量。不能与统一检索候选、anchored任务合同、其他引用失败策略或原文打包策略同时启用。

调用现有 `POST /v1/material-ask`，传入 `question` 与 `materials`；每条材料沿用 `SourceItem` 的 `id`、`document_id`、`source`、`title`、`score`、`snippet` 等字段。`snippet` 在此代表调用方提供的完整材料正文，应用不会回查或补全被调用方截掉的内容。HTTP 接口沿用最多20条材料的请求边界；Python 材料接口会排队处理全部传入材料，不按并发数裁剪。

本候选只支持单轮问题。非空 `history` 返回 `unsupported_history`，不静默忽略历史；普通 `/v1/ask` 的取材策略不由本选项切换。关闭候选时仍使用原材料直答接口。

## 执行与原文边界

1. 每条原文连同原问题单独送入 Reader。并发配置范围1–32，实际同时执行数还受共享 `NATIVE_MAX_CONCURRENCY` 限制。
2. Reader 通过 `findings[].source_id` 选择该原文，空数组表示不选择。代码只校验结构及来源身份；失败、长度退出和越界ID均不能选入原文。
3. 所有 Reader 结束后，按输入顺序汇集模型选中的原文，完整交给一次独立 Writer。阅读摘要不进入 Writer，不补选失败节点的原文。空证据默认仍调用 Writer；显式 `NATIVE_EMPTY_EVIDENCE_POLICY=fail` 时返回失败或无证据状态。

Reader 输出预算768 token，Writer 2048 token；temperature=0、seed=11。每次阅读最多6条findings，终答最多6个引用ID，这些均不是文档总数或子问题数量上限。原文超出模型上下文时由传输层记录失败，不裁剪来凑成功。

该候选采用不复用HTTP连接的传输配置，关闭自动重试；同一客户端的其他调用也使用该连接策略。请求总调用数和时限仍可用 `NATIVE_REQUEST_MAX_CALLS`、`NATIVE_REQUEST_TIMEOUT_SECONDS` 约束。若已知调用预算不足以覆盖全部材料加一次终答，整次请求在调用模型前失败。取消会等待所有本地在途任务退出，不推断远端已停止执行。

## 响应与审计

`answer` 是完整、未修改的 Writer raw JSON；`generation.parsed_answer` 仅在结构与ID准入后提供解析视图。失败 raw 仍保留，不修JSON、不补引用、不重写正文。`sources` 是 Writer 实际收到的模型所选原文。

`generation.model_calls` 记录精确消息、原始输出、哈希、阶段、来源ID和耗时。Reader 有失败而终答有效时状态为 `partial`；终答结构或引用ID非法时为 `invalid_output`。格式通过及来源ID合法都不能证明事实支持，`semantic_support_verified` 始终为 false，最终答案仍须独立评审。
