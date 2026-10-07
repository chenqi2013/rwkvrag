# RWKV 原生检索问答服务

> 应用安装与接口参考。模型、State和实际启用功能以部署配置为准；本地研究记录不随应用仓库发布。

基于 OpenSearch BM25、MongoDB 和 RWKV API。模型端点与兼容 State 由配置指定；不同模型与State需要分别验证。约束见 [ARCHITECTURE_RULES.md](ARCHITECTURE_RULES.md)。

POST /v1/ask 执行检索与作答；文件、知识库、历史和 trace 由管理接口维护。新代码中的单项证据接口见[证据契约](../docs/EVIDENCE.md)，默认需要显式配置才能启用。通用安装配置不等同于本机正在运行的配置。

## 依赖与启动

| 组件 | 作用 | 资源 |
| --- | --- | --- |
| OpenSearch | BM25 原文块检索 | CPU、内存、磁盘 |
| MongoDB | 知识库、文件、任务、问答及 trace | CPU、内存、磁盘 |
| RWKV API | 规划、逐来源阅读、作答 | 可独立部署的兼容模型端点 |
| FastAPI | 编排、导入、管理接口 | CPU |

不需要Qdrant或embedding服务。本机服务管理参考[本地部署说明](deploy/local/README.md)。以下为其他环境的通用配置和启动步骤。

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
| `NATIVE_STATE_ROUTING` | 未启用 | native/g1j_plain 的显式模型＋角色→初始State引用映射，见[接口与边界](NATIVE_STATE_ROUTING.md) |
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
| `NATIVE_TIMEOUT_SECONDS` | 180 | 单次模型操作时限；native 从获取并发槽位后开始，外部计数模式包含排队 |
| `NATIVE_REQUEST_MAX_CALLS` | 未启用 | 单次问答的逻辑模型操作总上限，包括路由、格式修复、Reader、Writer、审查及 token 预检 |
| `NATIVE_REQUEST_TIMEOUT_SECONDS` | 未启用 | 单次问答总时限，包含检索、模型排队及全部阶段 |
| `NATIVE_HISTORY_PROTOCOL` | `raw` | v1 保留兼容；完整角色历史的 `current-question-v2` 为未通过语义验收的实验协议 |
| `NATIVE_TASK_CONTRACT` | `legacy` | `anchored_v1` 是任务与检索提示分离的未验收架构候选，见[G1K架构](G1K_ARCHITECTURE.md)；不自动迁移配置 |
| `NATIVE_EMPTY_EVIDENCE_POLICY` | `write` | `fail` 在无已选证据时停止 Writer，不编造拒答 |
| `NATIVE_ANSWER_QUALITY_POLICY` | `disabled` | `fail_citation` 将缺失、非法或越界引用标为失败，不作事实判断 |

`candidate_k` 是每条检索式的候选量，总来源上限不是调用次数上限。超长原文可分多次读取。每次 Reader 的约 6,000 原文字符没有包括历史、任务、父级上下文和模板，不能当总 token 上限。外部计数默认关闭；启用后计数失败或超过显式应用上限均显式返回，不裁切输入。没有动态 state 续读，预算和动态 State 边界见 [超长上下文](../docs/long-context.md)。

计数开启时，总期限从调用入口开始，包含等槽和批次排队；失败收据的有界落盘收尾单独计时。计数关闭时保留原外部批次 HTTP 时限。

新链路不用旧 `MAX_CHUNKS_PER_DOCUMENT`、`RELATIVE_SCORE_THRESHOLD` 或语义规则门禁。多 worker 各有独立并发限制，部署时需分配总容量。

## G1K 7.2B native 连接配置

连接已提供该模型的 native 端点时，可显式设置以下配置；不会改变软件默认模型或自动切换已有服务：

```bash
RWKVRAG_RAG_PIPELINE=rwkv
RWKVRAG_NATIVE_TRANSPORT=native
RWKVRAG_NATIVE_BASE_URL=http://127.0.0.1:18426/v1
RWKVRAG_NATIVE_MODEL=rwkv7-g1k-7.2b-20260930-ctx25600
RWKVRAG_NATIVE_REQUIRE_MODEL_IDENTITY=true
RWKVRAG_NATIVE_COMPLETION_PROTOCOL=g1j_plain
RWKVRAG_NATIVE_PLANNER_PREFILL='<think></think'
RWKVRAG_NATIVE_RESOLVER_PREFILL='<think></think'
RWKVRAG_NATIVE_WRITER_PREFILL='<think></think'
RWKVRAG_NATIVE_CONTEXT_WINDOW_TOKENS=25600
RWKVRAG_GENERATION_MAX_TOKENS=2048
```

`g1j_plain` 是现有传输模板名称，不限制模型必须为G1J。实际端点地址需按部署设置；上述变量放入所用的环境文件或导出后再启动应用。25600是**完整输入加输出预算**，不是可另加2048输出的输入容量；服务上限更小时仍取较小值，不裁切输入或提高服务限制。模型身份不匹配会失败，不自动改接其他模型。

默认不启用State路由；不要复用G1J或2.9B的训练State。需要引用时另按[State接口](NATIVE_STATE_ROUTING.md)显式绑定兼容模型。引擎的FP16权重/FP32 recurrent State需在服务端配置，客户端设置不能证明实际数值精度。

结构化传输保留调用者的JSON Schema属性顺序，不按字母排序，也不替调用者重排。属性生成顺序可能影响回答；Schema合法不代表事实、缺失判断、部分支持或引用正确，不能据此增加代码兜底或修补原始回答。

## G1K架构候选

[G1K任务边界](G1K_ARCHITECTURE.md)明确区分当前任务、Planner检索提示、Reader逐字选择与Writer回答。显式`anchored_v1`时，Reader增加当前任务输入，Writer不再把Planner派生字段作为任务提示；原文和输出不改。默认legacy不变，未通过真实语义验收，不自动启用、训练或部署。RWKV初始State不是文档/历史数据库，也没有自动动态State续读或跨文档合并。

## 历史改写与请求级预算

`current-question-v2` 将完整 `history`（含角色、顺序、所有用户和助手消息）与最新问题作为JSON数据交给模型，不按关键词合并、删除或解析指代。只接受严格的 `{"question":"..."}`，下游使用该模型改写结果；原始对话和输出保留在trace。无历史时不调用改写模型。输入超出模型容量时显式失败，不截掉早期轮次。它修复输入丢失，但当前真实多轮诊断未通过，仍有条件丢失、指代未解析和撤回要求恢复。不能作为已验证的历史功能修复启用。

旧 `current-question-v1` 只传用户消息，`current-question-g1k-v1` 只传上一条用户消息，两者都有信息缺失限制。为保留旧实验绑定，不原地替换其提示，也不自动迁移现有配置。

历史协议与预算相互独立。RWKV 的 `/v1/ask` 和 `/v1/material-ask` 可显式配置安全策略与预算，以下示例不切换历史协议：

```bash
RWKVRAG_NATIVE_EMPTY_EVIDENCE_POLICY=fail
RWKVRAG_NATIVE_ANSWER_QUALITY_POLICY=fail_citation
RWKVRAG_NATIVE_REQUEST_MAX_CALLS=128
RWKVRAG_NATIVE_REQUEST_TIMEOUT_SECONDS=180
```

128/180仅为配置示例，不是质量或吞吐验收结论。历史改写仍限 native 非matrix路径；请求预算和空证据保护覆盖普通、matrix与分层Writer路径。固定材料接口仍要求1–20份材料。

调用预算按逻辑客户端操作计数，不是HTTP次数：一次生成的tokenize和completion属于同一操作；独立的token预检也占一个名额。并发请求各自计数，底层模型客户端和并发槽共享。达到上限后，后续模型操作在客户端入口被拒绝；已准入操作可收尾，但同样受请求总时限约束。被拒绝的trace含 `request_call_admitted=false`，并不代表模型执行。`generation.request_budget` 保存准入/拒绝计数、时限和停止原因。

总时限从问答管线入口开始，不包含之前的HTTP解析或之后的数据库落盘。超时取消本请求的异步等待，保留已观察到的原始Writer输出、对应证据和调用trace；`answer_call_id` 标记超时响应实际保留的输出来源。不保证远端模型、共享batch或已经进入线程的索引查询立即停止，`provider_execution_cancelled` 保持未知。它也不替代跨请求总容量、内存或全局并发限制。

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

`POST /v1/material-ask` 只运行固定材料 Writer，输入 question、history、materials（1–20 份 SourceItem）。POST /v1/search 按 retrieval_mode 检索知识库、网络或混合材料，不生成最终 Writer 答案；展示 top_k 不限制 Writer 返回的引用来源。管理、导入和任务接口复用基点实现，新切块应使用新索引。

`answer` 保留原始模型文本。`generation.answer_span` 标出正文的 Unicode 起止位置；不能把它当事实审核。没有模型文本时 answer 为空、raw_model_answer 为 null。

- `completed` 表示传输返回完成；外部 `termination_verified=false`，其 stop 不能证明自然结束。
- `length`、`budget_exceeded`、`token_count_failed`、`timeout` 等保留错误与实际已有输出。
- `current_question_failed`、`planner_failed`、`retrieval_failed` 明确指出停止阶段。
- `no_evidence` 表示未选中证据并停止Writer，不推断知识库不存在答案；若上游读取或解析失败，使用 `resolver_failed` 等实际失败状态，不掩盖成无证据。
- `resolver_partial_failure` 表示部分读取或解析失败，即便 Writer 返回也不标整链路成功。
- `call_budget_exceeded`、`request_timeout` 表示请求调用上限或总时限已达到，不能当作自然生成完成。
- `answer_quality_failed` 的 `quality_failure_reason` 为 `missing_valid_citation` 或 `citation_syntax_or_identity` 时，只代表引用标签检查失败，不是模型内容审查；原始答案保持不变。

`generation.model_calls` 保存本项 prompt、原始输出、SHA、输入来源及阶段耗时。多项共享 batch 的完整 HTTP 字节只进入私有 `model_receipts` GridFS bucket，每批每事件一次；公开 trace 使用 `payload_scope=single_item_projection` 和 `private_receipt_id`，不会带出其他调用的内容。单项请求仍可保留自身原始字节。共享 batch 按唯一 ID 统计，计数 HTTP 单列。`retrieval.candidates`、读取来源和预算排除项均可查；sources 与 citation_map 覆盖 Writer 全部资料。片段、原文块及父级上下文携带 Unicode 坐标和 SHA。

原始输出不增删、不补引用、不静默重试。citation_audit 只检查标签，semantic_support_verified 保持 false。模型自己生成的“证据”列表不能替代真实输入来源。

`POST /v1/admin/source-catalog/audit` 可只读核对保存来源的文件目录、原件/解析快照、文档身份及逐字范围；缺失或冲突不猜补、不改接最新文件。使用方法与验证边界见[来源目录契约](SOURCE_CATALOG.md)。该检查不代表索引成员、语义引用或已部署服务通过验收。

超过 8 MiB 的问答/请求完整内容保存到 `rag_payloads` GridFS bucket，Mongo 集合保存摘要和引用，历史读取自动还原。原始回答不会为了入库被删节。数据库备份必须包含这两个 bucket 的 `files` 和 `chunks` 集合；它们没有公开读取 API，访问由数据库权限控制。

## 验证与可选训练能力

应用提供 `rwkvrag-state-data` 和 `rwkvrag-state-release` 接口；实验脚本、训练包与历史评测不随应用仓库发布，需要在本地研究工作区另行准备。数据导出、格式通过和loss下降均不能代替真实问答验收。

应用单元测试使用本地合成材料，不依赖未发布的历史实验数据；历史数据回归保留在研究工作区单独运行。

规划失败后继续用原问题检索时，若 Writer 完成且证据读取未失败，状态为 `planner_partial_failure`，历史归类为 `partial`。`generation.stage_status` 分别记录各模型阶段是否成功，`planner_fallback` 记录回退方式。Writer 截断/超时等状态仍优先保留，原始答案不修改。此前已经存储的历史记录不会被本次代码更新自动重写。

```bash
uv run pytest -q tests/test_native_rwkv.py tests/test_rwkvos_batch.py tests/test_model_transport.py tests/test_rwkv_pipeline.py tests/test_native_integration.py tests/test_verbatim_chunking.py tests/test_ingest.py
```

软件测试不证明模型答案正确，部署前仍需核验真实检索、答案与引用。

## 自动联网与混合检索

管理页可调用配置好的 1.5B StateTune 模型服务判断是否补充网络材料；API 显式传 `retrieval_mode: "auto"` 启用。支持强制 knowledge_base / hybrid / web。模型选择器只需一个兼容的 HTTP 端点，不需要另一个项目的源码或虚拟环境。早期训练与历史评测见 混合检索交付报告（本地研究资料，不随应用发布）。

本仓库内置 Tavily 或 SearXNG 检索适配器。软件默认选择 Tavily，未配置 Key 时明确返回配置错误，不向上游发送请求。复制 `.env.example` 后，设置私有的 `RWKVRAG_WEB_TAVILY_API_KEY`，或使用权限为 `0600` 的单 Key 文件并设置 `RWKVRAG_WEB_TAVILY_API_KEY_FILE`；两种 Key 来源不能同时启用，文件或变量也不能包含 Key 列表。SearXNG 设置 `RWKVRAG_WEB_SEARCH_PROVIDER=searxng` 与 `RWKVRAG_WEB_SEARXNG_BASE_URL`。这只配置网络材料来源；OpenSearch、MongoDB 和 RWKV 模型端点仍按上文部署。调用 `/v1/ask` 并传 `retrieval_mode: "web"` 可只用网络资料，传 `"hybrid"` 可合并网络和知识库资料。返回的网络原文快照、检索时间和来源仍在 trace 与引用面板中。

内置适配器一次查询只使用配置的单个 Key，遇到失败或额度耗尽也不会尝试另一把 Key。切换 Key 是管理员更改私有配置后的独立操作。设置 `RWKVRAG_WEB_GUARD_PATH` 为所有 API worker 共用的可写 SQLite 文件路径后，网络请求会跨进程串行、至少间隔一秒；401/402/403/432/433 进入默认一天冷却，429 遵守有界的 `Retry-After`（缺失时默认 60 秒），不会立即重试。路径未设置时只有单个 API 进程内的保护；多实例部署必须共用该路径或在统一出口实现等效限流。更换为新的有效 Key 后使用新的哈希身份，不受旧 Key 的冷却记录影响。密钥不写入 guard 文件。

```bash
curl -sS http://127.0.0.1:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"question":"FastAPI 与 Starlette 最新发布说明有什么差异？","retrieval_mode":"web","top_k":5}'
```

这一步只检查网络检索与来源快照；完整回答用相同问题调用 `/v1/ask`。混合模式改传 `"retrieval_mode":"hybrid"` 并指定自己的 `knowledge_base_id`。不要把上面的命令成功当作答案正确。

内置提供方若返回401等错误，`retrieval.provider_failures[].error_code`会记录安全的`web_upstream_http_401`等代码；冷却期内记录`web_upstream_circuit_open_401`等代码，不再请求上游。不会保存提供方错误正文、请求密钥或轮换其他密钥。混合模式在知识库成功而网络失败时保留已有资料并标记部分检索失败。

`"auto"` 需要额外配置 `RWKVRAG_WEB_ROUTER_BASE_URL`、`RWKVRAG_WEB_ROUTER_MODEL`，以及模型服务需要时的 `RWKVRAG_WEB_ROUTER_API_KEY`；它让模型判断是否补网。选择器未配置或输出不合协议时会明确失败，不会改用关键词规则。内置网络检索的单元与模拟传输测试已覆盖快照、引用身份和凭据不进入 trace；真实提供方与多项目端到端质量需要单独实测，不能从配置可用推断通过。

不购买搜索 API 的本机备选部署见 [SearXNG Compose](deploy/searxng/README.md)。它已完成真实 `web`/`hybrid` 检索与一条简单问答冒烟检查；SearXNG 材料当前只含网页摘要，不能据此认定复杂比较题质量通过。本机验证记录（本地研究资料，不随应用发布）。
