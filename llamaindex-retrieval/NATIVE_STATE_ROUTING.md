# Native 请求级 State 路由

这是**默认关闭的应用接口**，不是已部署/通过数值验收的引擎功能。遵守[架构规则](ARCHITECTURE_RULES.md)。只选择已注册的初始 State；不修改 prompt、已选证据、模型原文或引用。

## 配置

仅适用 `RAG_PIPELINE=rwkv`、`NATIVE_TRANSPORT=native`、`NATIVE_COMPLETION_PROTOCOL=g1j_plain`，三个角色均采用闭合 no-think prefill。参数均有 `RWKVRAG_` 前缀。以下是示意，引用必须替换为对应服务实际注册的对象，不能直接用占位符运行：

```dotenv
RWKVRAG_RAG_PIPELINE=rwkv
RWKVRAG_NATIVE_TRANSPORT=native
RWKVRAG_NATIVE_MODEL=rwkv7-g1j-7.2b-20260831-ctx16384
RWKVRAG_NATIVE_COMPLETION_PROTOCOL=g1j_plain
RWKVRAG_NATIVE_PLANNER_PREFILL=<think></think
RWKVRAG_NATIVE_RESOLVER_PREFILL=<think></think
RWKVRAG_NATIVE_WRITER_PREFILL=<think></think
RWKVRAG_NATIVE_STATE_ROUTING='{"model":"rwkv7-g1j-7.2b-20260831-ctx16384","roles":{"plan":null,"reader":null,"writer":"registered-writer-reference"}}'
```

- `NATIVE_BASE_URL` 另行配置为待验证的实际引擎地址，本例不改变服务地址。
- `model` 必须精确等于 `NATIVE_MODEL`；这只是配置身份绑定，不是权重/State 内容哈希证明。
- 必须显式声明 `plan`、`reader`、`writer`。`planner` 阶段映射到 `plan`，`resolver`/`reader` 到 `reader`，`writer` 到 `writer`。
- `null` 明确选择 **base/no-ref 请求**，不继承另一角色。它不证明服务端零 State：若需严格零 State 对照，应由操作者注册经过验证的零 State 并显式填引用，同时完成引擎验证。
- 非空字符串引用只接受 1–128 个安全标识字符（首字符字母或数字，其余字母、数字、`_ . : -`）。不接受空串、文件路径或 URL，不做字符串修复。
- 启用 task matrix 时还须声明 `assessment`、`followup`、`review`，可显式为 null。已有 matrix 调用会传递逻辑角色：assessment/followup 使用 planner 阶段，review 使用 resolver 阶段。未声明角色不会退回主角色。
- 历史任务纠错仍用 planner 传输阶段；可显式选独立 `current_question` 角色，见下节。typed-funnel 的结构化节点仍多走 resolver，不是新增独立 condition State。
- 不与 `RWKVOS_*STATE_ID`/`RWKVOS_MATRIX_STATE_IDS` 混用。旧 batch 路由行为不变。
- 未配置 `NATIVE_STATE_ROUTING` 时保留原请求路径，不新增 State 探测或读取参数。

客户端构造时复制并固定映射。问答请求不能上传 State、指定任意引用、改变全局默认或续接上一请求的演化 State。当前没有热改映射管理接口；更新映射须受控重建应用客户端，不要求重新加载基础模型。

## 独立历史任务解析角色（可选）

在已有、非 `raw` 的 `NATIVE_HISTORY_PROTOCOL` 配置下，向 `NATIVE_STATE_ROUTING.roles` 显式添加 `"current_question":"registered-task-reference"`，或 `"current_question":null`。仍须保留 plan/reader/writer 三个键。应用只在真正执行历史改写时选此角色；检索 Planner 继续用 plan，Reader/Writer 不变。仅 native、非 task-matrix 可用；raw 历史配置下声明此键会被拒绝，避免配置被静默忽略。

- **键缺省**：保留旧 planner→plan 行为，不自动隔离或迁移；需要隔离时必须显式添加此键。
- **显式 null**：该角色发 no-ref 请求，不能继承 plan；不证明服务端零 State。
- **非空 ref**：只用该引用；不存在、元数据不合格或生成失败时不回退 plan/no-ref。任务解析失败仍阻断下游。
- 无历史不新增任务解析调用。角色选择与引用映射在构造客户端/流水线时固定，修改配置字典不热更新它们。
- 本接口不换提示词/协议、解码参数或模型输出；`current-question-v2` 仍未经语义验收，不能因增加路由而启用。当前没有由本接口自动获取或训练的 State。

`state_selection.role=current_question` 只证明应用选择/发送了这个逻辑角色，不证明实际张量隔离。操作者若给两个角色配置相同 ref，仍会共享初始模板；即使 ref 不同，也需独立验证其内容、训练来源、实际消费及数值行为。

## 每次生成的接口流程

在既有并发限制及活动调用超时内：

1. `/tokenize` 对完整输入计数，照旧检查输入＋输出上限。
2. `GET /v1/models`：必须只有一个与配置匹配的模型及足够上下文长度。
3. `GET /v1/rwkv/state/capabilities`：要求 `vllm-rwkv.state-cache.v1`、单 worker、enabled/supported/import_supported、process_local=true/durable=false 及有效快照大小。
4. 非 null 引用再 `GET /v1/rwkv/state/<ref>`：必须同协议/引用、同快照大小、initial、已处理/待处理 token 数与保留槽数均严格为整数 0。拒绝对话中间状态；`state_dtype` 在旧接口表示 shift 类型，不能当 recurrent matrix dtype。
5. `/v1/completions` 的该次 payload 独立携带 `vllm_xargs.rwkv_state_read_ref`；null 时不发此字段。强制核对完成响应中的模型身份，原文原样保留。

不缓存“引用可用”结论。每次非 null 生成增加 3 个 GET，null 增加 2 个 GET；这些成本计入该模型调用，不能把它当普通检索总调用预算已经实现。`writer_budget` 的 check-only 仍只做计数，不证明 State 就绪。

配置错误在创建客户端时拒绝；缺角色/非法调用返回 invalid_request；元数据不兼容/畸形返回 invalid_response；404 等返回 http_error；超时/取消保留已有收据。**该次模型调用不自动重试、不改为无 ref 生成。** 上层编排仍保留既有失败/部分失败策略，本接口不是新 E2E 质量门禁。

## 审计和不可证明的边界

`generation.model_calls` 中的 `state_selection` 记录 role、mode、state_ref、model、metadata_checked。HTTP trace 保存 GET/POST 方法及精确请求/响应 body 与 SHA；复用原认证及 recorder，不记录认证头。State 检查只读，不新增面向问答用户的 State 管理接口。

`metadata_checked=true` **不表示**：

- State 训练来源、权重版本、张量 shape/dtype/hash 已独立绑定；
- 服务确实消费了 ref、初始模板没有被修改、并发请求的演化状态没有串用；
- 数值一致、模型质量提升，或无 ref 服务端确为零 State。

所以 `state_consumption_verified`、`tensor_compatibility_verified` 保持 false。检查与生成之间存在竞争窗口；引擎必须原子获取引用，引用消失时拒绝请求，不能忽略 xargs 或静默回退。旧接口没有实例纪元/内容哈希/消费回执，同名 ref 重用无法在此可靠识别，不能把同模型名当驻留权重证明。

## 注册、容量和生命周期责任

引擎/操作者负责模型版本及 shape/dtype 兼容校验、不可变模板、每请求独立演化副本、容量配额、删除与重启失效。注册/查询/删除必须限制访问，引用 ID 不是授权凭据。应用不自动上传、重载或删除任何 ref，也不清理未知库存。服务重启或引用删除后，应重新核对并登记映射，不能复用失效配置或猜补。

应用测试使用 MockTransport，无真实引擎/GPU。启用实际 State 前仍须在确定版本上验证：导入与数值一致性、零/非零及无引用控制、并发模板隔离、过期/容量/重启行为、真实 Writer 有据/缺据/引用回归。工程测试不能替代这些验收。

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_native_state_routing.py tests/test_current_question_state_routing.py
```
