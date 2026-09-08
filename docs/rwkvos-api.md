# 2.9B 外部 API 与 state

当前使用 `https://api-3b.rwkvos.com/v1/batch/completions`，模型列表与实际响应均报告 `rwkv7-g1j-2.9b-20260831-ctx16384`。这验证了服务自报身份，远端权重 SHA 未独立核验。推理不可用时仅允许另行配置本机 GPU 备用，尚未实现自动切换或验证本机 2.9B 部署；不再占用 `rwkv-8222`。

## 已接入的传输

`RWKVRAG_NATIVE_TRANSPORT=rwkvos_batch` 使用独立客户端。原 `native` 传输保留兼容，不能仅替换它的 base URL。

- 完整原始 prompt 放进 `contents[]`，使用 `User: / Assistant:` 格式。同参数和 state 的并发调用合为 batch，按 `choices[].index` 严格对应 `message.content`，不猜测缺失或重复的 index。
- 全局默认 32 项并发、每批最多 8 项；取消某一调用不取消同批其他项。无自动重试。
- 认证由本地 CF Access 环境变量提供，配置使用 SecretStr，认证值不进入 trace。state 缺省时省略 `state_id`；过期状态的错误保留，不自动退回零状态。
- 精确输入、共享批次 ID、每项调用 ID、原始请求/响应字节与 SHA 均可追溯。中断时未确认是否发出的 HTTP 单列 unknown，不能算成零或已完成。
- `stop_tokens` 省略、`[]` 和 `[0]` 三种设置不同。已测开发配置为 `[0]`；温度最小 0.001，`top_p=0` 在上游进入贪心采样。

外部服务不返回 usage，并且**输出达到 max_tokens 时也会报告 `finish_reason="stop"`**。已用 1 token 边界生成实测，原始收据保留。因此 `completed` 只说明返回通过传输校验；`termination_verified=false`，不能称自然完成。初期 [连通性记录](rwkvos-api-probe-20260908.json) 保留，旧文字对 stop 的解释已由此次探针纠正。

默认 complete 前缀实际为 `Assistant: <think></think>`；可选 continuation 会省略末尾 `>`，保留模型实际生成的字符，再通过 answer_span 标识正文。两者是不同实验条件。官方未闭合 fake-think 候选在固定材料上增加了正确事实，也产生新无依据断言，未被设为默认。[固定版本模板实现](https://github.com/Alic-Li/rwkv_lightning_cuda/blob/6253c9e0345a1c5e42bf0a10e722ba09de06a2ac/src/rwkv_inference_engine.cpp#L680)。

## 输入预算与长上下文

`POST /v1/tokens/count` 使用 `{"text":完整原始prompt}` 获取单项服务计数。传入 `contents` 数组会返回拼接总数，不能当每项计数；`messages` 会加模板，不能替代 raw prompt。

可选输入计数默认关闭，保持基线请求条件。开启后记录每条完整输入计数；可另设显式应用输入上限，超过时返回 `budget_exceeded`，不裁切或发该项生成。计数失败也显式记录，不伪造预算通过。计数不提供真实输出 usage、server limit 或 EOS 证明。

实测 8,169 / 16,375 / 32,624 token 三档输入全部返回 200，各答对 3/3、0/3、1/3 个合成事实。不能把模型名中的 ctx16384 当部署硬上限，也不能把接受输入当成信息保持证明。服务内部的 prefill 分片和应用跨调用状态是不同机制，见 [超长上下文说明](long-context.md)。

## 外部 state 与本机 statetune

上游支持 `/v1/state/upload` 上传单个 `.pth`，之后通过 `state_id` 初始化 batch。上传阶段的检查不等于模型兼容验证，真正加载时才检查完整层数、形状和类型。[上传与加载实现](https://github.com/Alic-Li/rwkv_lightning_cuda/blob/6253c9e0345a1c5e42bf0a10e722ba09de06a2ac/src/rwkv7_fast_v4.cu#L2337)。

G1j 2.9B 预期为 32 个 `blocks.N.att.time_state`，每个 `[40,64,64]`，BF16 或 FP32。canonical 文件轴序为 `[H,V,K]`，CUDA loader 内部再转置一次。训练和上传前仍须记录确切基座 SHA 并检查所有张量；未进行本机全权重加载或训练验证。

上传文件初始化 WKV，不能恢复完整的动态 shift/elapsed。上游另有 session 状态接口，但本 RAG 尚未接入，也未验证部署的会话隔离、续读和分叉行为。不能把上传的 tuned state 称为已有长文记忆。

本轮全部为零 state，没有训练、上传新 state 或调用旧服务器。下一步从真实 trace 建立统一协议的训练目标，分开检查格式、任务范围、证据选择和事实引用，并用新留出题验证。已有开发题不能兼作训练后的盲测。

## 当前质量与运行位置

本机 CPU 的 OpenSearch 恢复了 5,000 文档 / 45,960 原文块，逐条 ID 与内容核验通过；本机新索引 UUID 与旧服务器不同。MongoDB 是管理 API 依赖，完整部署验收尚未完成。旧任务服务及自动启动已停用，约 89.89 GiB 显存已释放，磁盘资料保留。

外部完整 Wiki 基线为 0/29 正确事实、正式完整 0/8；固定材料 Writer 为 19/28。详细条件、失败分析和原始调用见 [2.9B 归档](../llamaindex-retrieval/eval/rwkvos-29b-20260909/README.md)，不与历史 13.3B 分数混用。
