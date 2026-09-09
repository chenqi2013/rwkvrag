# 2.9B 外部 API 与 state

当前使用 `https://api-3b.rwkvos.com/v1/batch/completions`，模型列表与实际响应均报告 `rwkv7-g1j-2.9b-20260831-ctx16384`。这验证了服务自报身份，远端权重 SHA 未独立核验。本机官方 2.9B 已通过完整加载、短输入推理和 state 梯度检查，尚未部署备用 API 或实现自动切换。本机资源不足时，最新授权允许使用 `rwkv-8222` 的 GPU2，仍使用 2.9B；本轮尚未访问或启用该服务器。

## 已接入的传输

`RWKVRAG_NATIVE_TRANSPORT=rwkvos_batch` 使用独立客户端。原 `native` 传输保留兼容，不能仅替换它的 base URL。

- 完整原始 prompt 放进 `contents[]`，使用 `User: / Assistant:` 格式。同参数和 state 的并发调用合为 batch，按 `choices[].index` 严格对应 `message.content`，不猜测缺失或重复的 index。
- 全局默认 32 项并发、每批最多 8 项；取消某一调用不取消同批其他项。无自动重试。
- 认证由本地 CF Access 环境变量提供，配置使用 SecretStr，认证值不进入 trace。state 缺省时省略 `state_id`；过期状态的错误保留，不自动退回零状态。
- 精确输入、共享批次 ID、每项调用 ID、原始请求/响应字节与 SHA 均可追溯。中断时未确认是否发出的 HTTP 单列 unknown，不能算成零或已完成。
- `stop_tokens` 省略、`[]` 和 `[0]` 三种设置不同。已测开发配置为 `[0]`；温度最小 0.001，`top_p=0` 在上游进入贪心采样。

外部服务不返回 usage，并且**输出达到 max_tokens 时也会报告 `finish_reason="stop"`**。已用 1 token 边界生成实测，原始收据保留。因此 `completed` 只说明返回通过传输校验；`termination_verified=false`，不能称自然完成。初期 [连通性记录](rwkvos-api-probe-20260908.json) 保留，旧文字对 stop 的解释已由此次探针纠正。

默认 complete 前缀实际为 `Assistant: <think></think>`；可选 continuation 会省略末尾 `>`，保留模型实际生成的字符，再通过 answer_span 标识正文。两者是不同实验条件。官方未闭合 fake-think 候选在固定材料上增加了正确事实，也产生新无依据断言，未被设为默认。[固定版本模板实现](https://github.com/Alic-Li/rwkv_lightning_cuda/blob/6253c9e0345a1c5e42bf0a10e722ba09de06a2ac/src/rwkv_inference_engine.cpp#L680)。

后续两组 batch8 → 相同输入逐条调用 → 原 batch8 重复实验中，16 个槽位的三次输出逐字一致；这个小样本没有观察到批处理造成的差异。不过另一组 80 项控制的 10 个请求正文、批次成员及顺序完全相同，仍有 6 项输出变化。`top_p=0` 因此不能作为本服务输出可逐字复现的保证。恢复历史请求的原始 JSON 字节也未恢复 CAN 的历史输出；不能单凭这些现象推断远端换了权重或确定某个内核有错。见 [完整对照](../llamaindex-retrieval/eval/rwkvos-reader-followup-20260909/README.md)。

## 输入预算与长上下文

`POST /v1/tokens/count` 使用 `{"text":完整原始prompt}` 获取单项服务计数。传入 `contents` 数组会返回拼接总数，不能当每项计数；`messages` 会加模板，不能替代 raw prompt。

可选输入计数默认关闭，保持基线请求条件。开启后记录每条完整输入计数；可另设显式应用输入上限，超过时返回 `budget_exceeded`，不裁切或发该项生成。计数失败也显式记录，不伪造预算通过。计数不提供真实输出 usage、server limit 或 EOS 证明。

实测 8,169 / 16,375 / 32,624 token 三档输入全部返回 200，各答对 3/3、0/3、1/3 个合成事实。不能把模型名中的 ctx16384 当部署硬上限，也不能把接受输入当成信息保持证明。服务内部的 prefill 分片和应用跨调用状态是不同机制，见 [超长上下文说明](long-context.md)。

## 外部 state 与本机 statetune

上游支持 `/v1/state/upload` 上传单个 `.pth`，之后通过 `state_id` 初始化 batch。上传阶段的检查不等于模型兼容验证，真正加载时才检查完整层数、形状和类型。[上传与加载实现](https://github.com/Alic-Li/rwkv_lightning_cuda/blob/6253c9e0345a1c5e42bf0a10e722ba09de06a2ac/src/rwkv7_fast_v4.cu#L2337)。

官方 G1j 2.9B 本地文件已经核验：SHA256 `966f3420f833532aae3fb1fd6326533b08d43d23b7b03eaa2f0694a30b64a239`，1,062 个基础张量、2,948,065,280 参数，完整 key/shape 匹配。32 个 `blocks.N.att.time_state` 各为 `[40,64,64]`。canonical 文件轴序为 `[H,V,K]`，外部上游 CUDA loader 内部再转置一次，不能提前双重转置。

本机 RTX 5070 Ti 的 BF16/FLA 40-token 前向与一次 state-only backward 已通过：所有基础权重冻结且无梯度，32 层 state 梯度全部有限且非零；峰值 allocated 约 5.79 GiB。这只是短输入计算图检查，不是长序列训练容量或收敛验证。优化器更新为零。

本地 PEFT/FLA 与 `rwkv` 0.8.30 纯 Torch 推理又做了同一短输入、零 state 和非对称 state 的逐位置 logits 对照：零 state 的 top1 相同 39/40，非对称 state 为 40/40。数值存在差异，不能把两实现称为逐字等价；非对称探针和原始 logits 已归档，未用零矩阵相等冒充轴序验证。

外部全零 state 兼容探针在上传约 10.5 MB 的文件时出现 `ReadError`，没有收到 HTTP 响应或 `state_id`。上传的远端结果未知，无法定位或删除可能已创建的状态；没有静默重试，也没有执行随后带 state 的生成。此次没有完成外部加载验收。该文件未训练，不能代表 tuned state 的能力或完整长文记忆。

上传文件初始化 WKV，不能恢复完整的动态 shift/elapsed。上游另有 session 状态接口，但本 RAG 尚未接入，也未验证部署的会话隔离、续读和分叉行为。不能把上传的 tuned state 称为已有长文记忆。

质量实验全部为零 state，没有优化器更新或调用旧服务器；非对称 state 仅用于本地数值诊断。下一步从真实 trace 建立统一协议的训练目标，分开检查格式、任务范围、证据选择和事实引用，并用新留出题验证。已有开发题不能兼作训练后的盲测。

## 当前质量与运行位置

本机 CPU 的 OpenSearch 恢复了 5,000 文档 / 45,960 原文块，逐条 ID 与内容核验通过；本机新索引 UUID 与旧服务器不同。MongoDB 是管理 API 依赖，完整部署验收尚未完成。旧任务服务及自动启动已停用，约 89.89 GiB 显存已释放，磁盘资料保留。

外部完整 Wiki 基线为 0/29 正确事实、正式完整 0/8；固定材料 Writer 为 19/28。详细条件、失败分析和原始调用见 [2.9B 归档](../llamaindex-retrieval/eval/rwkvos-29b-20260909/README.md)，不与历史 13.3B 分数混用。
