# RWKV 传输与 State

模型与端点由运行配置指定。身份、底模 SHA 和逐次 trace 用于核对实际请求；历史模型名或部署说明不能替代运行时验证。配置入口见[服务说明](../llamaindex-retrieval/README.md)。

`rwkvos_batch` 发送完整原始 prompt，并按返回的 index 对应每个请求。认证留在本机配置中。共享批次的私有完整收据与单项公开记录分开保存；每项保留输入、原始输出、哈希、阶段和错误。

外部服务的 `finish_reason="stop"` 本身不证明实际 EOS。`stop_tokens` 省略、空数组与 `[0]` 属于不同配置；验证终止条件需要实际 token 记录。

提示前缀、换行和 State 协议须匹配所选配置。模型、State 层数、轴序、dtype 和底模 SHA 必须兼容；其他底模的训练 State 不直接复用。具体绑定见[State 路由](../llamaindex-retrieval/NATIVE_STATE_ROUTING.md)。

初始训练 State 与动态会话状态的区别见[上下文边界](long-context.md)。此接口说明不承诺 session 续读、分叉或跨文档状态合并能力。
