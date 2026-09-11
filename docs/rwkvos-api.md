# RWKV传输与state

当前项目固定使用RWKV7 G1j 2.9B。服务器本轮训练与推理使用物理GPU3；训练、协议和结果见[StateTune经验](statetune-experience.md)。旧外部API探针仅证明当时服务自报身份，不能替代本轮底模SHA核验。

`rwkvos_batch`发送完整原始prompt，按批次返回的index对应每个请求。认证通过本机环境配置，不进入公开trace。共享批次的完整收据与单项公开记录分开保存；每项保留原始输入、输出、哈希、阶段与错误。

历史外部服务的 `finish_reason="stop"` 不能单独证明实际EOS。需要使用有实际token记录的服务验证正常结束。`stop_tokens`省略、空数组与`[0]`也不能混为同一协议。

本轮Writer/Reader使用带末尾换行的Assistant前缀，Planner使用对应的无末尾换行协议。模型名前缀、state层数、轴序、dtype和底模SHA需要同时匹配。六份最终state的映射见[STATES.json](../artifacts/statetune-20260911/STATES.json)，下载与恢复见[实验附件](artifacts.md)。

训练state不能替代完整动态会话状态。本项目没有已验证的session续读、分叉或跨文档状态合并。2026-09-11测试后按要求保留比较服务，未恢复原问答模型服务。

旧版文档中的外部API、GPU2初期训练和本机数值探针完整保存在归档的同一相对路径，属于历史条件，不是当前部署说明。
