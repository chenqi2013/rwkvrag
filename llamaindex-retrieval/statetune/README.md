# RWKV 2.9B StateTune 数据与训练

[实践总结](../../docs/statetune-experience.md)是本轮经验的主要入口。最新数据为 `datasets/trace-v1-2000/draft-v6-seeded`，正式训练包为 `datasets/trace-v1-2000/release-v1`。2000条唯一训练数据已经独立复核，六组state训练与本轮推理已完成；成绩与局限见总结。

## 数据流程

1. 从真实trace保留完整问题、历史、模型可见证据与错误输出。
2. 依据该阶段可见原文修正目标，独立复核纠错种子。
3. 构造同类机制的新实例，保留来源、种子关联和prompt/target哈希。
4. 按来源文档隔离训练与评测，单独标记暴露旧题和人工场景。
5. 对齐生产模板及传输前缀，导出只监督target与EOS的训练token。

本轮17个纠错种子派生Writer1400、Reader450、Planner150条。Writer100/300/600是嵌套训练子集。旧草稿不作为当前训练入口。

## 入口

- `prepare_writer_data.py`：早期Writer数据构造入口。
- `src/llamaindex_retrieval/statetune.py`、`state_release.py`：来源审计、数据导出和独立复核发布。
- `train_trace_state.py`：本轮正式分阶段训练入口；配置、数据和运行源码均按哈希绑定。
- `train_released_state.py`、`preflight_state.py`：训练与零更新反向预检。
- `state_tokens.py`：固定词表、prompt/target分开编码、mask和EOS检查。

运行本轮训练入口前，先恢复[实验附件](../../docs/artifacts.md)，准备与训练清单一致的底模、冻结RWKV-PEFT源码与Python环境。不要直接把历史服务器路径当成新机器配置。具体命令参数可执行：

```bash
python llamaindex-retrieval/statetune/train_trace_state.py --help
```

正式训练包仅提供train token；数据审计与评测标签复核不等于训练进程可以读取评测gold。模型只学习修正目标，原始错误输出留作审计。

## 协议与训练约束

Writer/Reader使用带末尾换行的Assistant前缀，Planner使用对应的无末尾换行协议。Prompt token的label为-100，只监督正确target和EOS=0，并使用next-token标签移位。超预算拒绝，不能靠截断答案通过检查。

每份state从零开始；底模冻结；逐次保存实际更新记录，结束时检查底模所有张量不变。独立复核记录是可追溯的审阅声明，不是身份认证签名。

旧Reader/Writer小型数据保留用于代码回归；原始实验trace、旧方案说明和过时大草稿位于[归档附件](../../docs/artifacts.md)，不再列为正在执行的工作。
