# 检索 StateTune 候选 V8：来源配对与完整数据池

[CANDIDATES.jsonl](CANDIDATES.jsonl)是 8,943 条结构候选的完整池：[ASSEMBLY-SUMMARY.json](ASSEMBLY-SUMMARY.json)固定输入哈希、分集、角色、生成运行和失败数。[REVIEW-QUEUE.jsonl](REVIEW-QUEUE.jsonl)是初版机械审读队列，5,421 条，所有行 `review.accepted=false`。这两份文件都不是训练导出；[V9 最终机械队列](../retrieval-v9-20260924/README.md)只增加 41 条明确的缺材料证据候选，不改本版原文。

[PAIR-JOBS.jsonl](PAIR-JOBS.jsonl)按同分集、同类项目配对固定 README 摘录；[配对生成器](../../../scripts/retrieval_statetune/build_balanced_evidence_v8.py)要求两个来源各有逐字引文，同时生成单项目证据不足对照。教师原始响应和失败仅保留本机 `data/quality-runs/retrieval-state-v8-20260924/`。V1 试点整体结构失败且保留；V2 在 438 个配对任务中得到 509 条结构候选。**逐字引用只说明原文存在，不说明两段原文回答同一比较维度。** 作者固定抽样审读发现明显错配，详见[数据报告](../../../docs/archive/2026-09/retrieval-dataset-v9-20260924.md)。

完整池还包含 V7 的单份证据、成对不同摘录反例、补查与停止状态，以及 V3/V4 规划草稿。来源先按仓库族训练/开发/留出隔离；历史旧题只作为禁入哈希和回归，不作新标签。代码许可证字段不等于 README 文本的商用训练许可。当前没有独立语义复核、训练或正式服务变更。
