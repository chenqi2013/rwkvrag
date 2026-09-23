# 检索决策候选 V7：单份证据与缺口补查

本目录只保存**未独立审读的候选和任务绑定**，不是可训练发布集。191 个来源任务、900 个补查任务依赖 [V4 固定 README 清单](../retrieval-v4-20260924/SOURCES-CURATED.json) 和原有规划草稿；教师原始请求、响应、失败及费用收据在本机忽略目录 `data/quality-runs/retrieval-state-v7-20260924/`，没有密钥进入仓库。生成器为 [build_decision_candidates_v7.py](../../../scripts/retrieval_statetune/build_decision_candidates_v7.py)。

单份证据任务让教师写支持或缺少另一项目资料的题，并验证引用是来源块内逐字连续跨度。补查任务使用**显式模拟执行状态**，分别表示格子未搜和读后尚无已验收证据；它没有声称真实网络检索发生。首批教师试点因项目简称与字段不符留下原失败，V2 提示另存；未把失败算入候选。此批正例偏单项目、负例偏比较题，单独训练会形成危险的题型/标签相关性，因此 V8 另外生成两来源正例及单项目不足对照。

V7 的教师草稿经[V8 组装器](../../../scripts/retrieval_statetune/assemble_dataset_v8.py)统一成 `status` 和逐字 `quotes` 目标；来源、格子、提示和目标哈希随每行保存。旧评测问题/答案未作为教师输入。V7 自身没有训练导出，也没有运行模型实验。
