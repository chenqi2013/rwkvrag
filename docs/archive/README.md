# 历史报告索引

这些文件是可追溯的阶段记录，不是当前任务指令。先读 [CURRENT](../CURRENT.md)。报告中的“最新”“已上线”“未部署”和“下一步”属于各自记录时点。

## 2026 年 9 月

- [按真实缺陷准备StateTune：36题2391节点统计、113组标注任务与8份诊断范例](2026-09/statetune-defect-audit-20260922.md)

| 报告 | 原路径 |
| --- | --- |
| [复读位置与输入格式核查](2026-09/repetition-input-audit-20260921.md) | 0新调用；旧28题prompt逐字一致，新7.2B循环均为短输入 |
| [2.9B / 7.2B模型配对与复读复验](2026-09/model-size-paired-results-20260921.md) | 752次完整执行；7.2B总体改善，选择与复读仍未通过 |
| [7.2B受控回放：重复性与旧State迁移](2026-09/layered-controlled-replay-results-20260921.md) | 1,890次新调用；漂移与事实值丢失复现，未晋级 |
| [分层协议离线验证](2026-09/layered-offline-validation-20260921.md) | 53项初期检查；历史7,085次调用传输审计 |
| [分层协议与失败回放设计准备](2026-09/layered-design-preparation-20260921.md) | 设计与只读材料；0模型调用、0训练更新 |
| [全题型应用 v2：478 题完整执行与 66 题复核](2026-09/broad-qa-v2-results-20260920.md) | 参考材料与实际检索分开；旧语料恢复另行回归 |
| [全题型 Reader 扩大回归：保留旧题，候选未通过](2026-09/broad-reader-results-20260920.md) | 4,896 次调用；40 个新增模板族 |
| [Reader 呈现回归：表格改善与负向迁移](2026-09/reader-presentation-results-20260920.md) | 新主线回归报告，640 次调用 |
| [本地 vllm-rwkv 更新与解码回归：未观察到采样收益](2026-09/vllm-decoding-results-20260920.md) | 新部署与 1,120 次回归报告 |
| [表头重复修复与本地 vLLM 引擎核对](2026-09/atomic-context-dedup-20260920.md) | 新修复报告与只读引擎审查 |
| [表格呈现检查：重复表头与实验设计](2026-09/table-presentation-inspection-20260920.md) | 新检查报告，未执行模型对照 |
| [Reader 标签扩大复验：收益与退步](2026-09/reader-label-replication-20260920.md) | 新报告，直接存入历史区 |
| [Reader 输出标签单变量对照](2026-09/reader-label-results-20260920.md) | 新报告，直接存入历史区 |
| [知识维护第四批：正式文档修订与失败恢复](2026-09/document-revisions-20260919.md) | docs/document-revisions-20260919.md |
| [P0 第一批：质量验收与真实基线](2026-09/p0-quality-baseline-20260916.md) | docs/p0-quality-baseline-20260916.md |
| [引用原文展示修复](2026-09/citations-ui-20260919.md) | docs/citations-ui-20260919.md |
| [G1j 7.2B 评测与证据分层架构设计](2026-09/g1j72-evidence-architecture-20260920.md) | docs/g1j72-evidence-architecture-20260920.md |
| [G1j 2.9B / 7.2B 固定材料对照测试](2026-09/g1j72-test-report-20260920.md) | docs/g1j72-test-report-20260920.md |
| [WeKnora 对照评审：RWKVRAG 还差在哪里](2026-09/weknora-gap-review-20260916.md) | docs/weknora-gap-review-20260916.md |
| [知识库与网络混合检索：自动选择器接入及 StateTune 验收](2026-09/hybrid-search-20260919.md) | docs/hybrid-search-20260919.md |
| [证据支持判断：四组固定对照结果](2026-09/evidence-support-results-20260920.md) | docs/evidence-support-results-20260920.md |
| [知识维护第三批：原文快照与索引来源绑定](2026-09/source-revisions-20260919.md) | docs/source-revisions-20260919.md |
| [知识维护第二批：索引版本查看与回滚](2026-09/index-rollback-20260919.md) | docs/index-rollback-20260919.md |
| [对比问答：逐项检索、证据绑定与回答核验](2026-09/comparison-task-matrix-20260920.md) | docs/comparison-task-matrix-20260920.md |
| [P0 第二轮：Writer 提示词对照与评测模板校正](2026-09/p0-writer-experiment-20260916.md) | docs/p0-writer-experiment-20260916.md |
| [知识维护第一批：失败不破坏活动索引](2026-09/index-maintenance-20260919.md) | docs/index-maintenance-20260919.md |
| [自动 Wiki：本地上线与验收](2026-09/automatic-wiki-20260919.md) | docs/automatic-wiki-20260919.md |
| [属性证据第一阶段测试报告](2026-09/atomic-evidence-test-report-20260920.md) | docs/atomic-evidence-test-report-20260920.md |
| [对比问题质量修复与验收（2026-09-19）](2026-09/comparison-quality-20260919.md) | docs/comparison-quality-20260919.md |
| [RWKV StateTune：数据集如何构建，用什么训练](2026-09/statetune-experience-20260911.md) | docs/statetune-experience.md |

## 查找与恢复

[迁移清单](MIGRATION-20260920.json)记录原路径、新路径和迁移前 SHA-256；基线提交为 77c74f9e。原始文件可以从该提交读取。本次只整理叙述文档，未修改冻结实验目录、原始 trace、模型和训练数据。

维护中的原子证据契约和 assessment 提案已分别改用 [EVIDENCE](../EVIDENCE.md) 与 [ASSESSMENTS](../ASSESSMENTS.md)。

## 2026-09-21 Typed漏斗修复

- [工程修复、原子13条与完整36题结果及限制](2026-09/typed-funnel-repair-20260921.md)

- [证据流转诊断与历史/当前结果区分](2026-09/evidence-flow-presentation-20260921.md)

- [Writer处理状态交接：144次双轮调用、漂移与失败诊断](2026-09/writer-evidence-handoff-20260921.md)
