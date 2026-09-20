# P0 第二轮：Writer 提示词对照与评测模板校正

日期：2026-09-16。承接 [WeKnora 差距评审](weknora-gap-review-20260916.md) 与 [首轮基线](p0-quality-baseline-20260916.md)。

## 结论

完成 48 次真实 Writer 调用。新增逐项核对提示词在受控对照中退步，未启用为默认配置。发现并修复 smoke 执行器未显式传递应用使用的 canonical 模板的问题；这是评测接线修复，应用此前已使用 canonical，不是上线后质量提升。

| 条件 | 开发题通过 | 新编题组通过 | 用途 |
| --- | --- | --- | --- |
| evidence_first + legacy | 3/8 | 5/8 | 提示词对照基线 |
| evidence_checked + legacy | 2/8 | 2/8 | 候选退步，不采用 |
| evidence_first + canonical | 3/8 | 7/8 | 复用题组的模板诊断补测 |

以上均为本任务主代理按冻结 gold、完整材料和历史逐题复核，不是第二位独立审核者。新编题组在最初对照前冻结；canonical 补测已复用暴露题组，不能再称新盲测。没有完整 RAG 验证，也没有 canonical 下的候选提示词结果。

## 已实现

- 增加通用 `evidence_checked` 实验选项：要求逐项列出原文、结论及证据不足。现有默认提示词保持不变。
- 新增 8 道冻结题（21 项事实），覆盖标题干扰、历史版本纠正、空资料、部分缺失、冲突、表格单位、长材料尾部及真实零值。没有针对题名或答案的业务规则。
- 实验计划在推理前绑定源码、题目和验收标准；开发与留出结果分开统计。未因失败改 gold、重试或删除分母。
- 引用审计现在识别 `[资料 N]`、非法或不存在编号；实验格式中的单行“原文”另作逐字匹配诊断。前端显示相应问题，模型原文不改写。逐字匹配只证明文本出现在被引材料中，不能证明结论被支持。
- 执行器新增 `--writer-transport-protocol` 和 `--writer-state-id`；质量验收核对 manifest 与实际 trace 的协议及有效 state。

## 为什么拒绝候选

legacy 下，开发题丢失原先通过的筹款题；留出题改善部分资料缺失题，但历史纠正、空材料、表格引用和真实零值题退步。开发题输出从 1,293 字符增至 11,325，耗时约 51 秒增至 421 秒；留出题从 463 字符增至 8,381，约 19 秒增至 315 秒。候选分别有 4/8、3/8 达输出上限；常见失效为重复格式、照抄占位引用和把转述写成“原文”。

这是该候选在 legacy 条件下的负面证据，不能外推为所有结构化提示词无效。根据冻结验收标准，本候选没有进入默认配置或完整 RAG 放行阶段。

## 模板差异与纠正后的基线

检查本机应用设置发现：应用使用 `rwkv_g1j_no_think_v1`，早期 smoke 使用 `legacy`。两者在 `Assistant: <think></think>` 后相差一个换行，这也是训练模板的一部分。

补测保持模型 `rwkv7-g1j-2.9b-20260831-ctx16384`、有效 state `writer-trace-300`、材料、历史、内容提示词及远端参数不变。逐题比较实际 trace：16/16 的 messages、parameters、evidence_ids 相同，canonical prompt 恰好是 legacy prompt 加一个换行。源码审计功能虽在两轮之间增加，但未改变上述实际模型输入。详见 [协议对照证据](../artifacts/writer-p0-20260916/PROTOCOL-COMPARISON.json)。

开发题仍通过 3/8，但通过项发生变化：筹款题将 29 天写成 13 天而失败，长材料题补上有效引用而通过；列车、工作站题重复至截断；空材料仍杜撰销量 0 台及资料 1。新编题组通过 7/8，剩余长材料题事实正确但没有引用。不能把小样本差异宣称为产品总体准确率。

## 证据与验证

- [冻结实验计划](../artifacts/writer-p0-20260916/PLAN.json)、[补测计划](../artifacts/writer-p0-20260916/PROTOCOL-FOLLOWUP-PLAN.json)
- [提示词对照报告](../artifacts/writer-p0-20260916/COMPARISON.json)、[引用独立诊断](../artifacts/writer-p0-20260916/CITATION-DIAGNOSTICS.json)
- 各组目录保留完整答案投影、逐题复核、manifest、执行汇总和质量报告；原始响应没有覆盖。
- 完整 HTTP 收据、冻结源码、材料和 trace 位于本机 `data/quality-runs/20260916-writer-checked-v1/` 与 `data/quality-runs/20260916-writer-canonical-{development,holdout}/`。仓库投影不替代完整运行包；跨机器重放验收需要完整目录。
- 前端 23 项测试及 TypeScript/Vite 构建通过；最终相关后端回归 336 项通过（覆盖本轮改动，非全仓库测试），结果见同目录测试日志。新增 canonical 执行器测试直接检查发往 mock 服务的换行和 Writer 专用 state，而非只检查配置对象。

当前应用未重启，源码修复尚未部署。下一步应以 canonical 配置为唯一实验起点，重点检查 Reader 到 Writer 的事实覆盖及引用绑定；新候选需要重新冻结未参与调试的题组，并在固定材料通过后验证完整 RAG。Wiki 和自主代理仍应排在可靠问答验收之后。
