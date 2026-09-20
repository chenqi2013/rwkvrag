# 全题型 Reader 扩大回归：未通过接入门槛

日期：2026-09-20。预登记提交 **65b4166c**；执行 15:17–15:38，8222 GPU3。

## 结论

4,896 次调用完整完成。多行呈现在一些表格和双记录任务上改善，但新 960 条整体正确率略低，旧题仍退步，跨轮存在少量语义变化。**没有证明多行格式整体更好，候选未接入主应用。** 旧题、标签和失败均未修改。

新集合首轮 JSON 正确率 87.604%，多行 86.979%，差值 −0.625 个百分点；40 个模板族重采样的 95% 区间为 −4.896 至 +3.333 个百分点。区间跨零，不能宣称格式带来确定收益，也不能仅凭这点差值宣布可靠退步幅度。

## 数据与变量

- 历史 40/64/160 集合全部保留，共 264 条成员记录；其中 40 条重复成员关系，224 个唯一原题 ID。
- 新增 960 条：40 个语义模板族，每族 12 对正负例。换名称、数值和三种问法不等于 960 种独立能力。合并后 1,224 条成员记录、1,184 个唯一原题 ID。
- 两种呈现各两轮；第二轮反转顺序。源文本、问题、上下文、语义提示、标签不变。新材料由助手编写自查，未经独立复核，后续只能作已见回归。
- 固定本地优化 vllm-rwkv 快照、G1j 7.2B、zero State、top_k=1、temperature=1、top_p=1、seed=11、32-token 输出预算。最大输入 989 tokens，均满足 4,096 上下文；无长度退出。
- 与实际 2.9B 应用共用授权 GPU，因此耗时不用于性能比较。完整请求与原始响应存档，没有隐藏重试或修补标签。

[冻结方案](../../../llamaindex-retrieval/eval/broad-reader-20260920/README.md)、[完整汇总](../../../artifacts/broad-regression-20260920/reader/SUMMARY.json)、[逐题结果](../../../artifacts/broad-regression-20260920/reader/ROWS.json)。

## 分题库结果

正确率包括非法输出在内，不能通过移除失败提高分数。下面是首轮；第二轮全表见汇总。

| 题库 | JSON 正确率 | 多行正确率 | JSON 误选/漏选/非法 | 多行误选/漏选/非法 |
| --- | ---: | ---: | --- | --- |
| 历史40 | 92.500% | 95.000% | 0 / 3 / 0 | 0 / 2 / 0 |
| 历史64 | 89.063% | 92.188% | 2 / 4 / 1 | 1 / 3 / 1 |
| 历史160 | 90.625% | 91.250% | 0 / 11 / 4 | 3 / 7 / 4 |
| 新960 | 87.604% | 86.979% | 0 / 119 / 0 | 0 / 125 / 0 |

第二轮新960：JSON 87.708%、多行87.396%；正例命中362/480与359/480，仍未观察到候选整体收益。新集合两种格式均未误选负例，但分别漏掉约四分之一正例；这不能称为可靠 Reader。旧集合仍有误选与非法输出。仅看零误选会掩盖阅读能力不足。

## 普通题和困难题同时看

- 实际发货数量的简单正例：JSON 12/12，多行 0/12；不能用表格收益抵消普通数值题损失。
- 明确否定 QA：12/12 → 4/12，第二轮多行 5/12。
- 容量表头与行：3/12 → 10/12；但宽度表头与行两种格式都为 0/12。
- 干扰后事实、明确条件例外：两种格式均 0/12。它们没有超出输入预算，不能解释成上下文截断。
- 双记录各自数值：0/12 → 8/12，第二轮多行 7/12；改善不代表已经完成冲突裁定。
- 定义、地点、日期、直接计数、零值、完整列表、许多范围和语言例在 JSON 下保持良好；这些普通能力必须继续进入接入门槛。

附录列出全部40族。名称按冻结数据保留；例如 `wrong_column` 当前具体材料是宽度表头与数据行/未测值，不能仅凭名字声称覆盖全部错列情况。

## 旧题退步和稳定性

首轮有 7 道历史题由正确变错，均来自 historical160；第二轮另有一条 historical64 正例变为漏选。所有差异在 [CHANGES](../../../artifacts/broad-regression-20260920/reader/CHANGES.json)，失败在 [FAILURES](../../../artifacts/broad-regression-20260920/reader/FAILURES.json)。旧题不会从下一轮删除。

首轮退步 ID：

- `historical160:9fe0dc1d2c6a2276`
- `historical160:d4110d22991b8c36`
- `historical160:f9b32528f2e7bd5f`
- `historical160:036dbbeebe64e2d4`
- `historical160:07ee90e80f74f725`
- `historical160:83dd712237eab922`
- `historical160:e8e19696c04aeb9e`

同配置跨轮比较，JSON 有 9 条原始 token 序列变化，其中 1 条语义标签变化；多行有 8 条原始序列变化，其中 7 条语义标签变化。JSON 与前轮 historical160 的 160 条基线中，1 条仅 JSON 空白格式变化，分类成绩一致，但预登记要求的原始输出完全一致仍未通过。

运行前后 2,444 个引擎文件、133 个依赖文件及实际加载的 FlashRWKV2 动态库哈希均一致，服务 PID 未变。**同配置、top-1 不能在本次运行中保证逐 token 一致。** 尚未定位数值计算、缓存或执行顺序等哪一环造成差异，不能把推测写成已确认根因。见 [重复差异](../../../artifacts/broad-regression-20260920/READER-REPEAT-DIFFERENCES.json)及前后 runtime 记录。

## 门槛、保留与下一步

无旧正确题损失、无题型误选增加、重复完全稳定、新题至少提升5个百分点、区间下界大于零、历史原始基线完全一致等门槛均未全部满足；仅“无题型非法输出增加”通过。候选未晋级。

保留清单中的 352 个旧文件哈希在运行后全部一致。原始调用归档为 [raw-calls.tar.gz](../../../artifacts/broad-regression-20260920/reader/raw-calls.tar.gz)，含 4,896 份原 JSON 文件；逐文件 SHA256 及解压回读核对见 [INTEGRITY](../../../artifacts/broad-regression-20260920/reader/INTEGRITY.json) / [RAW-MANIFEST](../../../artifacts/broad-regression-20260920/reader/RAW-MANIFEST.json)。

下一步仍应先设计原文字段绑定与独立问题匹配，单独验证抽取正确性、否定/条件/修订和普通事实题，再研究关系与分层总结。不能按本轮题型选择哪个格式分数更高，也不能把本轮已见失败拿来调参后再当新验证。实际应用 478 题、语料缺口及后端测试另外报告，不与本节分数合并。

## 附录：全部模板族正例命中

每格为12条正例命中数，第一轮 / 第二轮；每族另12条负例本次均未被误选。

| 模板族 | JSON | 多行 |
| --- | ---: | ---: |
| definition_explicit | 12 / 12 | 11 / 12 |
| term_meaning | 12 / 12 | 12 / 12 |
| room_location | 12 / 12 | 12 / 12 |
| location_vs_origin | 12 / 12 | 9 / 10 |
| responsible_person | 12 / 12 | 12 / 12 |
| founder_vs_director | 7 / 7 | 9 / 9 |
| opening_date | 12 / 12 | 12 / 12 |
| time_scope | 12 / 12 | 12 / 12 |
| direct_count | 12 / 12 | 12 / 12 |
| reservation_not_shipment | 12 / 12 | 0 / 0 |
| matching_dimension | 12 / 12 | 12 / 12 |
| unit_attached | 12 / 12 | 7 / 7 |
| recorded_zero | 12 / 12 | 12 / 12 |
| zero_vs_missing | 12 / 12 | 12 / 12 |
| explicit_prohibition | 12 / 12 | 12 / 12 |
| negative_qa | 12 / 12 | 4 / 5 |
| explicit_permission | 12 / 12 | 12 / 12 |
| feature_presence | 12 / 12 | 12 / 12 |
| complete_list | 12 / 12 | 12 / 12 |
| item_membership | 12 / 12 | 12 / 12 |
| first_step | 7 / 7 | 4 / 4 |
| ordered_steps | 0 / 0 | 2 / 4 |
| explicit_cause | 11 / 12 | 12 / 12 |
| purpose_not_cause | 0 / 0 | 4 / 4 |
| declared_range | 10 / 10 | 12 / 12 |
| exact_vs_approximate | 12 / 12 | 12 / 12 |
| model_suffix | 12 / 12 | 12 / 12 |
| region_binding | 12 / 12 | 9 / 9 |
| mode_binding | 12 / 12 | 12 / 12 |
| withdrawn_value | 0 / 0 | 1 / 1 |
| two_objects | 12 / 12 | 12 / 12 |
| report_both_records | 0 / 0 | 8 / 7 |
| header_and_row | 3 / 3 | 10 / 10 |
| wrong_column | 0 / 0 | 0 / 0 |
| english_fact | 12 / 12 | 12 / 12 |
| cross_language_fact | 12 / 12 | 12 / 12 |
| explicit_antecedent | 11 / 11 | 12 / 12 |
| distractor_then_fact | 0 / 0 | 0 / 0 |
| untrusted_instruction | 0 / 0 | 1 / 1 |
| explicit_condition | 0 / 0 | 0 / 0 |
