# 复读与提示输入只读核查

- [核查报告](../../docs/archive/2026-09/repetition-input-audit-20260921.md)
- [汇总及旧28题逐字prompt核对](SUMMARY.json)
- [44条首轮复读定位与第二轮一致性](LOOP-LOCATIONS.json)
- prompt-0043.txt、prompt-0097.txt、prompt-0124.txt：从冻结输入逐字导出。
- [本轮请求参数](REQUEST-PARAMETERS.json)、[来源绑定](BINDING.json)

0次新模型调用；定位不是新的语义评分。原始答案、历史评审、冻结输入和旧报告均未修改。

复现：`python3 llamaindex-retrieval/eval/repetition-audit-20260921/audit.py NEW_OUTPUT_DIRECTORY`。依赖仓库已归档tar、输入和评审，不需要启动模型。
