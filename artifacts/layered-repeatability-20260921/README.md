# R0完整重复性结果

冻结3ee2ce58；1,326次调用全部记录，432/442输入三轮一致，10个变化，重复性门槛未通过。原始输出不是正确gold，不能以一致率代表准确率。

- `raw-run1.tar.gz`：完整1336文件，含run1原始请求/响应、绑定、清理记录和部署快照。
- `RAW-MANIFEST.json`：归档内每个文件的SHA256。
- `review1/`：全部重复对照、差异、实现者逐项审读和进入R1的限制。
- `INTEGRITY.json`：本材料包文件哈希。

在新目录解压原始归档并逐项核对RAW-MANIFEST；精确输入和执行器位于仓库`llamaindex-retrieval/eval/layered-repeatability-20260921`。完整结论见[报告](../../docs/archive/2026-09/layered-controlled-replay-results-20260921.md)。
