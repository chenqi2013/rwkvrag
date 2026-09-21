# R1固定上游State对照结果

冻结ee68218c；141个共同输入，两组双轮564次调用全部完成。训练组每轮25次非法证据ID，zero每轮0；训练组1项跨轮输出改变，zero无变化。没有上线晋级或新训练。

- `raw-run1.tar.gz`：完整572文件，含每次请求/响应、模型原文和State绑定/清理。
- `RAW-MANIFEST.json`：归档逐文件哈希。
- `review1/REVIEW-PACKET.json`：全部输入期望和四份对应输出。
- `review1/SEMANTIC-REVIEW.json`：564项绑定原文SHA的实现者审读记录；非独立准确率。
- `review1/STRUCTURAL-SUMMARY.json`：机械可验证的ID/跨轮差异结果。
- `review1/REVIEW-SUMMARY.json`：事后描述性分级，不能代替未见或完整检索评测。
- `CLEANUP.json`：实验服务已停止，原搜索路由未停，GPU显存恢复。

在新目录解压归档并逐项核对RAW-MANIFEST。精确输入、运行前语义期望和执行器在`llamaindex-retrieval/eval/layered-state-replay-20260921`。完整结论见[报告](../../docs/archive/2026-09/layered-controlled-replay-results-20260921.md)。
