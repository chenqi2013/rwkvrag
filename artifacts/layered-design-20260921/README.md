# 分层设计的已见回放材料

设计准备资料，不是新模型实验结果。模型调用0、训练更新0。

- MANIFEST.json：全部160对、320份历史result的身份和SHA256，失败均保留。
- FOCUSED-REPLAY.json：20对故障定位/普通题保护案例，含原始输入、上游快照、原答案和884份trace绑定。不可当新留出集或新gold。
- INTEGRITY.json：两个输入JSON的内容哈希。
- verify.py / CHECKS.json：只读成员、字节哈希、快照和文档ID检查；不评判语义。

源文件恢复方式写在JSON metadata，历史归档在4482f3ad。恢复后运行：

```bash
python3 artifacts/layered-design-20260921/verify.py --source-root /path/to/restored/run4 --plan docs/LAYERED_REPLAY_PLAN.md
```

[协议设计](../../docs/LAYERED_REASONING_DESIGN.md)和[实验方案](../../docs/LAYERED_REPLAY_PLAN.md)说明输入/输出、责任边界及尚未冻结的执行参数。本材料包未修改历史原文、答案、提示或State。
