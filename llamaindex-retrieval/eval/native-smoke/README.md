# 固定材料回归

`fixtures.jsonl`保存8道开发题、原始材料与独立判据。`run_smoke.py`调用固定材料Writer，不运行BM25检索；gold不发送给模型。题目包含已经暴露的旧题和人工场景，不能当作新盲测。

从 `llamaindex-retrieval` 目录执行离线检查：

```bash
python eval/native-smoke/run_smoke.py --validate-only
python eval/native-smoke/run_smoke.py --self-test
```

真实调用参数见 `--help`，按目标2.9B服务填写；当前batch传输的完整评测执行器在[实验附件](../../../docs/artifacts.md)中。旧13.3B成绩和历史运行条件移入附件，不能计入2.9B结果。最新经验见[StateTune总结](../../../docs/statetune-experience.md)。
