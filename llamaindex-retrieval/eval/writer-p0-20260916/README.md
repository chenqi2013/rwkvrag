# Writer P0 对照实验

结果与限制见 [实验报告](../../../docs/p0-writer-experiment-20260916.md)。本目录的候选未获准作为默认配置。

- `build_holdout.py`：生成作者控制的新编 8 题组与带原文坐标的 gold。
- `holdout.jsonl`：首次调用前冻结；补测时已暴露，后续使用属于回归测试。
- `run_comparison.py`：按原冻结计划运行四组 legacy 模板对照；用于重放旧实验，不能当作应用 canonical 配置评测。
- `summarize_comparison.py`：核对每组复核、输入与配置绑定，分别报告开发和留出结果。

运行需要模型服务，禁止覆盖既有输出目录。先阅读脚本 `--help` 和冻结计划；重放需要匹配计划绑定的源码版本。现有运行的报告可从 `llamaindex-retrieval` 目录执行：

```bash
.venv/bin/python eval/writer-p0-20260916/summarize_comparison.py \
  --run-root ../data/quality-runs/20260916-writer-checked-v1 \
  --output /tmp/writer-comparison-new.json
```

新基线应使用 [质量验收入口](../native-smoke/QUALITY_GATE.md) 中的显式 canonical 模板参数。内容提示词 `evidence_first` 与传输模板 `rwkv_g1j_no_think_v1` 是两个独立选项。源码和材料冻结、审核记录绑定、8 题分母以及模型原文保留都不能省略。
