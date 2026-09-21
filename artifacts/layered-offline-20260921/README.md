# 离线验证（无模型调用）

`run1/REPORT.json`为CPU审计汇总；`FOCUSED-INPUTS.json`绑定20题的884份精确调用，`ASSESS-FINDINGS.json`保留问题定位，`SOURCE-HASHES.json`登记本次读取的文件哈希。完整原始输入从旧归档恢复，方法见[设计材料](../layered-design-20260921/README.md)。索引不是可直接发送的请求；旧State引用已经失效。

从仓库根目录执行，输出必须使用新的不存在目录：

```bash
PYTHONPATH=llamaindex-retrieval/src llamaindex-retrieval/.venv/bin/python \
  llamaindex-retrieval/eval/layered-offline-20260921/verify.py \
  --source-root data/quality-runs/g1j72-state-layered-2k-20260920/run4 \
  --focus artifacts/layered-design-20260921/FOCUSED-REPLAY.json \
  --output /tmp/layered-offline-new-run

PYTHONPATH=llamaindex-retrieval/src llamaindex-retrieval/.venv/bin/python -m pytest \
  llamaindex-retrieval/tests/test_choice_contract.py \
  llamaindex-retrieval/tests/test_comparison_contract.py \
  llamaindex-retrieval/tests/test_offline_replay.py -q
```

53项测试通过。选择契约来自ac604fbf（29项），另24项覆盖比较依赖及历史回放。未创建新模型答案，未将历史旧协议猜测转换成新四状态协议。R0/R1调用计数是后续实验预算，不是已完成调用。
