# 属性证据第一阶段测试报告

> **历史记录，非当前状态。** 本文保留当时的实验、部署或设计结论；“当前”“最新”“下一步”均指原记录时点。当前事实与行动以[当前状态](../../CURRENT.md)为准。原路径：docs/atomic-evidence-test-report-20260920.md。

## 结论

第一阶段的检索、短证据保存、版本绑定、历史回看和前端原文定位已实现，可在独立预览中检查。**模型的语义质量门槛尚未通过，不应接入自动冲突裁定、最终答案或自动 Wiki 发布。** 当前展示原文及模型标签，没有宣称已经抽出可信的标准化事实。

正式 18440 服务没有切换到此版本。GitHub 修改前备份为 `68b6bf4a6587ad1d446d70634d98299bdb54ab0e`；本次实现在 `chase/atomic-evidence-v1` 分支。

## 最终版本验证

| 验证 | 结果 | 含义 |
| --- | --- | --- |
| 新增后端单元与真实 Mongo/OpenSearch 集成 | 21 通过 | 原文位置、来源冲突、跨库范围、历史不可覆盖、源更新保留历史、预算、超时、取消、非法模型输出 |
| 现有相关回归 | 140 通过 | Repository、文件修订、来源版本、Reader、RWKV pipeline、Wiki |
| 前端 Node 测试 | 33 通过 | 含 Unicode/emoji 原文定位与错位拒绝；其中 2 项为本次新增 |
| 前端 TypeScript/Vite 构建 | 通过 | 仍有既有的大 bundle 提示 |
| 六类基础材料及变体 | 12 次执行完成 | 保存冲突、条件、修订、零值、未记载和长材料的原文；不等同于 12 次自动事实裁定正确 |
| 后续新增材料 | 8 次执行完成 | 仍发现 3 类语义问题，见下文 |
| 最终固定材料模型调用 | 78/78 以 stop 结束 | 此批没有输出截断；不能由小样本证明所有问题都不会循环 |
| 最终证据位置与哈希审计 | 45/45 一致 | 保存原文、SHA-256、Unicode 起止位置一致；这不验证内容真伪或语义相关性 |

浏览器使用真实预览 API，未拦截或伪造返回：提交查询→OpenSearch→7.2B 选片段→2.9B Reader→MongoDB→点击两份冲突原文→刷新打开历史→切换知识库清除旧记录。另检查 390px 手机布局、抽屉原文进入视口、无横向溢出和 JavaScript 错误。记录见 [browser-qa.json](../../../artifacts/atomic-evidence-20260920/browser-qa.json)，截图在同目录。

## 仍未通过的语义门槛

1. **地域范围误判**：查询 2026 年中国大陆售价时，Reader 同时把美国价格标为 reader_supported。年份不同的价格被标为 unconfirmed，但地域条件仍有漏洞。
2. **表头误选**：表格正文两行都保存了对应模式与时长，但 Reader 也把只有“模式 / 续航”的表头标为支持证据。结构上下文完整不代表它足以回答。
3. **QA 否定漏判**：最初选择器仅选了问题句、遗漏否定答案。v6 把相邻问答作为同一单元后，完整的“不支持，必须保持网络连接”得以保存，但 Reader 仍标为 unconfirmed。定位修复了，判断能力尚未修复。

其他新增材料中，单位错误撤回语句保持完整，实际零值未被改成预约数量，“未公开”原文未变成 0，恶意指令样例未选为设备参数，长记录末尾 9 秒证据保留。上述是对特定材料的观察，不能外推为对应问题已经全面解决。

## 实验过程与选择依据

| 版本 | 调整 | 观察 |
| --- | --- | --- |
| v1 | 模型直接复制主张值和条件 | 空格变化、无效条件字段、预约量混淆；长文末尾漏检 |
| v1-ranked | 只改源内词项排序 | 末尾材料进入候选，复制值仍不稳定 |
| v2 | 短语编号代替自由复制 | JSON 形状仍失控，不能算通过 |
| v3 | 明确 JSON 示例 | 执行完成但零值/未记载分类错误 |
| v4 | Reader 判断属性支持，7.2B 只选短语 | 属性匹配有所改善，但短语会只选中“正式更正：”；记录含一次 stop_tokens 配置错误的失败运行 |
| v5 | 保存完整 Reader 核对片段，取消值标准化与缺失推断 | 阻止截丢关键原文；NO 仅为 unconfirmed；仍有模型相关性错误 |
| v6 | 仅把相邻 QA 合成一个可选单元 | 否定答案不再因问题单独入选而丢失；Reader 漏判仍保留 |

各次运行输出分别保存，没有覆盖或修补旧模型回答。开发过程中固定了材料与调用，分开记录局部调整；这不是严格随机对照实验，不能将多阶段调整归因为单一模型提升。8 个新增用例在首次结果出炉后已成为开发可见材料；v6 是回归复测，不能再称为未见盲测。

历史代码快照、摘要与配置哈希在 [artifacts/atomic-evidence-20260920](../../../artifacts/atomic-evidence-20260920)。最终 v6 的逐调用 prompt、原始输出、来源和位置随测试材料一起保存在 `v6-final` / `v6-holdout` 子目录。完整 HTTP 收据另存于被 Git 忽略的本地 `data/quality-runs/atomic-evidence-20260920`。

## 如何复测

依赖项目虚拟环境、可访问的 MongoDB/OpenSearch，以及本机 18425 的 7.2B batch 服务和 18423 的 2.9B Reader 服务。不得把 2.9B 的 State 加载给 7.2B。

```bash
RWKVRAG_TEST_MONGO_URL=mongodb://127.0.0.1:18439 \
RWKVRAG_TEST_OPENSEARCH_URL=http://127.0.0.1:18438 \
llamaindex-retrieval/.venv/bin/python -m pytest \
  llamaindex-retrieval/tests/test_atomic_evidence.py \
  llamaindex-retrieval/tests/test_atomic_evidence_integration.py -q

PYTHONPATH=llamaindex-retrieval/src llamaindex-retrieval/.venv/bin/python \
  llamaindex-retrieval/eval/atomic-evidence-20260920/run.py \
  --output data/quality-runs/atomic-evidence-20260920/new-unique-run

PYTHONPATH=llamaindex-retrieval/src llamaindex-retrieval/.venv/bin/python \
  llamaindex-retrieval/eval/atomic-evidence-20260920/run.py \
  --fixtures llamaindex-retrieval/eval/atomic-evidence-20260920/holdout.jsonl \
  --output data/quality-runs/atomic-evidence-20260920/new-unique-holdout
```

输出目录必须不存在。模型测试为固定材料→实际选择/Reader 的测试；浏览器测试另覆盖真实检索接入。任何 completed 都只表示流程结束。

## 下一步

先补充独立标注的对象、时间、地域、否定与表格行支持判断数据，并冻结新的未见验收集；再比较 Reader 的提示/StateTune 方案。验收要同时统计误选、漏选和条件保留，不能只看生成是否停止或 JSON 是否有效。通过后再推进规范值原子化、分子关系和分层总结。
