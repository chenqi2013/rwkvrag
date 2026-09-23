# 7.2B StateTune新GitHub题成对结果

2026-09-23完成[冻结输入](../../../artifacts/state-progression-20260922/EVAL-PINS-v1.json)中的24题、零State/训练State各两轮，共96份原始回答。[全部原文、运行收据、逐份审读、机械与语义汇总](../../../artifacts/state-progression-20260923/fresh-v1/MANIFEST.json)已存档；本机[逐题可读结果页](http://127.0.0.1:18440/admin/experiments/state-fresh-github-20260923.html)显示完整回答与所给README材料。评测输入哈希为`c87b367510c54faffecc9a303eeb080092483a282a329e0717dd3932a3a991bb`，训练完成收据哈希为`8abe88dec5a2a7bfe44ef827dd319bab956c8960b86d2d2bd51a3460be92d18d`。两组只改变角色初始State；基座7.2B、提示、材料、贪心选词、输出上限均相同。第二轮反转组间调用顺序，全部48个“题×组”回答的原文、token和结束原因与首轮相同。

## 质量结论

按预先冻结的[语义审读标准](../../../llamaindex-retrieval/eval/state-fresh-github-semantic-v1/RUBRIC.md)，实现者逐份检查96份回答并绑定原文SHA。[逐条判决和理由](../../../artifacts/state-progression-20260923/fresh-v1/MANUAL-REVIEW.jsonl)及[汇总](../../../artifacts/state-progression-20260923/fresh-v1/SEMANTIC.json)可复查。每轮12道大型比较题，零State与训练State均为0道完整正确、12道错误：有材料错绑、虚构引文、编造来源编号和反复输出。**本轮State未解决主要目标，不晋级正式模型。**

每轮12道单项目普通题，零State完整正确6、错误6；训练State完整正确10、错误2。五道题在两轮都从错误转为完整正确：aiohttp、urllib3、DuckDB、Arrow和Ruff各一题；Polars普通题从完整正确退步为复读触顶。这个普通题收益和退步必须一起报告，不能把全题混算掩盖比较题失败。此处“完整正确”是实现者按给定README判读，未经独立盲审，也不是已部署用户准确率。

## 机械诊断

| 每组两轮合计24份比较、24份普通回答 | 零State | 训练State |
| --- | ---: | ---: |
| 比较题触及2048-token上限 | 14 | 18 |
| 普通题触及2048-token上限 | 8 | 4 |
| 全部48份中出现不存在的资料编号 | 8 | 10 |
| 全部48份中检出重复长片段 | 30 | 24 |

训练组相对零组新增4项无效引用、4项重复长片段、6项由正常结束变成未正常结束的配对诊断；[逐项清单](../../../artifacts/state-progression-20260923/fresh-v1/MECHANICAL-ISSUES.jsonl)保留。机械指标没有检查事实和引用语义；重复片段总数减少也不能抵消比较题更多触顶和错误。

具体失败可在结果页展开原文：Requests/HTTPX/aiohttp/urllib3比较题把Requests描述复制给多个来源，随后引用本题不存在的`资料 5`及后续编号；HTTP/2题把未获证据支持的HTTP/2能力归给Requests；Ruff/Black/isort/mypy题出现“Ruff完全替代mypy”的错误结论或把Ruff语句错归其他README。Polars普通题零State能收束，训练State则反复列举特性直至上限。即使某个输出正常停止，仍可能有错引文或错误归属。

## 边界与后续

这些题目只提供冻结的GitHub README片段，不运行实时网络检索、知识库召回或Wiki维护。实现者审读不是独立质量认证；第二轮逐字相同只说明这次调用次序下无漂移，不增加新题覆盖。格式转换层已让本机页面隐藏编号越界和明显复读的回答，原文仍可核对；它无法判定已存在编号的结论是否真的得到支持，不能把展示过滤算作模型能力提升。旧188题、旧434题和36题两变体、来源分离开发/留出题的顺序回归仍在GPU3执行；全部完成前不宣称普通能力稳住，也不切生产。
