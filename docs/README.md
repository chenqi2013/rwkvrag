# 文档索引

维护日期：2026-09-20。默认先读“当前状态”，只有需要核对某次结果时再打开历史报告。

## 当前工作

| 文档 | 用途 | 状态 |
| --- | --- | --- |
| [CURRENT](CURRENT.md) | 代码、运行配置、最近已完成评测、下一步 | 当前状态入口 |
| [EXPERIMENTS](EXPERIMENTS.md) | 单变量约束、冻结和评分规则 | 当前实验约束；下一轮尚未执行 |
| [EVIDENCE](EVIDENCE.md) | 单项证据记录与 API 契约 | 已实现，独立预览验证过 |
| [ASSESSMENTS](ASSESSMENTS.md) | 原文与模型判断分离的接入设计 | 设计草案，尚未接入线上 |
| [架构规则](../llamaindex-retrieval/ARCHITECTURE_RULES.md) | 代码与模型的职责边界 | 开发约束 |

## 使用和工具参考

| 文档 | 范围 |
| --- | --- |
| [服务说明](../llamaindex-retrieval/README.md) | 配置、API 与代码入口 |
| [本地运维](../llamaindex-retrieval/deploy/local/README.md) | 服务管理与健康核对；不替代运行状态检查 |
| [Linux 模板](../llamaindex-retrieval/deploy/linux/README.md) | 另一种部署方式，不是本机状态 |
| [RWKV 传输与 State](rwkvos-api.md) | 模型身份、State 兼容、传输边界 |
| [上下文与 State](long-context.md) | 输入预算与动态状态的区别 |
| [StateTune 方法入口](statetune-experience.md) | 方法与历史训练记录的区别 |
| [训练工具](../llamaindex-retrieval/statetune/README.md) | 命令、数据与检查入口 |
| [Wiki 试问题单](wiki-rag-test-guide.md) | 2026-09-11 冻结语料上的参考问题，非盲测或当前规模统计 |
| [实验附件](artifacts.md) | 2026-09-11 归档的下载与恢复 |

## 历史报告

见 [archive/README](archive/README.md)。过期状态不再列在项目首页。报告保留原结论并标明历史身份；实验原始输出、训练数据与冻结脚本不因本次整理而修改。

## 后续维护

1. 当前状态只在 CURRENT 更新，附核对日期和证据入口。
2. 维护中的契约使用稳定文件名；提案明确写“尚未实施”。
3. 新的完成报告放到 archive 对应月份，当前页只摘要，不堆叠多个“最新”。
4. 文档迁移要修复链接；旧路径通过[迁移清单](archive/MIGRATION-20260920.json)查找。
5. 历史事实的勘误另写更正，不能改模型原始回答、运行回执或实验绑定。
