# RWKVRAG

基于 RWKV、OpenSearch BM25 和 MongoDB 的知识库问答项目。模型负责语义判断，程序负责检索组织、任务预算、原文与版本保存、传输和校验。模型原始输出保持不变。

## 使用与开发

- [服务安装、配置和 API](llamaindex-retrieval/README.md)
- [本地服务管理](llamaindex-retrieval/deploy/local/README.md)
- [Linux 部署模板](llamaindex-retrieval/deploy/linux/README.md)
- [架构规则](llamaindex-retrieval/ARCHITECTURE_RULES.md)
- [证据接口](docs/EVIDENCE.md)
- [并行问答、原文对比与网络检索展示页](llamaindex-retrieval/INTERACTIVE_TRIAL.md)（`/studio/`）

```bash
cd llamaindex-retrieval
uv sync --frozen --extra dev
uv run pytest -q
```

模型端点、State 和实际启用的协议由运行配置决定。部署前需验证真实检索、答案与引用；代码和单元测试不代表模型质量验收。

## 许可证与实验性功能

仓库自身代码按 [MIT 许可证](LICENSE) 提供，使用和再分发时保留相应版权与许可声明。第三方依赖、模型权重和推理运行时各自适用其许可证，不能由本仓库的 MIT 统一覆盖。例如应用依赖 PyMuPDF，其许可为 GNU AGPL v3 或 Artifex 商业许可；请按实际使用、服务及分发方式核对要求。已有第三方声明见 [LICENSES](llamaindex-retrieval/LICENSES)。

源码可供开发与实验使用，不等于完整模型服务开箱可用。[统一检索候选](llamaindex-retrieval/UNIFIED_RETRIEVAL.md)默认关闭；显式选择 `span_ids_v1` 可使用已接入的片段ID阅读、来源继承、三叉归并和根JSON展示，quote阅读仍是独立对照组件。另显式选择 `control_protocol=review_cycle_v1` 可启用Review后重新选渠道、分轮预算及独立计划修订；失败记录不覆盖，也不保证模型修订成功。新流程需要兼容的Schema解码后端；代码与展示能力不代表事实支持、完整检索或生产就绪。模型端点、后端兼容性、索引和数据库须按部署说明自行配置，评测集与模型不随仓库提供。

## 仓库发布范围

GitHub 仅维护应用源码、前端、部署配置、必要依赖、应用单元测试与使用/接口说明。实验脚本、训练数据、评测集、模型产物和阶段报告不进入提交。

本地 `scripts/`、`data/`、`llamaindex-retrieval/eval/`、`llamaindex-retrieval/statetune/` 及阶段文档由 `.gitignore` 排除；不使用 `git add -f` 绕过。历史产物目录 `artifacts/` 已清理。完整研究工作区需要另行移交，不能从应用仓库推断实验状态。
