# RWKVRAG

基于 RWKV、OpenSearch BM25 和 MongoDB 的知识库问答项目。模型负责语义判断，程序负责检索组织、任务预算、原文与版本保存、传输和校验。模型原始输出保持不变。

## 使用与开发

- [服务安装、配置和 API](llamaindex-retrieval/README.md)
- [本地服务管理](llamaindex-retrieval/deploy/local/README.md)
- [Linux 部署模板](llamaindex-retrieval/deploy/linux/README.md)
- [架构规则](llamaindex-retrieval/ARCHITECTURE_RULES.md)
- [证据接口](docs/EVIDENCE.md)

```bash
cd llamaindex-retrieval
uv sync --frozen --extra dev
uv run pytest -q
```

模型端点、State 和实际启用的协议由运行配置决定。部署前需验证真实检索、答案与引用；代码和单元测试不代表模型质量验收。

## 仓库发布范围

GitHub 仅维护应用源码、前端、部署配置、必要依赖、应用单元测试与使用/接口说明。实验脚本、训练数据、评测集、模型产物和阶段报告不进入提交。

本地 `scripts/`、`data/`、`llamaindex-retrieval/eval/`、`llamaindex-retrieval/statetune/` 及阶段文档由 `.gitignore` 排除；不使用 `git add -f` 绕过。历史产物目录 `artifacts/` 已清理。完整研究工作区需要另行移交，不能从应用仓库推断实验状态。
