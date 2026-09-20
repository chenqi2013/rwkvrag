# RWKVRAG

基于 RWKV、OpenSearch BM25 和 MongoDB 的知识库问答项目。模型负责语义判断，程序负责检索组织、任务预算、原文与版本保存、传输和校验。模型原始输出保持不变。

## 从这里开始

- **[当前状态与下一步](docs/CURRENT.md)**：区分已实现、运行配置、实验结果和未实施设计。
- **[实验变量控制](docs/EXPERIMENTS.md)**：下一轮改变什么、固定什么、怎样计分。
- [完整文档索引](docs/README.md)：按用途查阅，不按文件新旧猜状态。
- [架构规则](llamaindex-retrieval/ARCHITECTURE_RULES.md)：开发约束。

当前工作集中在短证据定位、属性与条件匹配、否定和零值的可靠判断。单项证据核对已实现；标准化事实、独立判断记录、冲突关系与阶梯总结处于不同阶段，详见当前状态页。

## 使用与开发

- [Python 服务与接口](llamaindex-retrieval/README.md)
- [本地服务管理](llamaindex-retrieval/deploy/local/README.md)
- [Linux 部署模板](llamaindex-retrieval/deploy/linux/README.md)
- [StateTune 工具入口](llamaindex-retrieval/statetune/README.md)

```bash
cd llamaindex-retrieval
uv sync --frozen --extra dev
uv run pytest -q
```

模型端点、State 与部署状态以实际配置和健康检查为准；代码存在不代表运行服务已启用，测试能执行不代表质量通过。

## 历史资料

[历史报告索引](docs/archive/README.md)保留阶段验收、失败结果和旧设计。[实验附件](docs/artifacts.md)说明旧数据及权重的恢复方式。历史报告不作为当前默认配置或下一步任务。
