# 自动 Wiki：本地上线与验收

日期：2026-09-19。

## 已上线

本地管理页：<http://127.0.0.1:18440/admin/#/wiki>。正式本地 API 已重启加载本批代码，使用独立发布目录中的前端构建。模型服务恢复为原 RWKV7 G1j 2.9B 配置，Writer300 状态未更换。

- 上传入库成功后自动生成每文档 Wiki 草稿；文档修订入库成功后自动生成新版本。
- 每个版本绑定原文及解析快照哈希、索引版本、实际提示词；保留完整模型响应、来源、引用检查及历史版本。
- 来源修订、删除、索引回滚会影响新鲜度标记。索引与原文不一致时拒绝生成；启动补偿跳过这类文件，避免回滚后阻止应用启动。
- 自动任务按文件、解析快照和提示词去重；失败与原文入库结果分开。管理页可查看历史及手动重新生成。
- Wiki 原文不会自动成为检索证据；生成内容始终为未审核草稿。
- 本次同时部署前四批知识维护代码：索引分代发布、受校验回滚、来源快照、文档修订及恢复。

原有 5,019 个文件没有全部补做快照，不能视为已有自动 Wiki。旧文件需先重新索引，成功建立来源快照后才进入自动生成链路；未批量触发历史库重建。

## 验证证据

证据目录：`artifacts/wiki-20260919/`。

- 相关后端回归 48 项通过，包含真实隔离 MongoDB/OpenSearch 的修订、回滚、来源及 Wiki 生命周期测试。模型在这些生命周期测试中使用桩。
- 最终提示词、配置和回滚后启动补偿的定向复验记录见 `final-targeted-tests.log`；与上述测试有重叠，不能相加为独立测试数。
- 前端既有 23 项测试通过，TypeScript/Vite 构建通过。构建存在既有 bundle 大小提示。
- 隔离预览中使用真实模型、真实浏览器上传和修订：虚构设备材料从 12 V / 14 天变为 24 V / 21 天，自动生成新 Wiki，旧版本标为 source_changed，新版本 current；最终所选两个版本的四项事实与引用经人工核对。
- 上线后只读浏览器检查 Wiki 路由、接口和脚本错误；保留原有 46,051 个索引切片，健康检查正常。未向业务知识库插入虚构验收材料。

`preview-wiki-final-v1.json`、`preview-wiki-final-v2.json`、`browser-revision-qa.json` 和 `wiki-current.png` 保存最终真实模型及浏览器证据。`deployment-verification.json`、`deployed-wiki.png` 为正式本地部署检查。

## 负面结果与质量边界

初次验收因远端模型未运行而 token_count_failed，恢复原服务后已排除。较长提示词出现只生成标题的结果，保留在 `preview-wiki-response-title-only.json`；旧 `browser-qa.json` 的 draft 只表示执行和引用格式检查通过，不代表语义验收。调整为通用逐项事实提示后，另一个开发材料仍出现四项事实遗漏一项。最终选定 v1/v2 均覆盖四项，但这是已暴露开发材料，不是盲测，不能推导普遍完整率。

当前 citation 检查只判断引用格式及编号；没有自动事实完整性或语义真实性保证。本批不改变既有回答可靠性评测结论。

## 发布与恢复资料

私有发布目录：`data/services/local-app/releases/wiki-20260919/`。包含前端构建、后端源码快照、哈希、旧配置及停服期间导出的 MongoDB BSON 备份。备份中包含业务数据及配置，应保留在本地私有目录，不提交仓库。

配置仅增加/设置 `admin_static_dir` 与 `wiki_auto_generate=true`，未更换模型或状态。旧原文与索引未删除。恢复数据库必须结合索引及文件状态人工规划，不能仅回滚代码便宣称数据恢复完成。

## 尚未完成

- 跨文档主题页、自动主题合并、人工编辑/审核发布，以及知识依赖图。
- 长文分层 Wiki：目前材料超过默认 24,000 字符会明确失败，未静默截断。
- Wiki 列表与历史最多展示 100 项，尚无完整分页。
- 多实例任务租约、应用级资源授权、取消任务、远端模型开机自恢复。
- OCR、复杂表格、外部源同步、阅读后自主补充检索 Agent，及达到稳定质量门槛的回答可靠性。

因此本批为可运行且已部署的自动 Wiki 草稿闭环，不是商用质量认证。

## 重现说明

后端：在 `llamaindex-retrieval` 目录设置 `RWKVRAG_TEST_OPENSEARCH_URL=http://127.0.0.1:18438` 和 `RWKVRAG_TEST_MONGO_URL=mongodb://127.0.0.1:18439`，使用项目虚拟环境运行 `pytest -q tests/test_wiki.py tests/test_file_revisions.py tests/test_index_versions.py tests/test_source_revisions.py tests/test_schemas.py tests/test_admin_router.py tests/test_repository.py`。测试依赖本地 MongoDB/OpenSearch，使用隔离数据库和索引。

`verify_deployment.py` 是只读部署浏览器检查，需要安装 Python Playwright 与 Chromium。`browser_qa.py` / `browser_revision_qa.py` 是本次预览操作记录，固定指向隔离预览端口 18441 和验收文件；会写入测试数据，重复执行前需检查预览设置及已有文件，不能改为业务端口直接运行。
