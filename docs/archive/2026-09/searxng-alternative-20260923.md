# Tavily 停用后的 SearXNG 本机替代（2026-09-23）

## 部署与范围

本机使用 [SearXNG 官方容器镜像](https://github.com/searxng/searxng/pkgs/container/searxng)固定 digest 部署在 `127.0.0.1:18448`，开放官方 [JSON 搜索接口](https://docs.searxng.org/dev/search_api.html)。`docker compose` 文件和无密钥模板放在[项目部署目录](../../../llamaindex-retrieval/deploy/searxng/README.md)；随机服务密钥、本机代理地址和运行配置在 Git 忽略的私有文件中。容器使用 Linux host 网络访问本机既有 HTTP 代理，仅监听回环地址，不公开服务。

本机 18440 的 `web_search_provider` 已从 Tavily 切为 `searxng`，`web_searxng_base_url` 指向 `127.0.0.1:18448`；原自动联网模型选择器和跨进程 `web_guard_path` 保持配置。旧 Tavily Key 文件留在私有数据目录作回滚参考，当前提供方不会读取它。

## 实测

- 最初容器没有可用外网出口：JSON HTTP 200，但 `results=0`，五个上游搜索引擎超时。接入本机代理后，同一 FastAPI GitHub 查询返回 30 条结果，首条为官方仓库。代码新增：SearXNG 在零结果同时报告引擎故障时，标为安全的提供方错误，不能当作“没有匹配资料”。
- 内置适配器实际获取两条网络材料和两个快照，均标为 `snippet_only`。正式 `/v1/search` 强制 `web` 返回两条 GitHub 来源、无提供方失败；`hybrid` 返回 8 条知识库和 2 条网络来源、无提供方失败。
- 一条完整 `/v1/ask` 网络提问“FastAPI 的官方 GitHub 仓库是什么？”约 16.2 秒返回，正文给出 `https://github.com/fastapi/fastapi[资料 1]`，两条来源，`generation.status=completed`，无提供方失败。这是运行链路冒烟检查，不是大题质量验收。
- 一条自然三项目比较查询“vLLM、SGLang 和 llama.cpp 各适合什么场景？”的 SearXNG 原始搜索返回 38 条，但前列主要是知乎、掘金、CSDN 等二手比较，且一个引擎报告 CAPTCHA；当前内置适配器只拿网页摘要，不取完整正文。这说明可找到资料，**不证明**能支撑准确的多项目选择。
- 电脑重启后，API 与容器仍在运行，`/health` 和 `/v1/admin/health` 均为 200/ok，SearXNG JSON 搜索重新返回 32 条结果。重新调用 18440：`web` 返回两条网络来源，`hybrid` 返回 8 条知识库加 2 条网络来源；一条“FastAPI 的最新版本是什么？”的 `auto` 请求由已配置选择器判为 `hybrid`，两类来源均返回且无提供方失败。这只证明这一条路由与检索接线；重启后的完整多项目回答仍未重新验收。

## 当前判断

SearXNG 已恢复无需付费 Key 的真实联网检索和知识库+网络组合，可作当前运行提供方。[脱敏部署收据](../../../artifacts/searxng-alternative-20260923/DEPLOYMENT-v1.json)记录镜像与配置哈希、服务状态和测试边界。可用资料与排序取决于它聚合的搜索引擎；摘要材料与对比题需要的逐项目官方正文之间仍有明显差距。后续应单独验证按项目覆盖、正文获取、引用支持和复杂比较答案，不把一次简单问答成功扩大为语义质量结论。

若考虑付费备份，[Brave Search API](https://brave.com/search/api/) 的 Search 计划含 LLM Context，能返回抽取过的相关内容；但 Brave 官方明确提示保存 API 结果需要单独授予存储权。我们的 trace 会保存来源快照，商业部署前必须确认相应授权。当前不需要购买 API 或更多 Tavily Key。
