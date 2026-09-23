# 内置网络提供方失败诊断与V5页面（2026-09-23）

内置Tavily/SearXNG适配器的上游失败原先仅在检索trace里显示异常类型。现在由`WebProviderError`提供固定安全错误码：例如Tavily返回401时，`retrieval.provider_failures[].error_code`与`generation.retrieval_failures[]`保存`web_upstream_http_401`，页面提示HTTP状态。错误响应正文和密钥不进入运行记录；只认识本项目的安全错误类，不接受任意异常对象提供的字符串。

模拟HTTP回归覆盖了提供方失败、混合检索只剩知识库时的部分失败状态、trace中的安全错误码与UI提示。相关后端143项、前端47项测试通过，TypeScript/Vite构建通过。本机18440实际切到独立目录`answer-format-20260923-v5`，新JS、API、MongoDB和OpenSearch健康；[发布收据](../../../artifacts/answer-format-20260923/DEPLOYMENT-v5.json)保留SHA。正式2.9B模型、原SearchReader提供方和旧历史均未切换或改写。

一次真实Tavily探测仍是401，不能把模拟路径或UI提示当成可用的外网检索，也不能据此宣称多项目回答质量通过。[V2/V3逐层结果](layered-oracle-v3-results-20260923.md)显示最终Writer与部分硬条件仍未达标。
