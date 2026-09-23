# 本机 SearXNG 备用网络检索

此目录使用 [SearXNG 官方容器镜像](https://github.com/searxng/searxng/pkgs/container/searxng)的固定 digest，通过 Linux host 网络仅监听 `127.0.0.1:18448`，启用官方 [JSON 搜索 API](https://docs.searxng.org/dev/search_api.html)。只作为网络检索提供方；RWKV、OpenSearch、MongoDB 仍分别配置。公开部署需要独立鉴权和限流，不应直接把这个本机配置暴露到外网。

首次启动：

```bash
mkdir -p ../../../data/services/searxng/config
cp settings.yml ../../../data/services/searxng/config/settings.yml
printf 'SEARXNG_SECRET=%s\n' "$(openssl rand -hex 32)" > .env
chmod 600 .env
docker compose up -d
curl -fsS 'http://127.0.0.1:18448/search?q=python&format=json'
```

上面的路径以本目录为工作目录。`.env` 和运行配置都位于 Git 忽略范围，不能提交密钥。升级镜像需另行固定新 digest 并验证 JSON 搜索。若需要停用，执行 `docker compose stop`。

此机外网需经本机 HTTP 代理时，在私有 `.env` 中另设 `SEARXNG_HTTP_PROXY=http://127.0.0.1:代理端口`。容器使用 host 网络才能访问本机回环代理；是否取得有效来源仍须以 JSON 中的非空 `results` 验证，HTTP 200 本身不算成功。

检索接口通过 `RWKVRAG_WEB_SEARCH_PROVIDER=searxng` 和 `RWKVRAG_WEB_SEARXNG_BASE_URL=http://127.0.0.1:18448` 接入；本机 JSON 启动入口则在 `data/services/local-app/settings.json` 设置同名小写字段。`web_guard_path` 应保持共用，以限制多个 API 实例的出口请求。SearXNG 返回的网页摘要会标为 `snippet_only`，它不能代替页面正文；比较题仍需要检查来源覆盖和引用。
