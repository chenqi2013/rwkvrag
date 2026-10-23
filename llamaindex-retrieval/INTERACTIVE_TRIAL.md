# 上下文与并发漏斗问答

独立 FastAPI 入口 `llamaindex_retrieval.interactive_app:app` 提供 `/admin/#/try`，连接已有知识库、RWKV 原生推理服务及 Web 提供方。

## 原文摘抄扁平流程（`pipeline: "flat"`，唯一试用流程）

一次规划（判断新问题/追问/更正）、每份资料每个窗口一次逐字摘抄阅读、一次终答（结论+要点+未知项）、一次独立核对。
旧parallel/lean/needs/spans/funnel试用流程已删除；源码备份在本地研究目录，不随应用发布。
可选 `flat_options`：`window_chars`（2400）、`max_windows_per_source`（4）、`snippet_batch`（4）、`deadline_seconds`（180）、
`acquisition_timeout`（70）、`final_quote_limit`（40）、`per_need_limit`（12）、`supplement`（true）、`review`（true）。
`flat_title_options`：`read_all_below`（12）、`limit`（6）。Web提取并发由`web_provider_parallel`（默认1）和`web_search_concurrency`控制。
多轮：请求带`parent_job_id`即接续；服务按父链取最近3轮的用户原话、整理后问题、所问对象和上一轮回答首句，并在追问时重读上一轮摘抄过的页面。
公共`/v1/models`返回5xx时不阻止生成：上下文上限退回配置值，每次生成仍校验响应中的model。

## 上下文与资料

当前问题先交给 RWKV 整理。显式传入 `parent_job_id` 时，模型结合上一轮整理后的完整问题及上一轮回答理解指代和更正；上一轮回答仅帮助理解，不作为事实依据。没有父任务时从空上下文开始。历史原始记录独立保存，不把整段历史无限追加给模型；整理失败如实返回，不用程序猜出替代问题。页面“新的提问”清空当前父任务。

知识库分页读取全部目录，每批128个标题交给模型选择，再下载所选文件的 UTF-8 原文。128仅为提示分批大小，不限制选中文档数量。Web获取保留提供方返回的正文或摘要快照、URL、时间及抓取状态。不调用本地词项/向量相关度评分，不把标题当事实证据。

原文使用已有逐字结构分块器：连续正文4000字符软窗口、240字符重叠，保留坐标与哈希；表格、列表、代码和QA保持结构原子及父级上下文。所有块排队并发总结，不按前N块截断。超长原子无法满足单次输入预算时明确记录失败，不静默截断。

每个归并节点最多接收3个直接子输出；同层分组并发，再进入下一层。总文档数、总子任务数和总层数不设固定上限。归并仅接收本组模型raw及节点标识。

可选候选：终答（`interactive-funnel-v2`，默认关闭，传`original_chars`启用；2026-10-19 TEST24配对6/24→4/24未过门槛）另外接收叶层被模型选中的逐字原文块：沿归并链收集模型实际采用的叶，按选择顺序装入，输入证据预算单独配置（默认16000字符）；超出预算的叶不截断，记入`omitted_original_leaf_ids`。未被叶选中的原文块、失败叶和全量来源图不进入终答。提示要求事实以原文为准，总结只帮助定位。

上下文、叶总结、归并和终答分别保留输出预算1024/768/1024/2048 token；每次真实模型调用前通过服务tokenizer校验完整输入加输出预留。并发数只限制同时执行的数量。补查循环最多300秒，请求含队列最多900秒，均为资源时限，超时明确记录。

## Web记录与离线重放

`settings.web_search_provider` 可设为 `"anysearch"`，同时设置
`web_anysearch_api_key_file: "/absolute/path/active.key"`（仅当前用户可读，
只包含一把Key），或用 `web_anysearch_api_key` 配置单Key；两者不能同时设置。
默认仍为Tavily。AnySearch先调用 `/v1/search`，再按返回顺序用同一Key
调用 `/v1/extract` 获取正文。只把extract成功的正文标为`fetched`；
无法提取的页面保留原始摘要，标为`snippet_only`并记录获取错误。
搜索和正文请求共享限流与认证/额度冷却，不自动重试、轮换或匿名回退。
`web_anysearch_extract_timeout` 默认8秒，限制每页提取（包括等待共享Key）；
整个搜索获取仍受`web_search_timeout`约束。慢页超时或提取预算耗尽时保留
已获取正文，其余页面显示原摘要和明确的获取错误，不因一个慢页丢弃所有结果。
每页提取时限参与AnySearch重放键；更改它不会误用旧时限的记录。
单页上游5xx故障保留其他页面和该页摘要，共享冷却仍生效，不在请求内重试。
AnySearch没有文档化的`include_domains`参数，域名限制只过滤已返回URL的
主机及其子域名，原始查询不改写；不能保证限定域名内的完整召回。
录制包含搜索与成功正文获取的原始响应字节和哈希，重放不调用二者的接口。

顶层 `web_replay: {"mode": "off"|"record"|"replay", "store": "/absolute/path/captures"}` 控制Web获取，默认 `"off"`。`record` 按提供方、精确查询、按原顺序的 `include_domains`、结果数和 `full_snapshot` 保存完整提供方响应及哈希；同一请求重用首次记录。`replay` 只读对应记录，缺失、损坏或不可读均记为获取失败，绝不转为联网请求。原始HTTP响应字节和解析后的提供方载荷均保留在记录中，供审计复核。记录目录包含来源正文，应按本地研究数据管理，不提交版本库。

录制也保存首次上游5xx、网络故障和获取超时的安全错误码；重放同样抛出该故障，
不会把它当成零结果或重新联网。失败记录不含可能泄露凭证的上游错误体。
认证、额度、频率限制、凭证配置与存储故障不作为可重放来源记录，调用方应停止该轮。

## 原始输出和来源

模型决定选择、总结和补查问题，代码只调度并校验结构/ID。每次raw保持不变，失败raw保留在审计记录中，只将其失败状态传给父节点。不能由Schema通过、合法引用或页面非空推断答案正确。

最终模型输出中的`source_ids`可以指向本次直接子节点或终答收到的原文块，`generation.citation_bindings`将它们映射到原文来源。页面显示的是**来源链**，不是已完成事实核验的承诺；可打开原文核对。模型总结会有遗漏或误读，仍须单独评估最终回答。

## 启动

前端在`web`目录运行`npx tsc -b`和`npx vite build --outDir /absolute/path/trial/web`。准备仅当前用户可读的JSON配置：

```json
{
  "settings": {
    "rag_pipeline": "rwkv",
    "generation_output_mode": "immutable",
    "native_transport": "native",
    "native_completion_protocol": "g1j_plain",
    "native_require_model_identity": true,
    "native_require_fp32_runtime": true,
    "native_planner_prefill": "<think></think",
    "native_resolver_prefill": "<think></think",
    "native_writer_prefill": "<think></think",
    "native_max_concurrency": 32,
    "native_base_url": "http://127.0.0.1:18670/v1",
    "native_context_window_tokens": 25600
  },
  "admin_base_url": "http://127.0.0.1:18440",
  "records_dir": "/absolute/path/trial/records",
  "gpu": 2,
  "expires_at": null
}
```

还需补充实际`native_model`、`native_api_key`、兼容的`native_state_routing`角色映射和Web配置；示例不是可直接连接任意模型的完整凭证配置。试用客户端明确使用每次请求独立的HTTP连接，不加载材料直答编排或旧对照配置。

```bash
export RWKVRAG_TRIAL_CONFIG=/absolute/path/trial/config.json
export RWKVRAG_TRIAL_STATIC=/absolute/path/trial/web
PYTHONPATH=src .venv/bin/python -m uvicorn llamaindex_retrieval.interactive_app:app \
  --host 127.0.0.1 --port 18671
```

API key不进入前端或版本库。本机试用入口无用户认证，应绑定回环地址。请求之间串行调度，避免不同请求混合Reader/Writer State；一个请求内部的资料总结及同层归并并发执行。

## 接口与记录

- `GET /v1/try/meta`：知识库选项、配置模型、并发数和试用截止时间。
- `POST /v1/try/ask`：`question`、`retrieval_mode`（`hybrid`/`knowledge_base`/`web`）、`knowledge_base_ids`、`supplementary`（默认true）、可选`parent_job_id`及`pipeline`（仅`flat`）。返回任务ID。
- `GET /v1/try/jobs/{id}`：真实阶段、上下文整理结果、最终回答及来源图。
- `POST /v1/try/jobs/{id}/cancel`：取消应用任务及其等待中的调用，不能保证已提交的远端生成立即停止。

请求在`records_dir`保留原始问题、模型prompt/raw/hash/耗时、原文快照、分块坐标、归并图与失败状态。服务重启后可重新打开已完成记录。目录按本地数据管理要求保存，不提交到应用仓库。新部署使用独立记录目录与浏览器存储，不自动载入旧直拼试问。
