# RWKV 并行检索与原文阅读

独立 FastAPI 入口 `llamaindex_retrieval.interactive_app:app` 连接已有知识库、RWKV 原生推理服务及网络检索提供方。它提供两个页面：

- `/studio/`：随 Python 包提供的展示页，无需单独构建。默认展示16个独立聊天卡片，可添加至32个；每张卡片有输入框、来源设置、发送按钮、回答和实际进度。可逐张发送，也可一键提交所有已填写且空闲的卡片。点击“过程与原文”查看完整回答、并排原文和引用，再用“并行总览”返回。提交数量不代表模型同时运行的序列数，实际吞吐和质量须另测。
- `/admin/#/try`：原有 React 试用页，支持父任务追问与完整记录检查，需要构建前端。

不同聊天卡片之间不共享上下文；同一卡片完成后继续输入，会携带该卡片上一条已完成任务的父ID。点击“新对话”可清除该卡片下一问的父链，历史轮次仍可查看。某张卡片运行时，其他卡片可以继续编辑和发送。页面不附带训练题、预期答案或演示原文。离开或刷新页面不会取消后台任务；当前没有浏览器持久化或自动重发，取消指定任务使用试用页/API。资料对比和实时网络场景仍保留每行一个问题的批量输入方式。

## 操作示例：16个独立聊天

服务配置完成后，在浏览器打开服务地址下的 `/studio/`，按以下步骤操作：

1. 点击卡片上的来源设置，选择已有知识库、网络或两者；网络需要服务端配置可用提供方。每张卡片单独保存来源。
2. 在卡片自己的输入框中自然提问。Enter或箭头发送当前卡片，Shift + Enter换行；“全部运行”同时发送所有已填写且空闲的卡片，已提交的卡片不重复发送。
3. 观察每个聊天的实际状态。资料获取、阅读和生成完成的先后顺序可能不同；失败如实显示。完成不表示答案经过事实核验。
4. 点击“过程与原文”查看完整回答和引用依据，用“并行总览”返回。卡片内可以继续追问，历史菜单可以重新打开此前轮次。

比较论文或项目时，先将资料接入知识库，或者在卡片来源设置中填写要阅读的网页URL。网址作为获取来源单独提交，不拼接到用户问题中；问题里直接提供URL仍受支持。填写地址时获取指定页面；没有指定地址时由模型规划搜索。直接获取URL正文不等于导入整个GitHub仓库、Wiki或其所有链接；获取失败仍须检查记录。页面支持最多32个独立聊天，实际同时执行的模型请求数由服务端配置及容量决定。

也可点击“导入问题”，选择UTF-8 JSON文件。顶层为数组，每项包含以下四个必需字段，可选增加 `web_source_urls` 网页地址数组；不接受答案字段：

```json
[
  {"question": "比较知识库中的两份方案。", "retrieval_mode": "knowledge_base", "knowledge_base_ids": ["your-kb-id"], "web_include_domains": []},
  {"question": "查看最近的官方发布公告。", "retrieval_mode": "web", "knowledge_base_ids": [], "web_include_domains": ["example.org"]}
]
```

知识库ID须使用当前服务返回的实际ID。导入成功后替换当前卡片清单；有任务运行时不能导入。失败保留原卡片。空域名数组表示不限制域名。并行场景修改问题文字保留该卡片的来源设置，修改来源只影响该卡片下一次提交；已提交请求保留快照。其他两个场景使用整批输入，编辑导入文字会清除批次预设并恢复手选范围。

## 唯一试用流程：flat v8

`pipeline: "flat"` 使用 `interactive-flat-quotes-v8`：一次规划、取材与并行逐字摘抄、一次终答、可选独立核对。旧 parallel/lean/needs/funnel 试用流程已删除；并行提问是多个独立 flat 任务，不是恢复旧流程。

模型决定本轮关系（新问题/追问/更正）、检索词、所问对象、信息项与是否计算。信息项必须能绑定用户原话。直接查找数值不自动触发算术。规划和阅读要求紧凑 JSON；阅读省略无引文的空条目，但仍须检查所有信息项。

少量上传文件全部读取；超过 `flat_title_options.read_all_below` 时按文件标题与对象名字面匹配选择。使用已有解析快照的原文和哈希，连续正文按重叠窗口阅读。`max_windows_per_source` / `kb_max_windows` 是显式资源上限，超出会留下 `windows_limited` 诊断，不表示已经读完全部资料。

相邻同源窗口最多两窗合并。只有模型客户端提供无生成的实际 token 预算检查且完整输入/输出预算通过才合并；失败拆回完整窗口，不截掉文字。跨窗引文优先绑定完整引用，再绑定实际展示过的分句，不查未展示的间隙。

阅读引文容量随实际展示材料长度变化，完整JSON须通过Schema校验。语法/类型失败或长度耗尽最多重读一次，两次独立调用及原输出均保留，不补写JSON。名称字面匹配用于检索和审计，不能替代模型对跨语言指代和事实支持的判断。

Writer 只接收选中的逐字引文，保留来源URL及发布时间/抓取时间的区别。可按问题要求生成较长介绍或分项比较，输出预算为2048 token；较长要求仍可能超出预算，不能承诺无限长回答。回答按 `answer`、`points`、`unknown` 原样顺序展示，不补引用、不删句或改写。要点重复默认最多触发一次新的、单独留痕的生成；设置 `retry_repeated_points: false` 时保留第一次原回答及重复诊断。独立核对只做标记，不修改原输出。模型拒答、非法引用和获取失败同样留存，不能由完成状态推断事实正确。

常用 `flat_options`（括号为默认值）：

- 阅读：`window_chars`（2400）、`window_overlap`（200）、`max_windows_per_source`（4）、`kb_window_chars`（6000）、`kb_max_windows`（40）、`read_window_batch`（2）、`snippet_batch`（4）。
- 资源：`deadline_seconds`（180）、`acquisition_timeout`（70）、`final_reserve_seconds`（50）、`final_quote_limit`（40）、`per_need_limit`（12）。
- 行为：`supplement`（true）、`review`（true）、`retry_repeated_points`（true）。
- 可选输出契约：`read_require_assessment`（false）、`final_response_style`（`standard`）。

`read_require_assessment: true`要求Reader至少输出一条资料与信息项的判断记录，两个引文数组仍可都为空；空记录不会变成证据，也不会因此调用Writer。此选项改变生成契约，需单独验证漏摘与误摘，不能由记录非空推断召回成功。`final_response_style: "concise"`让模型在`answer`中完整回答所有要求，`points`生成空数组，`unknown`仍使用字符串保留缺项；答案仍可分段，长度预算不变。它不删除已经生成的句子，也不替代事实核验。默认`standard`保持原有补充要点行为。

`flat_title_options` 支持 `read_all_below`（12）与 `limit`（6）。Web 提取并发由 `web_provider_parallel` 和 `web_search_concurrency` 控制。`native_max_concurrency` 默认 32，可配置 1–1024；这只是调度上限，部署前须测量实际模型容量，并确认后端能隔离并发请求的 State。

## 执行进度与来源范围

顶层 `live_progress: true` 启用 `execution_events`：规划、取材开始/结束、资料进入、模型调用、逐字绑定、阅读结束及终答证据。调用事件使用 `operation` 配对；观测回调收到隔离副本，异常只记录，不改变模型输入或答案。Studio 以实际事件显示状态，未执行的阶段保持未执行；没有 token 流或计时模拟的步骤完成。关闭事件时仍可查看完成回答和来源快照，但不提供中间进度。

每次请求可指定 `web_include_domains`（最多 8 个纯域名，支持其子域名）；省略时沿用顶层 `flat_web_include_domains`，显式空列表解除默认范围。URL、路径和搜索指令会被拒绝。顶层 `flat_web_scope_query: true` 可把域名附加为 `site:` 查询范围，同时继续校验返回 URL。记录同时保留原模型 `query`、实际 `provider_query` 和 `include_domains`。域名范围不提供正文或期望答案。

`web_source_urls` 显式传递要阅读的 HTTP(S) 地址，仅用于网络或混合模式；与问题中的URL合并去重后最多8个。获取结果仍经过同一阅读、逐字绑定和终答流程，用户问题保持原样。来源地址不携带正文或预期答案。

问题正文中的显式HTTP(S) URL（最多8个）通过AnySearch逐URL获取完整正文，初始取材不改成模型搜索词。域名范围仍生效；每个URL的成功、失败或过滤状态进入记录，提取失败不制造空白证据。不支持直接提取的提供方返回明确错误。没有显式URL时沿用模型查询，补查仍由模型查询驱动。

## Web记录与离线重放

`settings.web_search_provider` 可设为 `"anysearch"`，同时设置
`web_anysearch_api_key_file: "/absolute/path/active.key"`（仅当前用户可读，
只包含一把Key），或用 `web_anysearch_api_key` 配置单Key；两者不能同时设置。
默认仍为Tavily。AnySearch先调用 `/v1/search`，再按返回顺序用同一Key
调用 `/v1/extract` 获取正文。只把extract成功的正文标为`fetched`；
无法提取的页面保留原始摘要，标为`snippet_only`并记录获取错误。
搜索和正文请求共享限流与认证/额度冷却。单Key不自动轮换；显式配置私有 `web_anysearch_api_key_pool_file` 时，可在认证/额度/频率错误后按文件顺序尝试下一把Key，trace只记录索引和安全错误码，不记录凭证。无匿名回退。
`web_anysearch_extract_timeout` 默认8秒，限制每页提取（包括等待共享Key）；
整个搜索获取仍受`web_search_timeout`约束。慢页超时或提取预算耗尽时保留
已获取正文，其余页面显示原摘要和明确的获取错误，不因一个慢页丢弃所有结果。
每页提取时限参与AnySearch重放键；更改它不会误用旧时限的记录。
单页上游5xx故障保留其他页面和该页摘要，共享冷却仍生效，不在请求内重试。
AnySearch没有文档化的`include_domains`参数，域名限制只过滤已返回URL的
主机及其子域名。默认不改写查询；显式启用 `flat_web_scope_query` 时，试用入口
会附加用户选择的 `site:` 范围，同时保留结果 URL 过滤。两种方式都不保证完整召回。
录制包含搜索与成功正文获取的原始响应字节和哈希，重放不调用二者的接口。

顶层 `web_replay: {"mode": "off"|"record"|"replay", "store": "/absolute/path/captures"}` 控制Web获取，默认 `"off"`。`record` 按提供方、精确查询、按原顺序的 `include_domains`、结果数和 `full_snapshot` 保存完整提供方响应及哈希；同一请求重用首次记录。`replay` 只读对应记录，缺失、损坏或不可读均记为获取失败，绝不转为联网请求。原始HTTP响应字节和解析后的提供方载荷均保留在记录中，供审计复核。记录目录包含来源正文，应按本地研究数据管理，不提交版本库。

录制也保存首次上游5xx、网络故障和获取超时的安全错误码；重放同样抛出该故障，
不会把它当成零结果或重新联网。失败记录不含可能泄露凭证的上游错误体。
认证、额度、频率限制、凭证配置与存储故障不作为可重放来源记录，调用方应停止该轮。

## 多轮与原始记录

带 `parent_job_id` 时按父链读取最近三轮用户原话、整理后问题、对象名称和上一轮回答首句；这些内容仅用于指代消解，不作为事实证据。追问重新读取上一轮摘抄过的原文，不生成对话总结。新话题不复用上一轮来源。

每次模型调用保存精确 prompt、raw、哈希、输入证据 ID 和耗时；来源保存快照、清洗阅读视图、引文坐标和绑定信息。`generation.citation_bindings` 表示来源绑定，不能代替独立事实核验。记录位于 `records_dir`；服务重启后可经任务 ID 读取已保存记录。目录可能包含完整用户资料，不进入版本库。

## 启动

准备仅当前用户可读的 JSON 配置：

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
  "live_progress": true,
  "flat_web_scope_query": false,
  "flat_options": {"read_window_batch": 2, "retry_repeated_points": true},
  "gpu": null,
  "expires_at": null
}
```

补充实际 `native_model`、`native_api_key`、兼容的 `native_state_routing` 角色映射和 Web 配置；示例不是完整凭证配置。API key 不进入前端或版本库。

```bash
export RWKVRAG_TRIAL_CONFIG=/absolute/path/trial/config.json
PYTHONPATH=src .venv/bin/python -m uvicorn llamaindex_retrieval.interactive_app:app \
  --host 127.0.0.1 --port 18671
# 打开 http://127.0.0.1:18671/studio/
```

原有 React 试用页需另在 `web` 中构建，并设置 `RWKVRAG_TRIAL_STATIC` 为构建目录。Studio 直接请求同源 `/v1/try/`，可与应用、模型部署在同一服务器，网页访问无需模型 SSH 转发。默认使用回环监听；需要通过 Tailscale 访问时可显式绑定该主机的 Tailscale IP，并用 tailnet ACL 限定访问。此入口没有用户认证，不应直接暴露到公网。

## 接口

- `GET /v1/try/meta`：知识库选项、模型、配置并发、`model_readiness`、`live_progress` 与试用截止时间。就绪状态与质量核验分开。
- `POST /v1/try/ask`：`question`、`retrieval_mode`（`hybrid`/`knowledge_base`/`web`）、`knowledge_base_ids`、可选 `web_include_domains`、`web_source_urls`、`supplementary`、`parent_job_id` 和 `pipeline`（仅 `flat`）。返回任务 ID。
- `GET /v1/try/jobs/{id}`：实际阶段、可选执行事件、最终原回答与来源图。
- `POST /v1/try/jobs/{id}/cancel`：取消应用任务和等待中的调用；不保证已提交的远端生成立即停止。

应用为每个请求创建独立任务；实际模型并发由配置和后端控制。CPU 与浏览器测试验证接口和展示行为，不构成模型速度或答案质量成绩。
