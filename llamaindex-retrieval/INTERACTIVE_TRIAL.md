# 上下文与并发漏斗问答

独立 FastAPI 入口 `llamaindex_retrieval.interactive_app:app` 提供 `/admin/#/try`，连接已有知识库、RWKV 原生推理服务及 Web 提供方。

处理流程：**上下文整理 → 获取原文 → 分块并发总结 → 分层并发归并 → 按缺口补查 → 最终回答**。不同领域的问题使用同一流程，不按题型写规则；旧直拼与两版对照入口已删除。

## 上下文与资料

当前问题先交给 RWKV 整理。显式传入 `parent_job_id` 时，模型结合上一轮整理后的完整问题及上一轮回答理解指代和更正；上一轮回答仅帮助理解，不作为事实依据。没有父任务时从空上下文开始。历史原始记录独立保存，不把整段历史无限追加给模型；整理失败如实返回，不用程序猜出替代问题。页面“新的提问”清空当前父任务。

知识库分页读取全部目录，每批128个标题交给模型选择，再下载所选文件的 UTF-8 原文。128仅为提示分批大小，不限制选中文档数量。Web获取保留提供方返回的正文或摘要快照、URL、时间及抓取状态。不调用本地词项/向量相关度评分，不把标题当事实证据。

原文使用已有逐字结构分块器：连续正文4000字符软窗口、240字符重叠，保留坐标与哈希；表格、列表、代码和QA保持结构原子及父级上下文。所有块排队并发总结，不按前N块截断。超长原子无法满足单次输入预算时明确记录失败，不静默截断。

每个归并节点最多接收3个直接子输出；同层分组并发，再进入下一层。总文档数、总子任务数和总层数不设固定上限。归并仅接收本组模型raw及节点标识。

可选候选：终答（`interactive-funnel-v2`，默认关闭，传`original_chars`启用；2026-10-19 TEST24配对6/24→4/24未过门槛）另外接收叶层被模型选中的逐字原文块：沿归并链收集模型实际采用的叶，按选择顺序装入，输入证据预算单独配置（默认16000字符）；超出预算的叶不截断，记入`omitted_original_leaf_ids`。未被叶选中的原文块、失败叶和全量来源图不进入终答。提示要求事实以原文为准，总结只帮助定位。

上下文、叶总结、归并和终答分别保留输出预算1024/768/1024/2048 token；每次真实模型调用前通过服务tokenizer校验完整输入加输出预留。并发数只限制同时执行的数量。补查循环最多300秒，请求含队列最多900秒，均为资源时限，超时明确记录。

## 片段候选链路（`pipeline: "spans"`，默认关闭）

配置顶层设置`"pipeline": "spans"`后改用`interactive_spans`：**制定检索 → 按检索语句获取 → 并发片段选择 → 三叉片段归并 → 按缺失信息项补查 → 终答**。

- 制定检索：模型输出回答必须找到的信息项`needs`和检索语句`queries`（数量不设上限）；完全相同的条目去重，原始计划保留。计划失败或为空时用整理后的问题检索并记录。
- 阅读：原文按3000/240窗口、窗口内640/80切成带ID的逐字片段。模型只输出所选片段ID及覆盖的信息项；未选片段不进入上层。
- 归并：只合并选中了片段的节点，每组最多3个；模型保留片段子集并报告`missing`信息项。
- 补查：根节点仍有缺失信息项时，模型为其写新检索语句（不重复已用语句），只阅读新获取的原文；最多2轮，300秒资源时限。
- 终答：只接收入选逐字片段（默认6000字符，超出记入`omitted_span_ids`），引用片段ID；`cited_spans`给出引用片段原文。

上下文预算：阅读窗口3000字符，单次归并的片段文本不超过4000字符（超出按片段分批，每批独立调用），终答片段不超过6000字符，超出部分记入`omitted_span_ids`。字符上限只控制分批，不能保证完整提示的token数量；需要在服务配置中设置`settings.native_context_window_tokens: 6000`，由服务tokenizer约束完整输入加输出预留，超预算明确记录失败。

2026-10-19 K01–K08诊断7/8（联网渠道因Tavily 432不可用，未与漏斗配对，不作晋级依据）。

## 逐信息项链路（`pipeline: "needs"`，默认关闭）

`interactive_needs`：制定检索（信息项＋按渠道查询）→ 获取 → 网页0–3打分门 → 每个阅读窗口一次**逐信息项片段打分**（只输出2/3分）→ 代码按模型分数为每个信息项取前3个片段；3分片段超过3个时发起一次比较调用（≤8片段、≤4000字符）→ **逐项回答**（只看该项≤3片段；`found`必须引用片段，`not_found`不得引用）→ 未找到的信息项按渠道补查，只读新材料、只重答未找到项（≤2轮）→ **汇总**（只看各项短回答及片段ID）。每次调用输入保持在约6k token以内；各阶段输入token与失败数记入`telemetry`。

逐信息项汇总只接收结构通过的 `found` 回答；未解决项仅传名称，不传缺证回答正文。全部缺证时不调用汇总，状态 `no_evidence`。这不代表混合回答的语义已验证，原始错误仍须记录和评分。`needs` 客户端通过服务 `/tokenize` 约束完整输入加输出预留不超过6000 token，超预算明确失败、不截断。

## 知识库导航卡片（`kb_cards`）

配置 `kb_cards: {知识库ID: "/absolute/path/CARDS.jsonl"}` 后，完整覆盖目录的卡片通过短ID锦标赛选择文件：每25张一批、每批至多2份，继续对胜出者选择，最终至多4个文件。类别门控默认关闭；卡片只作导航线索，生成失败的卡片保留文件可达性及 raw。

完整结构、预算和失败合同见[模型导航说明](MODEL_NAVIGATION.md)。

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
- `POST /v1/try/ask`：`question`、`retrieval_mode`（`hybrid`/`knowledge_base`/`web`）、`knowledge_base_ids`、`supplementary`（默认true）、可选`parent_job_id`。返回任务ID。旧`mode`/`pipeline`参数拒绝，不恢复直拼或对照分支。
- `GET /v1/try/jobs/{id}`：真实阶段、上下文整理结果、最终回答及来源图。
- `POST /v1/try/jobs/{id}/cancel`：取消应用任务及其等待中的调用，不能保证已提交的远端生成立即停止。

请求在`records_dir`保留原始问题、模型prompt/raw/hash/耗时、原文快照、分块坐标、归并图与失败状态。服务重启后可重新打开已完成记录。目录按本地数据管理要求保存，不提交到应用仓库。新部署使用独立记录目录与浏览器存储，不自动载入旧直拼试问。
