# RWKV统一检索候选 `rwkv-unified-v2 / v3 / v4`

**默认关闭，未通过语义/生产验收。** 知识库与Web是同一检索核心的来源适配器，不是两套回答器。不能从代码或离线测试存在推断已经部署或模型能正确执行。

## 启用边界

`native_unified` 默认 `null`（关闭）。在独立测试配置中显式选择：

- `native_unified: {}` 或 `{"reading_protocol":"answer_v1"}`：保留v2旧合同用于兼容/回归。
- `native_unified: {"reading_protocol":"span_ids_v1"}`：v3片段ID阅读、继承来源和终答合同。环境变量示例：`RWKVRAG_NATIVE_UNIFIED='{"reading_protocol":"span_ids_v1"}'`。

- `native_unified: {"reading_protocol":"span_ids_v1","control_protocol":"review_cycle_v1"}`：v4有界控制闭环；仅支持span阅读，默认最多1次Plan修订（`max_plan_revisions`可设0–2）。不设置control_protocol仍保留v3行为。可再显式设置`plan_contract:"bounded_v1"`启用逐调用Plan Schema；默认`static_v1`不变，该选项仅用于v4，不新增独立流水线。

三者复用同一Task/Direction/Plan/检索/Review/预算核心，不是独立回答器或失败回退。要求native/g1j_plain、模型身份检查、闭合no-think输入及immutable输出。模型URL/ID/上下文长度必须显式绑定。

可另配置默认关闭的 `retrieval_gate`，使用专用二元判断模型（与旧 `web_router_*` 联网判断配置分开）。该配置需要 `base_url`、`model`、`checkpoint_sha256`、`state_sha256`，可选 `api_key`、`state_ref` 和 `timeout_seconds`；仅支持 span 阅读入口。训练和服务必须共享 `retrieval_gate.decision_prompt` 的逐字协议，模型只返回 `true` 或 `false`。接口不会从权重文件存在推断模型已经部署。

启用后，普通请求先判断是否检索。`false` 不调用 Task、Direction、Plan 或资料渠道，终答模型直接处理原始对话；`true` 继续理解任务并生成 `queries/query/goal_ids`，每条模型查询原样交给所有显式允许的渠道。`auto` 和 `hybrid` 都执行KB及Web；调用方明确选择仅KB或仅Web仍是渠道权限限制。预算不能容纳完整渠道组时记录预算耗尽，不悄悄退化成只查一个渠道。判断失败不当成 `false`；固定材料入口不调用判断模型。判断调用计入请求总预算，精确prompt、raw、哈希和耗时进入trace。展开后的执行计划与模型原始查询计划分别记录，原文回答和模型raw均不改写。该接口和工程回归不代表分类准确率或最终答案已达标。

禁止与State、schema_v1、task matrix、typed funnel、anchored_v1、自动格式修订、自动原问插入或whole-source Writer packing混用。旧Reader/Writer语义协议不用于新核心；不要沿用一串旧开关推断新阶段行为。停用候选不是请求失败时的自动回退。

v2只用提示内JSON合同和严格解析。v3通过`StructuredNativeRWKVClient`为Task/Direction/Plan/Review/SpanReading/SpanAnswer显式发送`structured_outputs.json`，澄清仍为原始文本。需要后端支持这些实际Schema；不支持时失败，不回退无约束模式。旧`schema_v1`针对的三个合同不覆盖本流程，不能据其兼容性推断v3可运行。输出非法则记录失败，不剥围栏、补字段或换模型；v4保留明确的Plan修订调用，不自动重读失败材料或做native/传输重试。实验性后端兼容实现不是默认服务已经配置的证明。

## v4控制闭环

- Review选择search后另调Direction，在调用方`allowed_channels`内重新选择本轮渠道；Plan及修订仍只能用最新Direction的选择，不能自行扩权。Direction错误也不由代码强选渠道。
- 每轮阅读额度为剩余单元数除以剩余轮数向上取整；若两轮最小工作量可容纳，还预留下一轮Direction/Plan/至少1叶/Review与全部归并调用，避免首轮吃完控制预算。预算不足则如实停止或只做可承受的一轮，不伪造补查。跨动作再跨来源轮转，避免首个动作的大候选列表饿死第二渠道。固定材料闭集不扣留未来检索额度。
- Plan结构/绑定/预算失败最多触发`max_plan_revisions`次独立`plan_revision`调用；输入完整旧raw和错误，输出完整新Plan。被拒记录位于当前任务/预算/已执行动作/Review之前（`revision_input_layout=rejected_first_v1`），只改输入排列，不改任何值或旧raw。旧失败及新调用各自保留call/hash关联，不去重、截断或替换旧raw。已安排目标与unplanned不得重叠，动作数不能超过本轮阅读上限。基础设施失败不触发修订。
- 可选`plan_contract=bounded_v1`为Plan及修订绑定本轮/剩余动作上限、Direction选定且仍有额度的渠道和Task目标ID；仅Web时总动作上限再受Web额度约束。提示所印Schema与解码Schema完全相同，调用记录`plan_contract/plan_schema_sha256`。不改旧raw，不截断动作或代生成查询。混合渠道的非零Web次数、重复查询和已安排/未安排目标重叠仍靠严格事后校验，不能把局部Schema约束当全合同或事实通过。Schema是调用局部值，不污染别的请求。
- 只有该修订调用可看到被拒绝的控制raw，并明确标为非事实证据；Review、Direction和归并仍不接失败阅读raw或查找表原文。普通provided方向可经Review再次决策，显式material-ask仍不外查。
- `retrieval.directions/plan_attempts`记录事件/准入；方向事件另记`completion_attempted`，预算阻断标`not_started`，不冒充模型已选择。`execution_scope.round_budgets/plan_revision_calls/plan_revision_events`分别记录分轮预算、实际发出生成请求的修订数和修订事件数。前端按完成请求收据计数，不把事件、请求和成功混为一谈。方向失败不扩权，可使用预留终答报告执行缺口。

这些是可执行控制与预算机制，不保证模型选对渠道、计划修好、事实受支持或根回答完成任务。兼容后端仍需显式配置并验证上述六个合同及启用时的逐调用Plan Schema；普通未适配的RWKV tokenizer不能假定兼容。

## 数据流

1. `Task`：普通检索请求分别提供未改写的`history`和`latest_user_message`，只解释任务，不决定渠道。v3/v4显式材料入口直接阅读原始对话与材料，不生成Task，记录`task_execution=not_required_closed_materials`；不伪造Task或修补消息引用。v2兼容入口不变。goals/constraints/retracted统一为`{id,text,refs}`；unresolved为字符串数组。不再输出重复question或mode/channels。代码检查ID、逐字消息绑定、至少一个目标引用最新消息；这些结构锚点不证明历史对象/条件/撤回理解正确。
   `Direction`：普通请求另调RWKV选择provided/knowledge_base/web/hybrid/clarify，程序仅校验调用方允许的渠道；非法选择直接失败。固定材料入口不调用方向模型，显式记录`source=explicit_materials_entry, model_decision=false`，不伪造路由raw或越权外查。
2. `Plan`：模型按剩余查询/阅读/调用预算返回channel/query/goal_ids，仅使用Direction选定的渠道。全部目标在actions或unplanned_goal_ids中记账；超项、重复、越权或遗漏ID直接失败，不取前N条，不悄插原问。目标ID齐全不代表检索覆盖正确。
3. 来源适配：KB保留请求scope/candidate_k，读当前索引返回块；Web复用原provider/单Key共享guard，取得完整可得snapshot。不自动直取网页/下钻或换provider。snapshot不等于网页完整、官方或正确；当前provider收据仍是规范化结果，不是完整外部HTTP wire。
4. 原文目录：整份可得snapshot机械切分、通用BM25排序、各来源队列轮转。span模式在排序前按软窗口逐字打包同一snapshot的相邻结构块，避免短说明/代码各自成为一次阅读；不跨来源，不拼接不连续段落，不生成摘要。打包单元保留`packed_components`坐标/hash、原子属性和父上下文，单元正文仍是原文连续切片。正文overlap不丢，代码/表格/列表/QA原子不拆；原子超长仍交给实际token预算拒绝。v2及普通`ranked_windows`默认不打包。保留全部单元、各查询观察及未读范围。KB坐标相对保存块，不冒称已读取整篇文档。
5. 叶层：每单元先独立调用RWKV。v2返回`{answer,evidence_ids,missing}`，要求E编号及内联引用绑定。v3对本次原文建立`SpanCatalog`，返回`{findings:[{span_id,statement}],missing}`，只绑定这次真正送入模型的目录。合法ID不证明statement受到支持；不会依据程序的语义推测删除错误findings。格式、绑定、length和基础设施失败分别记录，raw不改。v3/v4的生成Schema只将ID范围绑定到本次目录（归并则为合格直接子节点的选择），自由文本不改写，严格JSON与span范围校验保持；记录`span_schema_sha256`，归并提示内Schema与实际解码Schema相同。失败阅读不自动重试或局部补救；全部原始输出留审计。ID范围约束不把合法ID变成事实认证。
6. `Review`：看任务、叶层派生输出、执行元数据和剩余预算，决定stop或补查。v2/v3下一轮再次调用Planner，v4先重新Direction再Plan；不把完整未选快照偷偷放进Review/Planner，也不把模型判断当已验证事实。
7. 归并：每组最多3个直接子节点，含不足3份的尾组，递归到1个root。归并只看派生输出和继承绑定，**没有收到这些原文正文**；trace原文输入ID为空，引用可追溯性不等于高层重新核读。原文在独立目录中保留供审计。失败节点的raw仍完整保存于审计图，但Review/归并输入只给状态、ID/call/hash及`raw_withheld=true`，不把失败正文作为可用事实传递；有效节点raw不改。v2的n=1仍返回叶raw。v3叶是局部findings，因此n=1也另调终答模型；中间层和最终层都用`SpanAnswer={answer,span_ids,missing}`，最终层提示明确回答原任务。输出span_ids只能来自准入直接子节点实际继承的选择集合，不能越组、恢复已丢弃选择或查全局最近目录。v3/v4的终答不因无合格子节点而跳过：仍调用RWKV，只有执行缺口和原始对话可用时，来源选择范围为空。中间层的全部失败组仍标blocked_dependency，失败正文不进入终答。没有程序编写的缺口答案；模型仍可能拒答、答错或失败，必须另报事实质量，不能以非空代替答案验收。

`/v1/search`和`/v1/ask`共用步骤1–6；search只返回模型选中的证据和trace，最后才按top_k限制展示，不进行答案归并。`ask_materials`复用单元阅读与归并，关闭外部渠道，适合固定材料验证；span模式不需要任务提取作为阅读前置条件。provided且未显式提供materials时仅使用用户角色原文，不把助手历史当事实来源；clarify单独调用模型产生原始澄清问题。

## 输出、失败与预算

本候选的`AskResponse.answer == generation.raw_model_answer`是**根节点的完整原始JSON字符串**，不是程序提取/拼接的answer字段。其内部answer为模型写的正文。v3/v4前端核对唯一root、call身份、与两份raw相等、节点completed及结构后，才纯投影answer/missing；不改API、不补写事实或内联引用，保留完整raw查看。选中的span IDs单独列示，原文抽屉明确标为ID查找收据（不是模型生成quote）。span正文中的`[资料 N]`、`[Source N]`、`[N]`和代码按字面显示，不自动生成第二套引用按钮，不因旧正文引用扫描或复读启发式隐藏正文；重复检测仅提示，不认证内容对错。未读/失败/模型缺口/语义未核验分别提示；投影不是质量验收。澄清返回原始文本。UI维持完成回合才自动纳入历史的规则，v3/v4纳入完整root raw JSON而非丢掉missing/选择；诊断的动态历史规则独立，非空错误父raw仍原样传递。

`generation.root_node_id/answer_call_id`指定唯一最终节点；`nodes`保留raw/hash、父节点/call/hash、输入原文ID和允许/选中证据ID。`citation_map`全请求稳定，sources过滤后不重排引用号。校验只标失败，不删改原答。节点`parsed_model_evidence_ids`记录结构解析得到的模型选择（不可解析时为null），`evidence_ids`仅记录校验准入；引用失败不表示模型没有选择ID。v2仍要求结构化ID和正文引用；v3节点另记录`parsed_model_span_ids`与实际准入`span_ids`，前者包括最终绑定失败的选择，不从截断JSON猜选择。v3/v4显式span_ids是唯一来源选择声明；正文不按v2内联标签语法另行准入。`citation_audit.reference_protocol=span_ids_v1`，`inline_citation_interpreted=false`，不以字符串字面量误拒有效选择。未知/越组/重复span ID仍严格拒绝，真正无据的陈述不会因此变成有据。两者均不证明逐句蕴含。`model_calls`保留叶prompt中本次实际目录及catalog hash；根`span_bindings`（search为`selected_span_catalog`）保留所选坐标/hash、lookup正文和叶node/call/raw hash；Review/归并只接无正文的继承元数据，不接lookup正文。`execution_scope`公开未读单元、控制/来源/节点失败；`root_contract`标记根合同。

- 默认上限：2轮、6总检索动作、24阅读单元、96模型调用、600秒；Web总动作仍受`web_search_max_queries`（默认2）约束。
- `stage_concurrency` 默认1，可在 `span_ids_v1` 中显式设为1–32：同一请求的独立阅读节点及同层归并节点按有界批次并发，父节点等待本组子节点全部终态后才运行。例：`RWKVRAG_NATIVE_UNIFIED='{"reading_protocol":"span_ids_v1","stage_concurrency":8}'`。此值控制单请求；实际并发还受共享 `native_max_concurrency` 和模型服务批处理容量限制，增大客户端值不会自动提高服务端容量。没有相关模型实测时不能据配置宣称提速或答案改善。
- 并发阶段仍先预留归并/终答和Review预算，结果按原始输入顺序进入每组三个子节点的树，不按完成先后重排。传输失败后不再启动下一批，已发出的同批请求等待终态；超时或取消会取消并等待所有本地在途任务，远端是否已经停止仍按传输收据记录，不能推定。失败raw继续隔离，不补来源、不重试、不改答。请求trace的 `request_budget.stage_concurrency` 记录实际策略，默认为1的兼容执行保持串行。
- 单元软窗口2400字符；Task/Direction/Plan/Review输出2048、叶768、中间归并1024 token；最终归并使用`generation_max_tokens`（默认2048）。只有v2的n=1沿用叶预算；v3的n=1也使用单独的最终输出预算。
- 较小的`native_request_max_calls/timeout_seconds`进一步限制候选；入口级时限覆盖检索、排队和归并。
- span Ask从初始Task/Direction起，就在每个控制调用前保留当前归并和终答额度。预算不足的控制事件标`final_budget_reserved`、`completion_attempted=false`，不占用剩余终答槽。初始Task/Direction或后续Plan/Review结构/绑定失败时，停止获取资料，仍可发起一次独立终答；只传原始对话、合格子raw/绑定和机械缺口，不传被拒控制正文或伪造路由。没有来源时选择范围为空，返回no_evidence，而不是把无据生成当成功。search/v2不增加终答回退。
- 真实澄清终答不再追加另一个final；传输失败/请求超时仍停止，不承诺所有请求都有答案。实际生成请求以`completion_attempted`收据为准，不用事件数或根准入状态代替。
- 失败阅读仍保留独立终答额度；不自动重读、不恢复截断JSON中的选择。`tests/test_unified_material_flow.py`覆盖原始对话保留、闭集免Task、失败raw隔离、实际终答调用与逐调用ID范围。
- 阅读前预留剩余归并/Review调用额度；实际Native客户端对完整prompt做真实token-ID检查，不截子raw或强塞原文。固定调用数不保证能在上下文/期限内完成。闭集读完可读额度后保留真实`read_budget_exhausted`及未读单元，不覆盖成全部材料处理完成。
- 超时只返回已指定root的raw（若确实存在）；不能使用最近一次中间输出。调用者取消继续传播，不能承诺远端线程/模型已停止。
- 基础设施模型失败停止后续调用；格式/length失败保留为失败分支，有效分支可归并为partial。未读/预算/失败台账不删除。
- `completed`只说明执行/结构与引用身份合同完成；`partial`保留未读范围/缺口/失败等执行限制，**不等于语义评分**。零已选来源标no_evidence，原始缺口说明仍可保留。

## 验证

`tests/test_unified_concurrency.py` 覆盖真实异步重叠、乱序完成下的来源/父子绑定、预算及终答预留、同批故障收尾、阅读/归并超时取消和跨请求隔离。这些调度测试不替代实际RWKV的吞吐或答案质量验证。

`tests/test_unified_engineering_boundaries.py`检查正文原样、归并不误拒字面量、严格ID拒绝、初始控制/最小调用额度的真实终答槽、权限/失败raw隔离及短材料输入覆盖。`tests/test_verbatim_packing.py`检查逐字打包、坐标/hash、上下文、overlap、QA/代码/表格/列表原子和跨来源隔离。`tests/test_inner_span_boundaries.py`覆盖内层打包、空白附着、无可选内容、超软窗口及外层默认行为不变；目录绑定仍不证明陈述支持。`web/tests/spanBody.test.mjs`检查纯字段显示、无隐式链接、不靠正文启发式隐藏答案及实际调用计数。均为工程夹具，不是模型答案质量或训练准入证据。

`tests/test_unified_plan_contract.py`覆盖逐调用边界、零/混合渠道额度、任意目标ID、请求隔离、提示/解码Schema相等、后端忽略约束时仍严格拒绝，以及旧配置/非Plan提示保持不变。测试不保证查询有用或修订有效。

`tests/test_unified_revision_layout.py`检查修订记录顺序、连续修订和截断旧raw的完整保留、静态/逐调用Schema兼容与重叠目标继续拒绝；不保证模型修订成功。

`tests/test_unified_control_cycle.py`覆盖v4重新选渠道与调用方权限、独立计划修订/次数/错误与取消、分轮阅读及后续调用预留、动作间公平轮转、provided复议和材料闭集、失败阅读隔离；不作为模型语义通过依据。

`tests/test_unified_task_contract.py`检查任务/权限分离、原始消息绑定及严格解析；`tests/test_unified_retrieval.py`覆盖共享检索闭环、n=0/1/2/3/4/9/10/27拓扑、尾组、不可变raw、引用隔离、来源坐标、全文排序、真实Native客户端token预算拒绝（假HTTP）、失败和超时根绑定等。是假模型/假传输合同，不是实际RWKV兼容性、语义或端到端质量验证。

另有尚未接入本流程的`unified_reading.py`局部阅读候选：`{findings:[{quote,statement}],missing:[text]}`。其校验仅将模型逐字引文绑定到本次原文body和所有匹配坐标，不查标题/其他单元、不规范化或补写引文；绑定不证明statement被支持。`native_unified`不会自动使用该模块，不能将它的存在解释为阅读问题已修复。对应离线测试见`tests/test_unified_reading.py`。

v3已接入的`unified_span_reading.py`将原文body机械划成带ID的逐字片段，RWKV返回`{findings:[{span_id,statement}],missing:[text]}`。`SpanCatalog.build(material)`以默认640字符/80重叠窗口逐字打包相邻结构块，保留表格、列表、fenced code及声明的QA原子；不按问题/事实筛选片段、不改原文。纯空白块附着相邻原文，不单独成为可选ID；可能超过软窗口，实际token硬预算仍照常检查。全部为空白的材料保留原始text但目录无可选ID；非空白材料所有原字符均在目录覆盖内。旧目录/ID不得用新划分重新解释，须用各调用已冻结的原目录绑定。`span_prompt(payload,catalog)`保留原始对话与本次逐字目录，阶段指引只说明职责和输出字段。重复的Task解释、路由和预算留在控制调用/审计，不再作为span阅读或归并的语义输入。归并仍收到原始对话、未改写的合格直接子raw、完整已选来源绑定及机械执行缺口；不补入lookup正文，不删除或修写子输出中的错误。`SpanReading.bind(catalog)`只查找模型已经选中的ID，不补选/改答；返回的`selected_original_text`是明确标记的原文查找收据，**不是模型生成的quote或答案投影**。ID在当前证据/调用中有效，调用方必须用本次真正送入模型的catalog绑定，不能跨请求使用全局“最近目录”。ID合法不证明相关性、事实支持或缺口判断。不会自动替换v2合同。短输入与错误/非法/截断root raw不改写的测试见`tests/test_unified_short_prompts.py`；组件测试见`tests/test_unified_span_reading.py`；`tests/test_unified_span_pipeline.py`覆盖真实管线的假模型调用、跨组/未知ID、失败raw隔离、丢弃引用不复活、QA原子、n=0/1/2/4/10、预算/超时、权限/KB scope、模型补查及并发请求隔离。前端`web/tests/unifiedAnswer.test.mjs`检查唯一根、字节匹配、失败不投影、缺口提示和历史；这些测试不等同真实模型或浏览器交互验收。

先固定材料分层诊断，再分别验证任务/渠道/查询规划和真实入口。原始结果/失败不改，模型/索引/提示/预算分别冻结；一次架构改动的总体结果不能称单变量收益。候选不恢复旧规则模块，不自动发布、部署或准入训练。
