# 属性证据第一阶段：数据契约与交付边界

维护契约；部署状态见 [CURRENT](CURRENT.md)。核对日期：2026-09-20。实现协议：atomic-evidence-v7。前置备份：68b6bf4a6587ad1d446d70634d98299bdb54ab0e。

## 本阶段解决什么

用户指定一个对象、一个属性和可选条件，系统检索并定位短原文，分别保存不同来源的记录，点击即可核对当时的原文。本阶段没有调用 Writer。

**这里的“原子”是面向一个属性的证据记录，仍不是标准化事实。** 实测中，7.2B 在复制数值和定位子句时可能丢空格、否定或只选中“正式更正：”。因此当前记录完整短证据，不声称已经可靠抽出数值、单位、时间、权威等级或撤回关系。模型输出本身仍原样保存。

## 流程与职责

1. 指定知识库，固定当前物理索引版本，执行 BM25 检索，最多检查 8 个来源。
2. 按句子/结构切分；QA 对作为同一个单元，表格行携带表头，长正文有重叠。
3. 每个来源通过词项的局部逆文档频率选最多 6 个短候选。记录未检查与超长数量。
4. 7.2B zero State 只输出候选编号数组。非法 JSON、重复编号、越界、输出截断均记为处理问题。
5. 现有 2.9B + Reader450 对每个选中的短片段判断能否回答该属性。
6. 程序复制该片段的准确原文与位置，保存来源快照、索引版本和模型调用。Reader 的 NO 保存为“相关性待确认”，**不推断为未记载、零值或事实不存在**。
7. 前端展示原文、Reader 标签、上下文、范围限制与处理记录；点击来源按 Unicode 字符位置高亮。

默认上限：32 次模型调用、180 秒、模型输入 4096 token、单次输出不超过 96 token。上限用于约束工作量，不能保证语义正确。预算不足或超时为 incomplete，已有结果保留。客户端超时/取消不代表远端推理已停止。

## 契约

### 请求

object 与 attribute 非空、各不超过 120 字符；conditions 不超过 300 字符。严格拒绝额外字段。知识库由路由参数限定，不能通过请求自定义模型端点。

### Run

| 字段 | 含义 |
| --- | --- |
| id / knowledge_base_id | 执行标识及知识库范围 |
| request / protocol | 原始请求及协议版本 |
| status | running / completed / incomplete / failed，仅表示执行状态 |
| semantic_verified | 固定 false，不能把完成状态理解为质量验收通过 |
| index_version | 开始检索时解析得到的物理索引 |
| snapshot_scope | 保存的是读取到的索引片段，不是数据库事务级时间点快照 |
| sources | 当时读取的完整索引片段及来源元数据 |
| claims | 面向该属性的原文记录 |
| calls | 精确 prompt、prompt 哈希、原始输出、调用标识、模型与阶段耗时 |
| coverage / issues | 检查数量、未检查数量、预算/格式/模型异常等 |

### Claim

| 字段 | 含义 |
| --- | --- |
| target | 用户指定的对象/属性/条件；不是模型已经证明的实体关系 |
| kind | reader_supported 或 unconfirmed，仅为 Reader 的判断 |
| statement_quote / statement_position | 完整选中短原文及其在保存片段中的位置 |
| evidence.context | 随附表头、标题等；外部元数据上下文有单独位置原点；仅在同源范围和哈希可验证时移除重复本地引用 |
| evidence.context_deduplication | 去掉的本地重复范围、覆盖它的元数据 context_index 与方法版本；不修改原文快照 |
| value_quote | 当前固定 null，尚未标准化出独立值 |
| normalization_status | not_performed |
| binding | 知识库、来源、文档、索引版本、文本哈希、可用的源文件版本哈希 |
| selection_call_id / support_call_id | 选择和 Reader 判断的模型调用标识 |
| source_claim_not_resolved_fact | true；禁止直接作为已裁定事实使用 |

位置采用 Python Unicode code points。前端用 Array.from 处理包含 emoji 的文本，逐字校验后才高亮。位置或原文不匹配时显示错误，不猜位置。

源文件/解析快照哈希缺失时保留为空，不能凭空生成“文件版本”。索引文本始终计算 SHA-256。索引版本相同也不证明文档没有原位更新；历史页面展示的是保存原文，不能冒充最新原文。

### 持久化与 API

MongoDB 集合 atomic_evidence_runs。先写 running，结束时仅允许对 running 的同一记录进行一次替换，完成后不允许覆盖。大记录沿用 GridFS 存储。历史最多返回最近 50 条。

- GET /v1/admin/atomic-evidence/capabilities
- POST /v1/admin/knowledge-bases/{kb}/atomic-evidence
- GET /v1/admin/knowledge-bases/{kb}/atomic-evidence
- GET /v1/admin/knowledge-bases/{kb}/atomic-evidence/{id}

记录读取、检索与来源绑定都限制知识库范围。**这是资源范围校验，不是新增的用户鉴权/RBAC。** 当前锁只限制一个应用实例的一次运行；没有实现多实例任务租约、重启恢复和跨实例资源配额。

## 六类基础验收

| 场景 | 本阶段要求 |
| --- | --- |
| 同属性冲突 | 18 W、25 W 分别保留来源，不能覆盖成一个值 |
| 不同适用条件 | 标准/增强模式原文同时保留，不丢条件 |
| 修订与撤回 | 保存旧草案和正式更正；不删除“旧值撤回”文字；本阶段不自动裁定 |
| 零值与预约量 | 0 台和预约 80 台保持准确原文及独立 Reader 标签，不能把预约改写成交付 |
| 明确未记载 | 保留含“未记载”的原文，不能生成 0，也不能把运行失败解释为未记载 |
| 长材料定位 | 在 140 条冗余记录后定位最终证据，同时说明其余未检查范围 |

每类有原版与来源顺序/冗余附加变体，共 12 个固定材料测试。另有 8 个后续新增用例检查表格、QA、单位、时间地域、未公开、零值、恶意指令和长记录；这些测试也不是商业领域泛化保证。

## 部署与下一阶段

功能默认关闭，需要配置 atomic_model_base_url 并具备已配置的 native batch Reader。预览使用独立 MongoDB、OpenSearch 索引和上传目录，不接管正式问答或 Wiki。

预览地址、服务寿命与本次核对状态统一见 [CURRENT](CURRENT.md)，不能从实现契约推断运行服务已经启用。

下一道门槛是严格验证“对象、属性、时间、地域、否定”的相符性，处理表头误选与 QA 否定漏判，再冻结新的未见测试集。通过后才能让标准化原子记录进入分子关系、冲突裁定、阶梯总结与 Wiki 更新。StateTune 训练、分子层、Writer 改造均未在本阶段实施。
