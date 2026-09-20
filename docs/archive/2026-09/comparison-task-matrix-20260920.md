# 对比问答：逐项检索、证据绑定与回答核验

> **历史记录，非当前状态。** 本文保留当时的实验、部署或设计结论；“当前”“最新”“下一步”均指原记录时点。当前事实与行动以[当前状态](../../CURRENT.md)为准。原路径：docs/comparison-task-matrix-20260920.md。

## 范围与运行状态

新增可配置的 task-matrix 链路。默认关闭；实验使用独立模型端口 18424 和独立
OpenSearch 索引 `rwkvrag-task-matrix-eval-20260920-<数据哈希>`，不改业务知识索引。
实现完成与模型质量达标分别验收，不能用单元测试通过代替答案质量通过。

核心实现：`llamaindex-retrieval/src/llamaindex_retrieval/task_matrix.py`。

## 行为

1. Planner 从完整对话提取对象、属性、条件和单项查询。共同属性的比较采用 `coverage=grid`，
   结构校验要求覆盖其声明的全部对象 × 属性组合；分别询问不同对象的不同属性采用
   `coverage=listed`，不扩充用户未问的组合。**校验不能发现 Planner 从一开始就漏掉的
   对象、条件或所求组合**，仍需语义评测。
2. 每项分别检索，保留知识库过滤条件。Reader 逐项判断原文能否提供证据。
3. Assessor 判断资料支持、资料不足或冲突。其选择只作为提示；Writer 保留该项目全部
   Reader 入选原文，避免 Assessor 漏判冲突时隐藏相反记录。
4. 仅对未解决项生成补查查询。限制轮数、查询数量、Reader 调用量和总时长；不无界循环。
5. Writer 接收完整任务、入选原文和项目到实际引用编号的映射。内部 cell ID 不作为引用。
6. 专用 Reviewer State 逐项检查答案与引用。每项只接收 Reader 为该项选中的全部证据，
   保留 Writer 的全局引用编号，不根据 Assessor 的偏好再缩减原文。
   任一项未通过时，默认保留首个原始答案并标明未通过模型检查。
   可配置最多一次有独立 trace 的 Writer 重答，但默认关闭：实际验证发现仅凭二元 NO
   指令重写会把正确零值改成错误的冲突判断。代码不改写答案或补引用。
7. 前端展示对象、属性、证据状态、回答核验和原文入口；Search、History、Wiki 共用组件。

所有模型判断都不等于独立事实核验。`semantic_support_verified` 始终为 false。
保留每次调用的精确输入、原始输出、内容哈希、State 和阶段状态。

## 一致性与资源边界

- 知识库请求固定到当前物理索引版本，不修改共享 index 对象。
- 这不是 OpenSearch PIT 快照；同一物理索引的就地更新仍可能被看到。
- 不同项目复用来源身份时检查正文、文档、URI 和来源是否一致。
- 网页查询额度按尚未尝试的项目优先分配，避免前几个对象长期占满额度。
- 请求超时记录已开始的调用；本地取消不声称远端 GPU 已停止执行。

## 配置

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `native_task_matrix_enabled` | false | 启用新链路 |
| `native_matrix_max_cells` | 8 | 最大单项任务数 |
| `native_matrix_max_rounds` | 2 | 含首次检索的总轮数 |
| `native_matrix_sources_per_cell` | 3 | 每项每轮候选来源数 |
| `native_matrix_max_reader_calls` | 48 | 单次请求 Reader 总调用上限 |
| `native_matrix_answer_repairs` | 0 | Writer 重答次数；二元核验误拒绝可能诱发重写错误，默认关闭 |
| `native_matrix_timeout_seconds` | 600 | 整个请求超时 |
| `rwkvos_planner_state_id` | null | 独立 Planner State |
| `rwkvos_matrix_state_ids` | {} | plan/reader/assessment/followup/review/writer 各自的 State |

batch 模式要求显式配置六个角色的 State，并使用逐项 binary Reader。
标准 Reader/Writer State 不被新链路覆盖；普通材料问答等既有调用保留原配置。
Planner、Assessor、补查使用 Planner 的 no-think 边界；Reviewer 使用 Reader 的带换行边界。
训练与推理逐字绑定，不共用一个 State 强行承担全部任务。

## 实验与验收记录

实验代码和冻结数据：`llamaindex-retrieval/eval/task-matrix-20260920/`。
原始输出：`data/quality-runs/task-matrix-20260920/`。
验收规则与摘要：`artifacts/task-matrix-20260920/`。

这些是作者控制的合成数据，分训练、验证、保留测试集，**不是外部盲测**。
GPU 训练仅使用授权的 rwkv-8222 GPU 3；基础模型冻结，记录训练前后权重一致性。
保留测试文件没有上传到训练服务器。验证集允许选择配置，保留集不得用于回头调参。

已观察结果：

| 阶段 | 结果 | 结论 |
| --- | --- | --- |
| Reader 原配置，验证 | 56/64；1 个误选、7 个漏选 | 基线 |
| Reader v1，验证 | 64/64；无误选和漏选 | 当前课程通过 |
| Reader v1，保留集 | 64/64；无误选和漏选 | 当前课程通过 |
| 共享 Planner v1，规划 | 1/8 | 不采用 |
| 独立 Planner v2，规划验证 | 8/8 对象和属性覆盖 | 条件与跨领域泛化另验 |
| Assessor v1 | 31/32；漏判一次冲突 | 不能据此丢弃 Reader 证据 |
| 整段 Reviewer v1 | 11/30；18 次错误放行 | 不作为质量门槛 |
| Writer v1 | 4/8 | 零值等发生回归，不采用 |
| Writer v2 | 7/8 | 单侧缺资料题漏引用，未过无回归门槛 |
| Writer v3，固定材料验证 | 8/8 | 当前课程通过 |
| Writer v3，固定材料保留集 | 8/8，逐字与目标相同 | 当前课程通过，不代表外部泛化 |
| Writer v3，旧跨领域题 | 6/8 | 普通 evidence-first 提示下仍有引用及忽略废止值问题 |
| 逐项 Reviewer v2，全局证据 | 105/128；20 次错误放行、3 次误拒绝 | 未过门槛 |
| 同一 Reviewer，仅改逐项证据范围 | 107/128；4 次错误放行、17 次误拒绝 | 仍未过门槛，单独训练新输入 |

后续逐项 Reviewer、Writer 追加训练及端到端结果以对应原始输出与结果摘要为准。
禁止仅凭训练 loss 降低宣布质量修复，也不能凭当前小样本宣布商用成熟。

## 已完成的工程验证

- task-matrix / batch transport：89 项通过。
- 原有 pipeline / repository / web retrieval / writer checked / quality gate：144 项通过。
- 前端测试、TypeScript/Vite 构建通过；构建仍有现存大 bundle 警告。
- 浏览器以拦截 API 的明确展示夹具检查表格、原文抽屉和失败状态；不充当真实 RAG 验收。

## 上线门槛

先固定材料 Writer，再 Reader、真实检索和端到端；同时检查旧失败题与非对比回归。
逐项 Reviewer 必须在故意放错值、引用和漏项的测试中证明不会错误放行。
适用门槛未通过时保留实验配置，不自动改生产默认值。每次发布需保存配置和静态资源回滚点。
