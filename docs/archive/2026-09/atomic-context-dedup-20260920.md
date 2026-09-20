# 原子证据表头去重修复与本地引擎核对

历史记录，2026-09-20。当前状态见 [CURRENT](../../CURRENT.md)。代码修复协议为 `atomic-evidence-v7`；本轮未执行模型质量或采样参数对照。

## 修复结果

`candidates()` 现在先调用 `attach_context()`，再排序、计算预算并构造模型输入。同一表头被 chunk 内部识别和父级元数据重复附加时，仅移除能够证明已由元数据覆盖的本地范围。

合并必须满足：chunk 的文档 ID、Unicode 字符偏移、长度和内容哈希匹配；父级范围落在该 chunk 内、内容与切片逐字一致、哈希匹配，且没有显式来源/版本冲突。不同来源、不同范围的同字表头保留；无绑定、错哈希、错偏移或错单位时保守保留。外部标题仍按原顺序保留，不拿字符串相同当重复依据。

原始 metadata、snippet、答案行和偏移不修改。证据记录中的 `context_deduplication.removed_local_ranges` 保存去掉的重复范围及所对应的 `context_index`，便于追溯。预算和 Selector/Reader 使用同一份 context；真正超限仍明确过滤并记录覆盖范围，不截断表格。

| 上轮同一宽表材料 | 修复前 | 修复后 |
| --- | ---: | ---: |
| 答案行携带的上下文字符数 | 909 | 459 |
| 600 字符限制下答案行是否保留 | 否 | 是 |
| 表头来自几份独立原文范围 | 1 | 1 |

[逐字修复前后记录](../../../artifacts/atomic-context-dedup-20260920/BEFORE-AFTER.json)引用旧检查材料；旧记录不覆盖。去重不提供新的语义证据，也不能解释前轮绕过候选路径的短表格判断失败。

## 验证

65 项通过、0 跳过，见[完整测试输出](../../../artifacts/atomic-context-dedup-20260920/tests.txt)。覆盖普通/宽表、多级标题、Unicode 偏移、原文不可变、来源/版本/哈希/单位冲突、不同范围同字表头和真实超限。真实 OpenSearch 与 MongoDB 集成验证了检索后答案行保留、原始元数据不变、去重审计和历史结果持久化。

```bash
PYTHONPATH=llamaindex-retrieval/src \
RWKVRAG_TEST_OPENSEARCH_URL=http://127.0.0.1:18438 \
RWKVRAG_TEST_MONGO_URL=mongodb://127.0.0.1:18439 \
llamaindex-retrieval/.venv/bin/pytest -q \
  llamaindex-retrieval/tests/test_atomic_context_dedup.py \
  llamaindex-retrieval/tests/test_atomic_evidence.py \
  llamaindex-retrieval/tests/test_atomic_evidence_integration.py \
  llamaindex-retrieval/tests/test_verbatim_chunking.py
```

集成测试使用独立随机名称的数据库和索引并清理自己创建的数据。模型回复是固定测试桩，不能把测试通过写成模型回答质量改善。

## 部署范围

修复提交 7a0df088 已推送。原子证据预览 18445 于 2026-09-20 13:35:51（Asia/Shanghai）重启，进程 active，能力接口返回 available=true。正式应用 18440 未重启，模型和采样配置未改变。这是代码加载/健康检查，不是新模型质量验收。

## 本地 vllm-rwkv：实际应以用户的优化实现为准

用户明确要求核对本地引擎，后续以 `/home/chase/GitHub/vllm-rwkv` 为依据，不以官方示例推断这套引擎。审查基线 HEAD 为 `4ee959dfa`，工作区已有用户未提交修改；本轮只读，未改动或提交该仓库。实际读到的文件哈希见[本地审查绑定](../../../artifacts/atomic-context-dedup-20260920/LOCAL-ENGINE-REVIEW.json)。该记录不是远程部署身份核验。

### 已核实的调用链

- `RwkvModelState.custom_sampler()` 创建 `RwkvSampler`，不是只走通用 vLLM sampler。
- `RwkvSampler` 调用 FlashRWKV2 的 `infer_sampling_six_parameter_forward_varlen`；CUDA Graph 路径也把对应参数复制到图输入。
- temperature、top_k、top_p、presence_penalty、frequency_penalty、penalty_decay 六项进入采样内核。frequency_penalty 对应逐 token 累积的加性惩罚，不能当作乘法 repetition_penalty。
- 采样随机状态与惩罚按请求槽隔离；新请求用其 seed 初始化，抢占路径保存、恢复两种状态。这是代码核对，未做新的 GPU 行为验收。
- 服务按模式读取 artifact 的 Open Think、Fake Think、Tools 配置；有 tools 时优先 Tools，`chat_template_kwargs.rwkv_generation_prompt=fake_think` 选择 Fake Think，显式请求参数覆盖配置。`--generation-config vllm` 可禁用 artifact 配置。
- 本地 requirements 和已安装包源码均指向 FlashRWKV2 `0.1.0a13`；文档安装段仍提到 a11，不能以那一行作为运行版本证据。

### 本地文档列出的三套配置

以下是该仓库文档中的 artifact 配置约定，不是本轮从正在运行的模型 artifact 读取的生效参数，也不是我们的质量结论。

| 配置 | temperature | top_p | top_k | presence | frequency | decay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Open Think | 0.96 | 0.76 | 32 | 1.0 | 0.1 | 0.988 |
| Fake Think | 1.0 | 0.28 | 32 | 0.0 | 0.0 | 1.0 |
| Tools | 0.96 | 0.76 | 32 | 0.0 | 0.0 | 1.0 |

### 贪心基准不能照搬接口习惯

本地已安装 FlashRWKV2 CUDA 源码将 temperature 下限钳到 0.001；`SamplingParams` 对 temperature=0 会把 top_k 改为 0，后续映射为完整词表。本轮查看的 Rapid 路径直接传递这些向量，不能仅凭“温度 0”声明与严格 argmax 完全等价。

用户仓库的 `benchmark_stateful_chat.py` 已明确用 **temperature=1.0、top_k=1** 做 top-1 基准。后续应沿用其约定，并显式固定 top_p、三项惩罚、stop、seed、输出预算与 prompt/token IDs，防止 artifact 默认值引入额外差异。top-1 下若存在分数并列，也仍需检查与旧服务的具体打破并列方式，不预先承诺逐 token 一致。

此外，RWKVoS 适配器当前发送 top_p=0、alpha_presence 等字段，而本地 vLLM 的 SamplingParams 要求 top_p 在 (0,1]，使用 presence_penalty 等字段。两套协议和模板不能原样替换端口后当作同一个实验。

## 对“换参数是否更好”的结论与下一步

**本地引擎确实支持有效的采样和专项优化；还没有这批 RAG 材料上优于 top-1 的实测结果。** 之前 7.2B 标签结果来自项目内的简化 argmax 服务，不是用户的 vllm-rwkv，只能代表那套固定执行路径。

下一步先核实实际服务加载的本地代码/依赖/权重、模型 artifact 与模板，建立本地引擎的 top-1 基线；模型/State、tokenizer、精度、模板或 stop 改变都要单独记录。然后分别测试呈现格式和解码策略，不能一次更换引擎、模板、标签及采样后把收益归到温度。

解码对照应优先参考用户已有的 Fake Think 配置（适用于本项目当前不输出思考的判断任务），但沿用相同输入 token 序列；若模式开关同时改变模板，须拆开验证。完整配置比较是“策略组合比较”，不是“只改温度”。再做温度或惩罚单变量消融。随机组使用事先固定的多个 seed，统计每次调用平均正确率、误选、格式失败、重复输出/预算耗尽、耗时和波动，不挑最好的一次，也不把重复调用当新增独立材料。

本轮保留旧评测服务和正式配置。采样对照尚未冻结数据和执行，不宣称已经找到更优参数。
