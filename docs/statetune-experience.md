# RWKV StateTune：数据集如何构建，用什么训练

这轮工作的核心是：从RWKV 2.9B真实RAG调用的trace中找错，依据当时可见的证据修正输出，用这些纠错样本指导构造2000条训练数据，再使用服务器上的RWKV-PEFT模型实现训练初始state。数据构建与训练入口都放在本项目，以下说明实际做法。

## 1. 从真实trace建立纠错种子

首先保存每个阶段真正收到的完整prompt、历史、证据、模型原始输出及来源哈希，逐阶段定位问题。Writer引用错误、Reader漏选证据、Planner丢失当前任务，分别建立对应阶段的样本。

本轮由本任务的作者代理依据原文编写和修正目标，再交给独立审查子代理复核。数据扩展由Python脚本按受控规则完成。生成与复核的角色分开，不能用作者再次自查替代独立审查。

一条纠错种子包含：

| 字段 | 保存什么 |
|---|---|
| `original_prompt` | 失败调用真正收到的完整输入 |
| `original_wrong_output` | 原始错误输出，留作分析 |
| `corrected_output` | 只依据该调用可见证据修正的正确输出 |
| `correction_reason` | 错在何处、为何这样修正 |
| `generation_invariants` | 后续构造同类问题时应保留的条件 |
| `source_trace`、各项SHA | 原请求、响应和修正目标的溯源信息 |

本轮得到17条纠错种子。实际例子是同一问题同时询问A款泵2022年、A款泵2023年和B款泵2023年的流量及离线能力。原模型把A款2023年错引到资料2，又把B款2023年错引到资料1。正确目标为：

```text
A款2022年：额定流量8 L/min，不能离线运行。[资料 1]
A款2023年：额定流量6 L/min，支持离线运行。[资料 1]
B款2023年：额定质量流量4 kg/min，不能离线运行。[资料 2]
```

这个种子要教的是“按支持事实的来源绑定引用”，因此派生数据会同时变化对象、年份、资料排列和问题顺序，避免模型只学会按回答顺序写1、2、3。

另一类错误是Reader只看到了“速度是传统CAN的5倍”，却把它判定为能回答“绝对速率是多少Mbit/s”。正确标签是NO，因为该单元没有绝对值或计算基数。即便其他未传入单元有答案，也不能把它倒灌进这条训练目标。

纠错种子在 [seeds.jsonl](../llamaindex-retrieval/statetune/datasets/trace-v1-2000/correction-seeds-v1/seeds.jsonl)，构造入口为 [build_correction_seeds.py](../llamaindex-retrieval/statetune/datasets/trace-v1-2000/build_correction_seeds.py)。错误输出不会进入派生样本的模型输入或训练target；17个种子也不计入2000条派生数据的数量。

## 2. 围绕缺陷构造2000条数据

最终构建入口为 [build_v6_seeded.py](../llamaindex-retrieval/statetune/datasets/trace-v1-2000/build_v6_seeded.py)。脚本读取纠错种子、已复核来源与事实标注，构造问题、证据、历史和正确目标，再调用本项目实际使用的阶段prompt函数。

来源分为真实资料与明确标注的人工场景。真实资料取自冻结的FineWiki快照；信息框字段可以转换为带来源的表格行，正文事实保留依据。人工场景用于构造单位、版本、相对量与绝对量、部分缺失等对照，不伪称为自然发生的用户问题。

| 阶段与缺陷 | 条数 | 具体构造方式 |
|---|---:|---|
| Writer：引用绑定 | 250 | 交错排列多个对象的证据，改变首个答案对应的资料编号 |
| Writer：对象与字段绑定 | 300 | 同时提供多个对象、日期、状态或版本，要求取指定字段 |
| Writer：单位与限定条件 | 150 | 保留位/字节、比例、费用范围等差异，不擅自换算或扩大条件 |
| Writer：缺失证据 | 300 | 构造空材料、部分缺失、同主题无答案等情况；有据部分正常答，缺失部分明确说明 |
| Writer：多问覆盖 | 150 | 同时要求多个独立字段，正确目标逐项回答 |
| Writer：历史更正 | 150 | 保留更正、撤回和无关历史，最新问题间接指向最终有效任务 |
| Writer：回答后结束 | 100 | 放入大量未要求的事实，目标只回答当前问题并结束 |
| Reader：漏选正例 | 150 | 单个证据单元直接给出所需事实，目标为YES |
| Reader：相对量、绝对量及范围对照 | 300 | 混合正反例，变化相对倍数、绝对值、版本和缺失字段 |
| Planner：任务与输出协议 | 150 | 变化1–6个有效对象、2–12个字段，加入撤回历史，目标保留当前检索任务 |
| **合计** | **2000** | **Writer1400、Reader450、Planner150** |

其中774条重新构造输入，1226条沿用已复核的受控实例并补充种子关联。这不是2000条全新的真实失败trace。种子提供设计指导，每条样本用 `causal_setup` 单独说明实际构造条件，不能声称它覆盖了父种子的所有缺陷。

训练真实来源为313页，与通用200题评测所用30页按来源文档隔离。参与纠错的旧题属于暴露回放材料，不能再当作独立泛化测试。

## 3. 一个样本保存什么、各阶段学什么

可阅读样本保存在 `draft-v6-seeded/{writer,resolver,planner}.drafts.jsonl`。每行包括：

- `id`、`stage`、`split`、`primary_failure_family`：身份、阶段和缺陷类别。
- `prompt`、`target`：完整实际输入与应学习的正确输出。
- `question`、`history`、`evidence`，或Reader对应的`queries/source/contexts/text`：构造输入所用材料。
- `origins`、`anchor`、`parent_corrected_seeds`：来源文档、原始trace和父纠错种子。
- `prompt_sha256`、`target_sha256`、token计数与`causal_setup`：字节绑定和审计依据。

三个阶段分别生成、分别训练state：

| 阶段 | 输入 | 正确target |
|---|---|---|
| Writer | 问题、完整历史、带资料编号的证据 | 有依据、覆盖有效问题、逐事实引用的答案正文 |
| Reader（代码中记为`resolver`） | 待查问题、单个来源、父级上下文和原文 | `{"answer": "YES"}` 或 `{"answer": "NO"}` |
| Planner | 最新问题和历史 | 恰有`queries`与`fields`两个字符串数组的JSON |

例如Reader看到《克赖沙》的时区原文为 `CET（UTC+1）`，问题问时区，target就是 `{"answer": "YES"}`；它学习的是该片段能否支持问题，不在这一步生成最终答案。

Writer使用 `writer_prompt_v2`，Reader使用 `binary_query_prompt`，Planner使用 `planner_prompt` 的 `queries_fields` 配置。模板直接复用生产代码，避免数据脚本另写一套格式。

## 4. 对齐训练格式，再复核发布

本轮重点对齐完整prompt、Assistant前缀、换行和EOS。Writer/Reader的前缀结尾为：

```text
\n\nAssistant: <think></think>\n
```

Planner使用对应的无末尾换行形式。两者都经过同一固定词表编码，不能把一种阶段格式直接套到另一种阶段。

[state_tokens.py](../llamaindex-retrieval/src/llamaindex_retrieval/state_tokens.py)分别编码prompt与target，再追加EOS=0。训练数据的核心关系是：

```text
input_ids = prompt_tokens + target_tokens + [0]
labels    = [-100] × prompt长度 + target_tokens + [0]
模型输入   = input_ids[:-1]
预测标签   = labels[1:]
```

模型读取完整prompt，但只对正确target和结束符计算loss。超出预算直接拒绝，不截断答案。配置上限为8192 tokens，本轮Writer实际最长训练行5012 tokens；这个数字不代表所有8192-token输入都经过验证。

进入训练前进行来源、split、ID、prompt/target哈希、词表、EOS、mask和阶段输出协议检查。独立审查子代理复核2000条样本的语义与缺陷构造，通过后由 [package_release.py](../llamaindex-retrieval/statetune/datasets/trace-v1-2000/package_release.py)打包：

```text
release-v1/
  writer-100/
  writer-300/
  writer-600/
  writer-1400/
  resolver-450/
  planner-150/
```

每包都有 `train.tokens.jsonl`、`INDEPENDENT-REVIEW.json` 和 `RELEASE.json`。Writer按缺陷类别比例确定顺序，100/300/600是1400中的嵌套子集，选择发生在这轮新评测输出之前。训练发布包只向训练入口提供train token，不提供评测gold。

草稿保留生成时的`independent_review_pending`字段，独立审查与发布结论保存在外部复核和release文件中，不通过事后改写草稿来破坏原始哈希。复核记录是可追溯声明，不是身份认证签名。

## 5. 实际使用什么项目训练

**使用服务器上的RWKV-PEFT环境与模型实现，由本项目的训练脚本调用。** 实际位置如下：

| 项目或入口 | 本轮作用 |
|---|---|
| `/home/chase/chase/RWKV-PEFT` | 服务器上的训练项目与已安装环境 |
| `/home/chase/chase/RWKV-PEFT/.venv/bin/python` | 实际执行训练的Python |
| `/home/chase/rwkvrag/llamaindex-retrieval/statetune/train_trace_state.py` | 本轮正式入口，检查数据发布包、配置、底模及预检绑定 |
| `statetune/train_released_state.py` | 复用其中的loss计算、梯度累积、优化器更新及保存逻辑 |
| `statetune/pilot_runtime.py` | 从冻结RWKV-PEFT源码加载`RWKV7`，冻结底模、初始化可训练state |
| `data/experiments/state-protocol-20260909/training-v2/execution-source` | 实际导入的RWKV-PEFT模型源码快照 |

训练循环是本项目调用PyTorch编写的受控入口，没有直接运行上游通用训练launcher。计算后端使用RWKV-PEFT的FLA路径，设置 `WKV=fla`、`RWKV_TRAIN_TYPE=state`、`FUSED_KERNEL=0`。

环境固定为PyTorch2.9.0、Triton3.5.0、rwkv-fla0.7.202508221413、DeepSpeed0.18.1、einops0.8.1、NumPy2.3.4。源码快照按manifest逐文件哈希检查，不能仅凭RWKV-PEFT目录名判断实际训练实现。

底模为 `rwkv7-g1j-2.9b-20260831-ctx16384.pth`。每组从零初始state独立开始，只优化32层 `blocks.N.att.time_state`，每层形状为`[40,64,64]`，共5,242,880个FP32参数。底模使用BF16并保持冻结；最终保存的是state文件。

## 6. 训练具体怎么执行

执行顺序为：CPU检查发布包与全部绑定 → 用最长训练行做一次零更新反向预检 → 核验梯度、state与底模 → 正式训练 → 保存state并读回核验。

每组在SSH别名`rwkv-8222`、实际hostname为`rwkv-82`的服务器物理GPU3上执行。配置统一为1 epoch、学习率1e-5、每次处理1条样本并累积2条后更新、样本洗牌seed为20260911；运行时`torch.manual_seed`为20260909。使用AdamW，`betas=(0.9,0.999)`、`eps=1e-8`、`weight_decay=0`，梯度L2裁剪上限1.0。每条样本开始使用该组正在学习的初始state，不将上一条样本的动态阅读状态当作下一条输入。

| 训练包 | 样本数 | 优化器更新次数 |
|---|---:|---:|
| Writer100 / Writer300 / Writer600 / Writer1400 | 100 / 300 / 600 / 1400 | 50 / 150 / 300 / 700 |
| Reader450 | 450 | 225 |
| Planner150 | 150 | 75 |

计入Writer子集重复实验，共3000次样本反向计算、1500次更新，另有6次零更新预检。结束后逐张量核验1062个底模张量保持不变。每组保存step0、最终state、更新记录和完成回执。

命令入口的形式为：

```bash
/home/chase/chase/RWKV-PEFT/.venv/bin/python \
  /home/chase/rwkvrag/llamaindex-retrieval/statetune/train_trace_state.py \
  --config /path/to/reviewed-config.json \
  --config-sha256 CONFIG_SHA256 \
  --output data/experiments/new-run/writer-300
```

这是参数形式示意；实际配置必须匹配新的时限、发布包、源码哈希和预检，并设置绑定物理GPU3 UUID的`CUDA_VISIBLE_DEVICES`。旧配置已过期，脚本也会拒绝占用中的GPU，不能直接重用旧命令启动。实际训练计划见[TRAINING-PLAN.json](../llamaindex-retrieval/eval/trace-training-20260911/TRAINING-PLAN.json)，完成核验见[TRAINING-VERIFIED.json](../llamaindex-retrieval/eval/trace-training-20260911/TRAINING-VERIFIED.json)。

## 7. 这套数据和训练方法带来了什么

Reader在60道新受控挑战中由48/60提高到60/60。Writer正常EOS从零state的155/200提高到Writer600的200/200；固定36题独立语义抽样中，零state有据可用10/36，Writer300为17/36，Writer1400为15/36。Planner合法结构输出从21/24降到16/24。

这说明针对trace缺陷构建的数据能改善部分行为，但数据增加不保证正确性持续提升。正常结束、格式合法和答案正确需要分别评估；来源隔离的实际问答测试仍是判断训练是否有效的依据。完整分组结果见[本轮评测](../llamaindex-retrieval/eval/trace-eval-20260911/RESULTS.md)。原始trace、旧草稿、冻结运行源码和六份state按[实验附件说明](artifacts.md)恢复。
