# 固定材料质量验收（P0 第一批）

这一入口用于发现 Writer 失败和防止错误放行。它复用现有 `run_smoke.py`、8 道开发题、gold 与调用收据，不运行检索，也不把开发题当作留出集。Planner/Reader/完整 RAG 的统一质量入口仍待补齐。

## 1. 执行固定材料基线

从 `llamaindex-retrieval` 目录执行。以下示例显式对齐应用的 canonical Writer 模板；其他环境需要提供相同模型及有效 state 的服务地址。

```bash
.venv/bin/python eval/native-smoke/run_smoke.py \
  --base-url http://127.0.0.1:18423/v1 \
  --model rwkv7-g1j-2.9b-20260831-ctx16384 \
  --transport rwkvos_batch \
  --writer-prefill '<think></think' \
  --writer-prompt-protocol evidence_first \
  --writer-transport-protocol rwkv_g1j_no_think_v1 \
  --prefill-mode complete --writer-state-id writer-trace-300 --stop-tokens '[0]' \
  --concurrency 2 --batch-size 2 --max-output-tokens 2048 --timeout-seconds 180 \
  --output ../data/quality-runs/NEW-RUN
```

输出目录必须不存在。每次比较使用新目录；不自动重试或覆盖旧结果。源码、题目、请求、回答和模型调用保存在运行目录，gold 不进入模型输入。`--writer-prompt-protocol` 现在显式控制 Writer 提示词，manifest 保存该选择；`--writer-transport-protocol` 独立控制底层模板（canonical 在回答前缀后保留训练所需换行）；`--writer-state-id` 指定 Writer 专用 state。旧实验的全局 `--state-id` 仍可重放。manifest 与 trace 中的协议和有效 state 必须相符。

当前示例没有开启服务端输入 token 计数；`--context-window` 不代表外部 batch 服务一定执行相同硬限制。外部终止原因仍未独立核验。单轮基线不证明 state、prompt 或模型已经最优。

## 2. 自动检查并产生待审核表

```bash
.venv/bin/python -m llamaindex_retrieval.quality_gate \
  --run ../data/quality-runs/NEW-RUN \
  --output ../data/quality-runs/NEW-RUN/quality-automatic.json \
  --review-template ../data/quality-runs/NEW-RUN/review-template.json
```

检查包括：

- 固定 8 题分母和顺序，冻结题目/源码/runner 的哈希、原文引用坐标、开始/完成收据与结果一致性。
- 请求只含题目、历史、材料和 ID；响应材料与输入材料相同。
- 模型调用确实完成、无规划回退或未处理的阶段解析失败。
- 原始答案未改写、正文范围有效、模型身份字段符合该次 manifest。
- 从正文重新解析引用编号，确认编号有对应来源；不信任回答自己声称的“校验通过”。

这些检查不证明事实正确。没有引用也不自动判成有效拒答；需要下一步结合题目和材料复核。

## 3. 逐题语义复核

另存 `review-template.json` 为 `review.json`，填写 `reviewer`、`reviewed_at`，逐题核对冻结题目、完整历史、实际材料和最终正文。不要把思考内容当作已经回答的事实。

每题检查：

- `decision_correct`：应该回答、部分回答、拒答或澄清的决策是否正确；不代替事实完整性。
- `facts[].correct`：每项必要事实是否正确且确实写入答案。
- `facts[].citation_supported`：答案对该事实的实际引用是否存在并支持它；漏答或漏引用填 `false`。
- `no_unsupported_claims`：额外解释、数值和其他主张是否也有原文支持。
- `no_uncontrolled_repetition`：是否没有失控重复；轻微重复应在 notes 中单独记录，避免任意扩大失败口径。
- `notes`：用具体原文位置或事实解释判定，不能只写“通过”。

允许同一明确引用支持一个紧密关联的列表，不强制每一行重复同一编号；若引用范围不清或来源错误则不通过。没有正向事实的拒答题不计入事实/事实引用指标的已评数量，仍必须审核拒答决策和是否有无据主张。

保留 `manifest_sha256`、每题 `response_sha256`、题目与事实 ID。任何判据尚未审核则保持 `null`；工具不会补成 `true`。审核者须真实署名，本次主代理复核不等同于第二位独立审核者。

```bash
.venv/bin/python -m llamaindex_retrieval.quality_gate \
  --run ../data/quality-runs/NEW-RUN \
  --review ../data/quality-runs/NEW-RUN/review.json \
  --output ../data/quality-runs/NEW-RUN/quality-reviewed.json
```

## 4. 结果和退出码

| 退出码 | 含义 |
| --- | --- |
| 0 | 本组开发题的自动检查和全部显式语义判据通过 |
| 1 | 至少一题自动检查或已完成的语义判据失败 |
| 2 | 文件、哈希、格式、审核绑定等无效，不能作为验收结果 |
| 3 | 自动检查未发现失败，但仍有语义审核未完成 |

每项指标以题为单位，分别报告 `passed / assessed / total`。未审核不能从总分母消失；失败或未开始的调用也保留其题目位置。报告绑定运行 manifest、审核文件和验收代码 SHA。哈希用于一致性核对，不是服务端硬件身份认证，也不能证明审核者判断正确。

新增 schema 当前只接受固定材料 smoke；不能直接拿其他实验的 JSON 当作同一验收输入。程序退出码 0 也不是整个产品的发布认证。

## 首轮记录

2026-09-16 首轮指定 2.9B + Writer300 **legacy 模板诊断基线**（不是应用 canonical 配置）：8/8 执行完成，主代理按冻结 gold 复核后 3/8 通过本组全部质量要求。详见 [P0 首轮结果](../../../docs/p0-quality-baseline-20260916.md)。

## 第二轮实验

新增独立冻结的 8 题组；开发题和留出题分别计分，不能混池。详见 [第二轮报告](../../../docs/p0-writer-experiment-20260916.md)。`evidence_checked` 是未获准采用的实验提示词，不是默认推荐配置。引用审计新增非法编号检查；原文逐字匹配只作为独立诊断，不等同于语义支持判断。
