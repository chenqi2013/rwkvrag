# bm250820 原生 RWKV RAG：已完成的开发实验

这份归档记录从 `chenqi2013/rwkvrag` 的 `bm250820` 基点 `2bbc406125e7e030eda98b11993dc13fc4534ca4` 开始的第一轮重建。它保留失败、原始请求/响应、冻结源码、索引核验和逐题审阅。**这里的两批真实 Wiki 端到端测试，完整通过均为 0/8，系统尚未达到可用的问答质量。**

当前包只包含下表列出的已完成实验。后来新增的 Resolver、Planner 或端到端条件不在本包结果内；当前工作树的解析器和提示词也不能代替归档中的冻结版本。

## 运行与结果

| 记录 | 范围与变量 | 实际生成 / 模型 HTTP 尝试 | 保留的结果 |
|---|---|---:|---|
| `materials-smoke-v1` | 8 题固定材料 Writer，最初传输尝试 | 0 / 8 | 8 次均在 tokenize 传输读取失败，completion 没有发出；不能算 8 次模型生成 |
| `materials-smoke-v2` | 同 8 题，在服务器执行 | 8 / 16 | 8 completed；事实 27/28 正确；正式引用完整通过 1/8，明确人读引用口径 2/8 |
| `materials-smoke-v3` | 同 8 题，仅材料标签形式对照 | 8 / 16 | 7 completed、1 length；事实 24/28 正确；两种引用口径完整通过均 3/8；未满足预先规定的全部采用条件，标签改动撤回 |
| `wiki-smoke-v1` | 5000 篇索引上的 8 题完整链路 | 33 / 66 | 7 个 Planner 达到 1024 token 上限；仅 1 题进入检索。事实 0/29，完整通过 0/8 |
| `wiki-smoke-v2` | 同 8 题，仅 Planner 改为 closed thinking prefill | 208 / 416 | 8 个 Planner 均完成；192 个 Resolver 中 21 length；8 个 Writer 完成。事实 2/29 正确、25 遗漏、2 错误；正确且有支持引用 1/29，完整通过 0/8 |
| `resolver-reminder-experiment` | 重放 v1 同一题的 24 份 Resolver 输入，仅追加编号提醒 | 24 / 48 | 原严格协议下完整选择 22/24 → 7/24；关键来源的 3 项事实仍未成功交付；未采用提醒 |
| `literal-quote-feasibility` | v2 全部 192 次 Resolver 的离线逐字定位诊断 | 0 / 0 | 原规则接受 127 次，拟议字面规则接受 130 次；新增绑定仍有不支持指定字段的情况，不能算新增正确答案 |

以上合计 **281 次实际生成、570 次模型 HTTP 尝试**，其中最初 8 次 HTTP 传输失败，后续 562 次 HTTP 返回成功。固定材料测试没有自动检索；24 来源重放不是 24 道端到端题。Wiki 两批各保留 8 题、29 个必需事实的原分母；不从 thinking 中补算答案。

`completed` 只表示模型传输/结束状态，不表示事实正确或引用支持。审阅由 Codex 代理逐项核对实际答案及来源完成，包含作者审阅和其他代理复核，属于透明、非盲的开发审阅，不称为人工盲测。材料与聊天历史是合成开发场景；Wiki 文档来自公开原文。所有题目均已曝光，不能再当作封存泛化测试。

## 证据链定位

冻结链路为 Planner → OpenSearch BM25 → 并行 Resolver → Writer。Writer 只获得 Resolver 选中的逐字片段和真实父级上下文；代码验证编号、来源身份、偏移与预算，不替模型决定事实是否支持字段。该路线没有固定“每文档最多 2 块”配额，也未调用向量 embedding、Qdrant 或 MongoDB。

v1 唯一进入检索的 Ghost 题，三个事实的原文已经位于候选第 4 位，并完整发送给 Resolver。Resolver 最终输出事实值而非 E 编号，原解析器整段拒绝，Writer 最终拿到零证据。这是阅读结果交接失败，不能记为 BM25 未找到事实。只有标题的另一来源还生成了无据数字，原解析器也拒绝了它。

v2 证明 closed Planner prefill 能在该批中避免计划阶段的长度失败，但没有解决计划语义、Resolver 选择和 Writer 引用问题。例如一题的 Planner 恢复了用户已经排除的字段；另一题的 Writer 在零证据时仍给出了错误具体答案。多阶段的格式成功不能替代最终质量。

编号提醒重放使用原冻结解析器。正确 E 编号被挤在一行仍判失败，旧结果保持原样。之后实现的逗号/分号语法版本不回写这批分数；对旧输出用新解析器再解释也只能作为离线诊断，必须通过独立的新端到端运行验证实际影响。

## 冻结索引

索引 `rwkvrag-bm250820-wiki-5000-20260908`，UUID `AaG0YmAZT5i8EHEhJHyNYw`，KB `wiki-5000-20260908`，OpenSearch 3.8.0。新建前保留 404 收据，`recreate=False`，没有重建或删除旧索引。

源顺序前 5000 篇原始 FineWiki 文章形成 **45,960 个节点**。全量 scroll 核验了页面/版本身份、原文和每块 SHA、Unicode 字符偏移、78,551 个父级上下文片段，覆盖全部 6,437,704 个原始字符。正文软窗口为 2400 字符、重叠 180；表格、列表、代码等结构块可以超过软窗口，模型输入硬上限单独处理，不静默截断。

这次保守切分也产生了 19,785 个标题节点和 2,797 个仅空白节点。它们意味着可能多做阅读调用；不能仅凭短标题就断言丢失了事实。完整 scroll 保留为 `records/index-import-v1/indexed-records.jsonl`，未压缩 126,610,875 字节，SHA256 `e649fd2991bf7228d0bc73ece4b415cf3f2b14c8d795add1ef27f0054a7077d8`。索引完整性通过不等于检索或答案质量通过。

## 文件与复算

[ARCHIVE-SELECTION.json](ARCHIVE-SELECTION.json) 固定 1,897 个原始文件，共 212,288,427 字节；每项有相对路径、大小、SHA256 和来源根目录。被排除的缓存、虚拟环境、环境文件、重复传输包、可变进度及未完成实验逐项记在 `exclusions`。没有改写原始收据，也没有从响应中删改内容。选中内容通过常见密钥/私钥模式、非空认证字段和 base64 wire body 扫描；检测结果不声称能排除所有未知形式的敏感信息。

[RECORDS-MANIFEST.json](archives/RECORDS-MANIFEST.json) 绑定每个归档及每个解压成员，SHA256 为 `ea872a9349b92116de43704591317cbe758e16b9f1a1d9062f72dfccdbebb0e1`。全部归档已经重新打开，逐文件复核大小和 SHA；每个压缩文件均小于 50,000,000 字节。

| 归档 | 压缩字节 | 内容 |
|---|---:|---|
| [checks.tar.gz](archives/checks.tar.gz) | 94,932 | 原基点与当前 pytest 原日志、失败 ID 对比 |
| [index.tar.gz](archives/index.tar.gz) | 23,985,157 | 完整索引 scroll、导入/核验记录、冻结源码、5000 页归属清单、FineWiki 原数据卡 |
| [materials.tar.gz](archives/materials.tar.gz) | 4,292,856 | 三批固定材料全部收据、冻结源码、传输失败、审阅与采用判据 |
| [wiki.tar.gz](archives/wiki.tar.gz) | 9,654,888 | 两批完整 Wiki 链路、失败预检、源码冻结、HTTP/阶段审计及语义审阅 |
| [resolver-reminder.tar.gz](archives/resolver-reminder.tar.gz) | 942,126 | 24 来源重放的预登记、基线收据、差异证明、原始 wire 和审阅 |
| [literal-quote.tar.gz](archives/literal-quote.tar.gz) | 115,626 | 192 次离线字面定位规则、结果、语义限制和两个 v1 控制 |

从本目录运行，只需 Python 标准库，不访问模型、索引或网络：

```bash
python3 build_archive.py verify --manifest archives/RECORDS-MANIFEST.json

# 解压目标必须不存在。校验全部归档后才创建目标，拒绝路径穿越、重复、链接和额外成员。
python3 build_archive.py verify --manifest archives/RECORDS-MANIFEST.json \
  --extract /tmp/bm250820-records-new

# 原 B/W 文件仍在时，按已固定的 selection 重建到一个新目录。
python3 build_archive.py build \
  --records /home/chase/GitHub/rwkvrag-bm-rebuild-20260908 \
  --corpus /home/chase/GitHub/rwkvrag-wiki-rag-20260908/corpus \
  --selection ARCHIVE-SELECTION.json --output /tmp/bm250820-archives-new
```

gzip 时间戳、tar 成员顺序与权限固定；选中文件变动会导致失败。归档根目录分别为 `records/` 和 `corpus/`，原相对路径保留。压缩封装改变的是存储方式，不是文件内容。`build_archive.py prepare` 仅用于重新冻结允许范围，不是修改旧 selection 的操作。

这个校验器只复算字节完整性，不自动替代原有 runner、索引偏移审计或语义审阅。原源码 manifest 中的本地绝对路径不作替换；它们记录当时环境，外部机器不能凭这些路径直接运行。重跑模型仍需要相同的冻结源码、依赖、模型/分词器和索引配置；模型权重、服务运行时及大 parquet 不装入本包。原始 FineWiki 每页正文文件也未单独重复装包；索引 scroll 保存了实际索引全文片段。

失败证据包括 Wiki 首次 CPU 预检的 runner 参数记录错误、材料传输/收集失败，以及编号提醒重放第一次 systemd 在 Python 启动前的 `209/STDOUT` 失败。后者由执行方未等待传输完成引起，零模型调用；随后唯一实际重放批次没有模型重试。不能用后续成功记录覆盖失败批次。

## 测试边界

干净基点为 **176 passed / 114 failed**；当前冻结比较为 **408 passed / 114 failed**。114 个失败测试 ID 完全一致，新增 232 个通过项，既有失败没有隐藏或修复。原日志及比较 JSON 在 `checks.tar.gz`。这是回归差异说明，**不是全量测试通过**，更不是 RAG 正确率证明。

## 原文来源、归属与许可

文本原作者为 **Wikipedia contributors**；FineWiki 整理发布由 **Guilherme Penedo / HuggingFaceFW** 完成。固定数据集 revision 为 `8bd13e72e6a002407649b3e898535f42ceb1aeb9`，中文首分片 `data/zhwiki/000_00000.parquet`。官方数据卡说明其来源是 2025 年 8 月 Wikipedia Enterprise HTML 快照，上游已经移除参考资料、注释、外链和导航等部分，因此本文的“完整原文”只指本次冻结的 FineWiki `text` 字段。[官方固定版本数据卡](https://huggingface.co/datasets/HuggingFaceFW/finewiki/blob/8bd13e72e6a002407649b3e898535f42ceb1aeb9/README.md)

FineWiki 的整理文本按 **CC BY-SA 4.0** 发布。包内 Wikipedia/FineWiki 正文、索引片段和引文保留这一许可与原作者归属，不因存入代码仓库而改成仓库代码许可证。再次分享或改编这些文本时，须保留署名和来源、链接许可、说明修改，并遵守相同方式共享要求。[Wikimedia dump 许可说明](https://dumps.wikimedia.org/legal.html) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

`corpus/finewiki-zh-5000/manifest.jsonl` 为每页保留 title、page_id、URL、修订 version、修改日期和原文 SHA；索引记录也保留这些身份。可通过页面 URL 的历史及相应修订确认原作者归属。我们的处理是按固定顺序抽取 5000 篇、逐字切块并添加偏移/哈希/父级上下文元数据，未将正文归一化或冒充自创文本。源卡、清单与 freeze 原字节一并留存。官方许可页于 2026-09-08 复核；未打包图片或视频。
