# 5000篇中文Wiki：RWKV 2.9B RAG试用与参考问答

本项目使用RWKV7 G1j 2.9B与OpenSearch BM25，不使用embedding。现有管理页来自父分支`bm250820`，页面通过`POST /v1/ask`调用真实检索、Reader证据判断与Writer生成。

如果主要想学习StateTune，请先看[数据集如何构建、用什么训练](statetune-experience.md)，再看[正式训练数据与代码](../llamaindex-retrieval/statetune/README.md)。本页用于实际试问和核对来源。

## 在已部署的本地环境试用

1. 打开 <http://127.0.0.1:18440/admin/#/search>。
2. 知识库选择 **FineWiki 中文百科（5000篇）**。独立问题之间点击“开始新对话”。
3. 输入下表问题，比较实际回答与参考答案，并展开来源核对原文。
4. 在“文档管理”选择同一知识库，可以分页查看全部5000篇文章、查看切片或下载原文。

`127.0.0.1`只指向打开浏览器的那台电脑；分享GitHub链接不会把当前本地页面变成公共在线演示。其他电脑需要依照[服务说明](../llamaindex-retrieval/README.md)部署，并准备2.9B模型、OpenSearch和MongoDB。

当前试用配置：Planner为zero state，Reader为`reader-trace-450`，Writer为`writer-trace-300`，由同一个推理服务按阶段加载。这个组合尚未证明是端到端最优。

## 数据规模与接入方式

| 范围 | 文章数 | 原文片段数 |
|---|---:|---:|
| 本次接入的FineWiki知识库 | 5000 | 45960 |
| 原有小型知识库合计 | 19 | 91 |
| 接入后页面检索索引合计 | 5019 | 46051 |

这里的5000是可检索文章数；2000是之前构建的StateTune训练样本数；几十或几百是不同评测子集的题数。三者不是同一计数。

接入复用原索引`rwkvrag-bm250820-wiki-5000-20260908`的45960个片段，逐条验证Unicode原文位置和SHA256，然后复制到页面索引`rwkvrag-local-use-v1`，补齐MongoDB知识库及文件记录。原索引保留，原有页面片段未变更。下载文件放在独立的每文件目录，避免管理页删除单篇文件时影响共享语料目录。文件列表已改为服务端分页，按创建时间和ID稳定排序。

[接入脚本](../llamaindex-retrieval/deploy/local/connect_wiki.py)用于**已存在上述冻结语料及原索引的环境**，不是从零恢复OpenSearch的通用导入器。对应原文、manifest及冻结记录可从[实验附件](artifacts.md)恢复；附件不包含正在运行的数据库目录。现有环境在仓库根目录执行：

```bash
llamaindex-retrieval/.venv/bin/python llamaindex-retrieval/deploy/local/connect_wiki.py \
  --settings data/services/local-app/settings.json \
  --corpus data/corpora/finewiki-zh-5000 \
  --output data/publication/wiki-ui-20260911
```

## 13道测试题与参考答案

下列答案依据**本次冻结语料**人工整理，不是模型已经答对的承诺。部分主题曾用于训练或旧评测，这份题单用于人工试用，不能当作独立盲测准确率。只向问答接口传问题和必要历史，不把答案表作为提示或导入知识库。

每题的来源版本、原文摘录、Unicode起止位置、摘录SHA及全文SHA保存在[questions.jsonl](../llamaindex-retrieval/eval/wiki-ui-20260911/questions.jsonl)。参考答案是对原文的中文整理，原文摘录保持不变。

| 题号 | 测试问题 | 参考答案 | 来源版本 |
|---|---|---|---|
| 1 | CAN FD 是谁开发的？哪一年开始发展，哪一年推出？ | 罗伯特·博世公司；2011年发展，2012年推出。 | [CAN FD](https://zh.wikipedia.org/w/index.php?oldid=82332849) |
| 2 | CAN FD 支持哪两种长度的消息 ID？每帧实际数据量最多多少字节？ | 11位和29位消息ID；每帧实际数据量最多64字节。 | [CAN FD](https://zh.wikipedia.org/w/index.php?oldid=82332849) |
| 3 | 依据《CAN FD》条目，扩展后的资料率是多少 Mbit/s？是原标准的多少倍？ | 5 Mbit/s，是原CAN标准的5倍。 | [CAN FD](https://zh.wikipedia.org/w/index.php?oldid=82332849) |
| 4 | IEC 61508 的安全生命周期包含多少个阶段？分析、实现、运作及维护各对应哪些阶段？ | 共16个阶段：1—5分析，6—13实现，14—16运作及维护。 | [IEC 61508](https://zh.wikipedia.org/w/index.php?oldid=85921036) |
| 5 | Windows NT 3.5 的发布日期和停止支持、更新的日期分别是什么？ | 1994年9月21日发布；2001年12月31日停止支持和更新。 | [Windows NT 3.5](https://zh.wikipedia.org/w/index.php?oldid=83713963) |
| 6 | Windows NT 3.5 工作站版允许多少个客户端并发访问文件服务器？支持 Mac 客户端吗？ | 允许10个客户端并发访问；不支持Mac客户端。 | [Windows NT 3.5](https://zh.wikipedia.org/w/index.php?oldid=83713963) |
| 7 | TG100 型柴油机车由哪家工厂设计制造？哪一年开始研制，哪一年试制首台？ | 卢甘斯克机车制造厂；1958年开始研制，1959年试制首台。 | [TG100型柴油机车](https://zh.wikipedia.org/w/index.php?oldid=74891122) |
| 8 | TG100 型柴油机车为什么没有投入批量生产？试验完成后移交哪里，作什么用途？ | 高速柴油机及液力机械传动装置故障较多，因此未获批准批量生产；移交第聂伯罗彼得罗夫斯克运输工程研究所作研究及培训用途。 | [TG100型柴油机车](https://zh.wikipedia.org/w/index.php?oldid=74891122) |
| 9 | 克赖沙的面积和人口分别是多少？人口统计截至哪一年？ | 面积28.92平方千米；截至2020年人口4533人。 | [克赖沙](https://zh.wikipedia.org/w/index.php?oldid=70993679) |
| 10 | 仅依据《克赖沙》这篇资料，2026年人口是多少？ | 资料没有2026年人口，无法确定；文中只给出截至2020年的4533人，不能当作2026年的数值。 | [克赖沙](https://zh.wikipedia.org/w/index.php?oldid=70993679) |
| 11 | 张天赋的《Loser》何时发行，由哪家公司发行？ | 2021年8月26日，由香港华纳唱片数字发行。 | [Loser (张天赋歌曲)](https://zh.wikipedia.org/w/index.php?oldid=83294499) |
| 12 | 张天赋的《Loser》音乐录影带由谁执导，主要在哪里取景？ | 张蔓姿执导，主要在茶餐厅内取景。 | [Loser (张天赋歌曲)](https://zh.wikipedia.org/w/index.php?oldid=83294499) |
| 13 | 张天赋的《Loser》在歌曲列表中的时长是多少？ | 4分13秒（4:13）。 | [Loser (张天赋歌曲)](https://zh.wikipedia.org/w/index.php?oldid=83294499) |

第10题用于检查缺失资料处理：只给2020年人口的资料，不能支持2026年人口。第4题要区分“16个生命周期阶段”与“标准由7个部分组成”。第2、3题要区分位、字节、资料率和倍数。

## 实际验证

接入后逐页调用文件接口，确认5000个唯一文件ID、全部ready、片段数合计45960；抽查以上6篇文章的下载SHA与冻结原文一致。浏览器检查第1、2、500页均返回10篇文章。

通过现有页面实际提交3题，均返回HTTP 200、无浏览器脚本错误，Planner/Reader/Writer使用的state与配置一致；模型回答未被代码改写。

| 题号 | 页面实际结果 | 核对结论 |
|---|---|---|
| 9 | 面积28.92平方千米、人口4533、截至2020年，并引用资料1 | 事实与引用依据正确 |
| 6 | 允许10个客户端并发访问文件服务器，不支持Mac，引用资料1 | 事实与引用依据正确 |
| 10 | `2026年人口数据不可用。[资料 2]` | 没有编造数值，但返回来源为空，资料2为无效引用；不能算完整通过 |

原样答案、调用阶段、来源与引用审计保存在[BROWSER.json](../llamaindex-retrieval/eval/wiki-ui-20260911/BROWSER.json)。这3次检查只证明本次接入后的具体表现，未重新评测全部13题或所有5000篇文章，也不代表整体准确率。

前端构建通过，仍有已有的大包体提示。后端全量测试为761通过、114失败、2跳过，114个失败用例ID与此前记录一致；不能称为全量测试通过。接入收据与验证记录见[本次验证目录](../llamaindex-retrieval/eval/wiki-ui-20260911/)。

## 来源与许可

语料来自HuggingFaceFW/finewiki的中文Wiki快照，数据集revision为`8bd13e72e6a002407649b3e898535f42ceb1aeb9`。每篇保留Wikipedia历史版本URL。原文由Wikipedia contributors贡献，按CC-BY-SA-4.0使用；本页摘录及依据原文整理的参考答案沿用该许可并保留署名、版本链接。冻结文章可能落后于现状，本题单只检查是否忠实回答所给资料。
