# Wiki检索回归

`fixtures.jsonl`保存8道已暴露开发题；`run_wiki.py`执行规划、BM25检索、Reader与Writer，并保存输入、来源、请求与原始输出。判据不发送给模型，运行完成不等于答案正确。

运行器提供 `--dry-run`、`--preflight` 和 `--run`。真实执行需要按环境填写模型地址、OpenSearch地址、索引名与UUID；参数见 `python eval/native-wiki/run_wiki.py --help`。当前2.9B完整对照及历史源码见[实验附件](../../../docs/artifacts.md)。

固定语料来自5,000篇FineWiki文章，原文归Wikipedia contributors，整理归Guilherme Penedo / HuggingFaceFW；正文按CC BY-SA 4.0保留署名、页面URL和修订号。数据revision为 `8bd13e72e6a002407649b3e898535f42ceb1aeb9`，来源与许可声明随附件保留。
