# 恢复语料与可重建备份

- 上游：[HuggingFaceFW/finewiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki)，
  固定提交 `8bd13e72e6a002407649b3e898535f42ceb1aeb9`。
- 5 个中文分片共 5,526,531,060 字节，下载日志记录逐个 SHA256 验证。
- 选择后的 5,573 条完整 FineWiki 文章版本保留全部原字段及原始正文，
  包括原有 5,000 条背景记录；另加 4 篇固定版本 Wikipedia 原文。
- 最终 5,577 条版本记录、5,429 个文章 ID。全部旧 400 题对应文章的旧标识匹配。
  其余 34 题每题至少有对应文章；旧 `expected_titles` 中的备选标题不等于每个标题
  都存在。多对象比较的 `expected_titles_all` 已单独纳入选择。
- `complete-articles.tar.gz` 含完整选中语料、4 篇补充的原始 API/HTML/wikitext、
  Markdown、转换元数据、出处和哈希。5,594 个文件逐个解包读取验证通过。
  不含 5.5 GB 上游全量分片；可按保存的上游版本重新下载。

恢复脚本见 [v1](../../../llamaindex-retrieval/eval/corpus-restore-20260920/README.md)
及 [文件登记 v2](../../../llamaindex-retrieval/eval/corpus-restore-v2-20260920/README.md)。
`BUILD-V2.json` 是成功启动的独立部署：5,524 份唯一正文文件登记，仍保留全部
5,577 条文章版本和 58,594 个检索节点；所有节点逐字范围及哈希已对照完整原文校验。

v1 的启动失败日志、未发送任何模型问题的预检失败日志均保留。没有删除旧索引
或修改主应用；不是原删除索引的逐字重建，也不能沿用原检索排序成绩。

## Attribution and licensing

Article text is attributed to **Wikipedia contributors**, under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), with revision URLs,
article identities and provenance retained per manifest entry. FineWiki processing
is attributed to **HuggingFaceFW/finewiki**; exact upstream records are preserved.
Supplemental HTML-to-Markdown conversion is identified per entry (markdownify
1.2.3, heading style ATX). Preserve these notices and per-article source links when
redistributing the corpus. No authorship of the original articles is claimed.
