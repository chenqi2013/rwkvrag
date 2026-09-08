# 原生 RWKV Wiki 开发 smoke

这里的 8 题来自已经公开的开发题，包含 29 个必需事实，问题和聊天历史为合成场景，证据取自固定的 5000 篇真实 FineWiki 文章。这不是新盲测，也不是自动生成问题后即可声称检索有效的演示。

`run_wiki.py` 运行 Planner → OpenSearch BM25 → Resolver → Writer，并保存完整原始输入、模型/索引 HTTP、阶段收据、源码与索引身份。oracle 只留作离线审阅，不发送给模型。索引必须绑定具体名称和 UUID；运行器只执行索引读取，结束后再查身份、文档数和版本状态。失败、length 和空证据题保留原分母；不从 thinking 补算答案，不修改模型最终文字。

首批 `wiki-smoke-v1` 中，7/8 Planner 达到 1024 token 上限；仅 1 题进入检索。第二批 `wiki-smoke-v2` 仅改变 Planner 的 thinking prefill，8 个 Planner 均完成，但 192 个 Resolver 中仍有 21 个 length。**两批完整通过均为 0/8**；第二批仅 2/29 必需事实正确，不能把阶段完成率当作问答质量。

完整失败记录、固定材料对照、索引全量 scroll、源码快照、逐项语义审阅和离线复算方法见 [第一轮公开归档](../bm250820-rebuild-20260908/README.md)。当前工作树继续开发，重现这两批必须使用归档中的冻结源码与依赖，不能拿当前 parser/prompt 回写旧成绩。后续未完成实验不计入此页结果。

运行器提供三种明确模式：

- `--dry-run`：冻结输入和源码，不发网络请求。
- `--preflight`：只读索引预检，不调用模型。
- `--run`：实际检索和模型生成，输出目录必须是新路径。

可先在模块根目录检查参数：

```bash
python eval/native-wiki/run_wiki.py --help
```

运行前需要原生 RWKV completion/tokenize 服务、独立 OpenSearch 索引、准确模型名及冻结索引 UUID/记录。`--base-url`、`--opensearch-url`、`--model` 和 `--out` 由实际环境填写；认证值通过 `--api-key-env` 指定的环境变量传入，不写入公开命令或收据。不要将本机地址或旧绝对路径当成可移植服务配置。

本轮索引有 5000 篇、45,960 块，全部原始字符和来源偏移已核验；它与旧分支的分块数不同。没有每文档 2 块上限，没有调用向量 embedding。结构块超过软窗口时保留原文，硬 token 预算失败如实记录。

原文归 **Wikipedia contributors**，FineWiki 整理归 **Guilherme Penedo / HuggingFaceFW**。固定数据集 revision `8bd13e72e6a002407649b3e898535f42ceb1aeb9`；正文及引文按 **CC BY-SA 4.0** 保留署名、页面 URL 和修订号。FineWiki 上游已移除部分参考/导航内容，“完整”指冻结 `text`，不指完整 Wikipedia HTML。[官方数据卡及许可](https://huggingface.co/datasets/HuggingFaceFW/finewiki/blob/8bd13e72e6a002407649b3e898535f42ceb1aeb9/README.md) · [详细归属说明](../bm250820-rebuild-20260908/README.md#原文来源归属与许可)
