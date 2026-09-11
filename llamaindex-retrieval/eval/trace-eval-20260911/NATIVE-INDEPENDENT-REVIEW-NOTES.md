# 完整RAG独立裁决补注

以下为根代理转录 `/root/v3_semantic_review` 的独立阅读结论，不是认证签名。79项裁决及161个输入/输出文件绑定见 `NATIVE-INDEPENDENT-REVIEW.json`。native Writer档仅指Writer1400。

- Planner state的15/15请求全部解析或长度失败，实际采用原问题fallback。新题收益不能归因为更好的模型规划；fallback保留完整对象名，修复了部分原Planner丢对象的查询。
- Reader state修复了多项漏选，也选入标题、空白和无关片段。组合组运动员、网球日期、学校事故日期三题值正确，却引用只含标题的资料1；事实实际在资料2。
- held_004：zero/bm的“64位元（byte）”自相矛盾；writer/reader/combined明确写64位元。writer的绝对5Mbit/s错引只含“5倍”的资料1；combined直接以5倍回答绝对速率。planner空证据编8/12位、8/12字节、1.5Mbit/s及额外线路条件。
- held_010：zero/writer的语言和事件模型正确，书色及五阶段错误，资料3/4不存在。reader正确给五语言、事件模型、16阶段，但语言列表引到IEC61508资料6；书色只复述问题，且把62056写62066。combined只实答事件模型。bm只有IEC61508证据；语言/事件虽符合完整库原文，实际引用不支持，书色错误。
- held_019：zero/reader/planner编工作站20并发、支持Mac，漏Daytona并循环到length；writer/combined反称“文件服务器版，不是工作站版”，只重复字段名到length，三个值均缺失。
- held_021：zero/reader选错船、错配日期并循环；writer列五艘而非两艘，把服役/铺龙骨日期当鱼礁日期；planner编“两艘红雪松”和年末日期；combined只循环标题和不存在编号。
- 原bm的019/021在OpenSearch发生too_many_nested_clauses异常，Writer未调用，不能写成模型拒答。
- fresh_6069162：zero/writer/planner空证据编0/3人。reader虽然拿到至少26原文，仍用“未撤离24人”回答死亡下限并循环。combined的“下限26人”保留了下限含义，引用实际支持。
- fresh_8051473：zero/reader回答其他城镇419人。writer明确说佩尔讷莱布洛涅419人，该句本身及引用真实，但没有回答伟大十月镇，记缺失而非同一对象数值错误。
- fresh_8313714：zero只循环字段；writer留空姓名；planner仅说“一名男子”且无证据。combined姓名正确，错引标题。
- fresh_1242294、fresh_924568：combined正确日期错引标题。924568 zero还编2024日期、时间和调查等内容；reader追加约10时有原文支持。
- fresh_223830：zero/planner引用完整决赛比分行，冠军可直接识别；额外对手/比分均有据，计可用。
- fresh_3234125：zero/planner正确“零特征”但空证据、虚构资料2、重复至length；writer误答有限域。
- fresh_1270249：reader首条日期/资料1正确，随后将相同日期归于五条不支持的资料，不计有据可用。
- fresh_9042440：原Planner检索式丢人物；zero/reader编0，writer编1。planner fallback后32颗及额外彗星、命名事实均有据。
- fresh_8903072：zero/reader复制提示词和JSON证据；167.17虽出现在被复制内容中，但没有形成答案。zero还循环至length，不能靠字符串命中判通过。
- fresh_2794356：zero户部正确但空证据、虚构编号并循环；writer没有部门值；reader户部及引用正确，另有重复和四行空字段占位，记可用但不记严格简洁格式通过。

可用口径允许可辨认且确有支持的非标准引用。因此Writer档4/11可用中有3条使用`[资料 1: source-id]`，严格引用语法只有1/11；不把格式偏差与事实错误、无据断言等量计分。Reader八旗题的冗余也单独记录。
