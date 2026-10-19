# 上下文与 State 边界

检索和 Reader 证据选择减少 Writer 输入。历史、任务和原文保存在应用数据层；各模型调用之间没有自动延续完整动态 State，也没有已验证的跨文档 State 合并。

训练 State 是读取 prompt 前的初始状态，不包含特定文档的阅读记忆。问题、历史或证据变化后须读取对应输入，不把不同前缀的状态当作可互换缓存。

输入预算按实际模板和 tokenizer 计算，并为输出预留容量。原文字符数、模型名和历史探针不能替代实际服务限制。应用配置见[服务说明](../llamaindex-retrieval/README.md)，初始 State 的绑定和兼容边界见[State 路由](../llamaindex-retrieval/NATIVE_STATE_ROUTING.md)。
