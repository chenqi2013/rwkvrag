# StateTune 方法参考

本页是稳定的方法入口，不表示正在训练某份 State。当前实验与训练状态见 [CURRENT](CURRENT.md)。

## 可复用的方法

1. 从真实调用保存准确 prompt、原文、版本和错误输出，定位发生问题的阶段。
2. 只依据该调用实际可见的材料标注目标，区分否定答案、未知信息和无证据。
3. 在训练前复核标注，按实体和模板族隔离训练/验证；已用来分析错误的材料不再算未见测试。
4. 绑定底模、State 布局、tokenizer、提示模板、训练参数与数据哈希。
5. 同条件比较 zero State 与训练后 State，保留所有失败，不用训练完成代替效果验收。

工具入口见 [statetune/README](../llamaindex-retrieval/statetune/README.md)。后续新 Reader 的变量控制见 [EXPERIMENTS](EXPERIMENTS.md)。

## 2026-09-11 的具体经验

当时的 2000 条数据、17 个纠错种子、六组 State、训练过程和得失完整保存在[历史经验报告](archive/2026-09/statetune-experience-20260911.md)。这些数字和成绩只属于那一轮，不能直接用于判断后续问题。

旧权重和原始资料的恢复方式见[实验附件](artifacts.md)。本页路径保留，方便历史脚本和冻结文档找到已归档的详细记录。
