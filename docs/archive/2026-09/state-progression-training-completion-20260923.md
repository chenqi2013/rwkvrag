# StateTune发布V1训练完成核验

2026-09-23核验8222物理GPU3的`rwkvrag-state-release-v1-train.service`：`Result=success`、`ExecMainStatus=0`、现为inactive。原始[完成收据](../../../artifacts/state-progression-20260922/train-release-v1/COMPLETED.json)SHA-256为`8abe88dec5a2a7bfe44ef827dd319bab956c8960b86d2d2bd51a3460be92d18d`，[数据绑定](../../../artifacts/state-progression-20260922/train-release-v1/DATA.json)确认训练输入SHA-256为`a7bb1adf1d21e47789b7b169d488d32e0e56924c226063877fe0b45a4f61aa89`，Resolver 1546条、Writer 2394条，合计3940条。旧评测题和答案没有进入训练发布集。

训练配置：7.2B、RWKV fp32io16、FP32循环State，冻结基座，只更新角色独立初始State；学习率`1e-5`、两轮、梯度累积4。Resolver每轮387次、两轮774次；Writer每轮599次、两轮1198次，共1972次。日志均无非有限浮点值，最终记录的损失及梯度范数有限。总耗时6652秒，进程最大预留显存45,992,640,512字节，低于50GiB限制；收据记录`base_unchanged=true`。最长7508-token样本的完整前后向预检损失有限，未用于更新前的性能结论。

| 角色 | 轮次 | checkpoint SHA-256 |
| --- | ---: | --- |
| Resolver | 1 | `19161783917ca424f30aca69eb4b3e91fcbcd3f9e32d2efce58312df787917ae` |
| Resolver | 2 | `713fb83a9efe4721c2ea829186166f94e4c6bdcb75579f30234aab71bef25d25` |
| Writer | 1 | `5d20fe74f2136370ad18516390c29328f984e39236339ea6ea6dae55a6a840e1` |
| Writer | 2 | `e820b7f0d908d2bb313e3f5b37e53cd191b442149f85bda9d60efb6d0d5461a1` |

四个远端文件实际SHA-256与完成收据逐项相同，状态格式在收据中均为FP32。二轮Resolver和Writer仅作为**待评测候选**；训练损失、正常退出、哈希一致均不能证明答案可靠、比较题提升、旧能力保持或联网检索改善。完成收据显式记录`quality_verified=false`和`production_promoted=false`。随后已启动[冻结评测](../../../artifacts/state-progression-20260922/EVAL-PINS-v1.json)的新24题批次；全部原始输出和语义复核完成前，正式服务不切换。
