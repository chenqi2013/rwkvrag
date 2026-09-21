# R0重复性执行登记（2026-09-21）

## 问题与范围

验证相同历史输入、初始State、top_k=1和固定seed下是否仍出现输出漂移。用户授权继续受控验证；不训练，不改协议/Writer，不切换生产。

20道已见定位题的zero组442个调用：plan20、observe261、assess141、writer20，各完整执行三轮，共1,326次。按照index、原started_at、call_id排序，每轮同顺序，并发1，不去重、删失败或挑最好轮次。历史320份结果仍保留；本轮只是阶段回放，不是20题重新走完整检索链路。

## 冻结输入与运行

INPUTS.json保存逐字messages、prompt、token序列、参数、原输出和源trace哈希。prepare.py逐条核对旧绑定并提取实际HTTP请求。唯一运行性替换是ASSESS已过期的State引用换成同文件的新引用，其余请求字段完全保留。非ASSESS无导入State，ASSESS读取固定zero State；从不写回State。

- 主机rwkv-8222/rwkv-82，machine-id bcd164d5ad3a4ab3b0790412e32e69f3。
- 物理GPU3，UUID GPU-a9570da2-547a-c2b3-0cab-7bbdc1a8a8b0；现有搜索路由服务不动。
- 引擎沿用用户本地快照，ENGINE-SOURCE.json SHA256 b8be05cd8dddcdb9eac88bca6af0ff2486982edd9407328fb6b064fd0c416168。启动逐文件核对引擎、模型元数据和六个权重分片。
- 基座rwkv7-g1j-7.2b-20260831-ctx16384，FP16计算、FP32 recurrent；16K上下文、max-num-seqs4、batched2048、prefix cache启用、async scheduling启用、显存比例0.30。
- zero State SHA256 2451b35fbca3c563739b375046e1219427614da6b0bc6909690c17f114ecbb66。上传及结束后均核对initial/processed0/pending0，完成后删除引用。
- 原请求top_k1、temperature1、top_p1、seed11及所有其它字段逐条保留。输出预算plan768、observe/assess512、Writer2048；不加宽。
- 服务新目录/home/chase/rwkvrag/data/services/layered-repeatability-20260921，18426端口，新unit rwkvrag-layered-repeatability-20260921。launcher只改旧部署记录目录，模型/运行参数不变；单独绑定新launcher哈希。
- 执行目录/home/chase/rwkvrag/data/experiments/layered-repeatability-20260921/run1。每次调用180秒，全局3小时，客户端0重试；HTTP失败、token不一致或响应投影不可靠立即停止并保存未完成分母。长度退出保留并继续。
- 启动前提交并上传源码、输入和PINS.json至GitHub；PINS绑定具体文件，部署记录保存实际模型分片和依赖版本。

## 判定规则与限制

本轮的预定义判据是三个重复的输入token、输出字节/输出token和finish reason一致性，另与历史结果分别比较。历史输出仅作重复性参考，**不是语义正确gold**；因此本R0不计算语义准确率，也不需要把旧答案标为正确。输出不同时保留全部轮次，逐项审读其语义影响，标注实现者评审、非独立评审。

只有1,326次全部完成、输入token全部一致才称执行完整。存在任何跨轮输出变化就报告重复性未通过；不使用99%阈值掩盖漂移。即使完全一致，也不能证明答案正确或所有硬件配置确定性。

新State对照R1仍需独立冻结141个同上游输入的语义期望及双轮交叉执行，不随R0脚本自动启动。若R0发现漂移，先定位再判断R1因果解释范围。R2/R3层级架构语义实验未包含在本次1,326调用内。
