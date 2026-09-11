# WSL 本地问答

本地API地址为 `http://127.0.0.1:18440`，交互接口文档为 `/docs`，RAG调用为 `POST /v1/ask`。页面入口为 <http://127.0.0.1:18440/admin/#/search>，已有前端用于实际试用。

使用 **RWKV7 G1j 2.9B + OpenSearch BM25 + MongoDB**，没有embedding，服务器推理使用物理GPU3。本轮训练和评测见[StateTune经验](../../../docs/statetune-experience.md)。2026-09-11测试后保留比较服务，未恢复原问答推理服务；前端试用配置将Reader设为reader-trace-450、Writer设为writer-trace-300，Planner保持zero；三阶段由同一比较推理服务按请求加载对应state，不启动第二个GPU模型服务。

## 本地服务

四个专用 systemd 用户服务已经安装并启用，引用本目录内的配置原件。它们使用原有数据目录；WSL 用户服务管理器运行时会启动，并在进程异常退出后重启。SSH 隧道独立于其他任务的共享 SSH 连接。

```bash
systemctl --user start rwkvrag-api.service
systemctl --user status rwkvrag-api.service rwkvrag-mongodb.service rwkvrag-opensearch.service rwkvrag-tunnel.service
journalctl --user -u rwkvrag-api.service -n 100 --no-pager
```

停止本地服务（不停止远端 GPU）：

```bash
systemctl --user stop rwkvrag-api.service rwkvrag-mongodb.service rwkvrag-opensearch.service rwkvrag-tunnel.service
```

- API：`127.0.0.1:18440`，配置 `data/services/local-app/settings.json`。
- MongoDB：`127.0.0.1:18439`，数据 `data/services/mongodb/data`。
- OpenSearch：`127.0.0.1:18438`，数据 `data/services/opensearch/data`。
- GPU3 推理转发：`127.0.0.1:18423`。

本地服务启动后，需等 MongoDB、OpenSearch 恢复完成；`/v1/admin/health` 检查两者，模型就绪另看 `http://127.0.0.1:18423/health`。`active` 本身不证明模型已加载。

“验收资料”知识库包含人工构造材料。原5,000篇Wiki索引保留用于大库测试；`rwkvrag-local-use-v1`是另一个知识库。

## GPU3 服务

远端 `/home/chase/rwkvrag/data/services/rwkv-gpu3/settings.json` 绑定模型与推理源码 SHA。服务脱离 SSH 会话运行，没有之前实验的 90 分钟或 256 次调用限制；逐请求保存原始 token 收据。服务器重启后需要手动启动，不更改远端其他用户服务的启动策略。

```bash
ssh rwkv-8222 '/home/chase/chase/RWKV-PEFT/.venv/bin/python /home/chase/rwkvrag/llamaindex-retrieval/deploy/local/manage_gpu3.py start'
```

`start` 检查已登记进程的完整命令及启动时间，已有服务时不会启动第二份；新进程加载模型需数分钟。GPU3 被其他进程占用时启动会失败，绝不停止其他负载或改用其他显卡。日志在远端 `data/services/rwkv-gpu3/console.log`。

要释放本项目的 GPU3 服务：

```bash
ssh rwkv-8222 '/home/chase/chase/RWKV-PEFT/.venv/bin/python /home/chase/rwkvrag/llamaindex-retrieval/deploy/local/manage_gpu3.py stop'
```

`stop` 只给身份完全匹配的已登记进程发送 SIGTERM。数据、模型和历史收据保留。

Reader协议取决于选用的配置，合法输出仍可能选错原文，需结合来源核对。

服务代码同时校验2.9B模型名、权重SHA及32层/40头布局，不能仅修改配置便换成其他模型。

## 当前页面试用

打开 <http://127.0.0.1:18440/admin/#/search>，知识库过滤选择“StateTune 数据构建与训练方法”。这里导入的是项目正式方法文档，模型仍需通过BM25检索、Reader判断和Writer生成；没有预置这些问题的答案。

独立问题之间点击“开始新对话”；连续追问则保留当前会话。可以依次试：

1. 这轮StateTune的训练数据一共多少条？Writer、Reader和Planner分别多少条？
2. 实际使用哪个项目训练？正式入口脚本是什么？
3. 纠错种子怎样从真实trace中生成？错误输出会进入训练target吗？
4. Writer100、Writer300、Writer600、Writer1400各做了多少次优化器更新？
5. 为什么只有“速度是传统CAN的5倍”不能回答绝对速率是多少Mbit/s？
6. 本轮训练一共花了多少电费？
7. 连续追问：先问“分别介绍Writer、Reader和Planner的输入与输出格式”；再问“更正，只保留Reader和Planner，不需要Writer，也不需要训练参数”。

核对依据：第1题应为2000、1400、450、150；第2题应提到RWKV-PEFT与train_trace_state.py；第4题应为50、150、300、700；第6题没有资料依据，应明确不能确定。历史更正题应只保留最后要求的两个阶段。

实际页面试用中，第1题总量答对，但分阶段数量错答成155、48、36，还生成了错误链接。该输出已保留，页面连通与state加载正确不代表答案正确。详见[UI-VALIDATION.json](../../../artifacts/statetune-20260911/UI-VALIDATION.json)。
