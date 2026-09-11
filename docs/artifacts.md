# StateTune实验附件与文件整理

正式经验只记录问题来源、纠错数据、训练方法和效果，见[StateTune经验](statetune-experience.md)。本页说明原始记录和权重的存放位置。

## 下载与校验

[GitHub实验发布：statetune-2.9b-20260911](https://github.com/chenqi2013/rwkvrag/releases/tag/statetune-2.9b-20260911)提供两个附件：

| 附件 | 内容 |
|---|---|
| `statetune-records-20260911.tar.gz` | 47,362个文件：原始请求和输出、评测标签与复核、训练收据、数据草稿、语料快照、冻结源码和历史说明 |
| `statetune-states-20260911.tar.gz` | 本轮六份最终state：Writer100/300/600/1400、Reader450、Planner150 |

这是实验结果发布，不表示全部场景已达标。Writer300只在固定材料抽样中较好；完整RAG没有比较其余三个Writer训练规模。Planner存在格式退步。

文件大小和SHA见[ARCHIVES.json](../artifacts/statetune-20260911/ARCHIVES.json)。每个归档都有逐成员路径、类型、大小及SHA清单，打包后已逐文件解压读取并校验。六份state与训练收据的对应关系见[STATES.json](../artifacts/statetune-20260911/STATES.json)。

在仓库根目录执行：

```bash
mkdir -p data/downloads/statetune-20260911
gh release download statetune-2.9b-20260911 --repo chenqi2013/rwkvrag \
  --pattern '*.tar.gz' --dir data/downloads/statetune-20260911
python3 artifacts/statetune-20260911/verify_archives.py data/downloads/statetune-20260911
```

校验通过后，解压到一个新建的空目录。归档保留仓库相对路径；不要覆盖当前代码，因为其中包含历史源码、前端和已过时文档。

```bash
mkdir data/restored-statetune-20260911
tar -xzf data/downloads/statetune-20260911/statetune-records-20260911.tar.gz \
  -C data/restored-statetune-20260911
tar -xzf data/downloads/statetune-20260911/statetune-states-20260911.tar.gz \
  -C data/restored-statetune-20260911
```

当前正式数据也保留在Git的 `llamaindex-retrieval/statetune/datasets/trace-v1-2000/` 中。归档中的原始回执、哈希、时间和当时状态不追写成最新结论，例如训练完成回执中的“尚未评测”描述的是训练结束时点。

## 复现实验的边界

记录附件包含数据生成所用的5,000页FineWiki快照、17条纠错种子、各版本数据、冻结RWKV-PEFT训练源码、官方推理源码，以及本轮预检、训练和评测执行资料。来源URL、许可证与原始声明随资料保留；它们不改变各上游文件的使用条款。

底模不包含在附件中。必须另备 `rwkv7-g1j-2.9b-20260831-ctx16384.pth`，SHA256为 `966f3420f833532aae3fb1fd6326533b08d43d23b7b03eaa2f0694a30b64a239`。Python/CUDA环境和正在运行的MongoDB/OpenSearch数据目录也未打包；历史脚本中的服务器绝对路径、GPU检查、时限和配置绑定需要按新环境重新准备，不能直接将旧执行配置当成启动命令。

旧阶段权重保留在本地，明确列在[EXCLUDED-STATES.json](../artifacts/statetune-20260911/EXCLUDED-STATES.json)，不属于本次附件。历史tar包内部可能另有当时已公开的小型state；本次没有将未发布的旧训练检查点补齐，不能声称附件可恢复所有历史权重。

## 本次清理

维护入口收敛为根README、后端说明、StateTune经验和数据入口。重复阶段报告、旧草稿和原始调用移出活动目录，保存在已校验附件及本地归档中；原有Git历史保持不变。具体路径见[CLEANUP.json](../artifacts/statetune-20260911/CLEANUP.json)。

旧Go原型已移除。现有前端根据最新试用要求恢复，连接同一套2.9B RAG后端；实验记录与过时文档的清理保持有效。后续部署验证单独记录，不改写历史测试回执。

必要的小型回归材料保留在Git。发布时测试结果及历史失败对照见[VALIDATION.json](../artifacts/statetune-20260911/VALIDATION.json)，没有把已知失败改成跳过或声称全量测试通过。
