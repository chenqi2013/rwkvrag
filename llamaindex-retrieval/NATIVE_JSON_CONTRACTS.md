# 普通链路的原生 JSON 约束解码

**实现状态：应用接入及离线合同测试完成；默认关闭，尚未完成新配置的真实推理/语义验收。** 不是新问答旁路，不恢复旧规则模块，不修改模型原始输出。遵守[架构规则](ARCHITECTURE_RULES.md)。

## 接口与接入点

vllm-rwkv 的 `/v1/completions` 接受顶层 `structured_outputs.json`。OpenAI SDK 中使用 `extra_body={"structured_outputs": ...}`；直接 HTTP 不再包一层 `extra_body`。`response_format.type=json_schema` 是另一种接口形式；应用选择前者，不同时发送两套约束。

下面只是 HTTP 字段示例，完整请求的模板、token预算、模型身份和采样参数仍由 Native 客户端负责：

```json
{
  "structured_outputs": {
    "json": {
      "type": "object",
      "properties": {"answer": {"type": "string", "enum": ["YES", "NO"]}},
      "required": ["answer"],
      "additionalProperties": false
    }
  }
}
```

`native_json_contract.py` 从当前解析合同生成Schema；没有题目答案、实体名单或gold证据：

| 调用目的 | 约束 |
| --- | --- |
| 历史改写 | 唯一`question`字符串，非空白，最多8000字符 |
| 查询规划 | 唯一`queries`数组，1至`native_max_queries`项，每项非空白且最多2000字符 |
| 二值Reader | 唯一`answer`，只能为`YES`或`NO` |
| Writer | 本版本不改写作协议，不施加JSON Schema |

历史改写和Planner虽共享`planner`传输阶段，仍由各调用点**显式**传入不同Schema，不根据提示内容猜用途。普通链路可以单独选择既有`StructuredNativeRWKVClient`，不必启用typed funnel。Schema字段顺序保留，HTTP请求及trace记录完整约束；请求之间使用既有ContextVar隔离。

## 显式配置

在已经配置好的native G1K连接上设置：

```dotenv
RWKVRAG_NATIVE_JSON_CONSTRAINTS=schema_v1
RWKVRAG_NATIVE_PLANNER_FORMAT_REPAIR=false
RWKVRAG_NATIVE_RESOLVER_FORMAT_REPAIR=false
```

此profile要求已有配置同时满足：

- native传输、`g1j_plain`模板、模型身份检查开启、三个阶段使用闭合no-think前缀。
- `fact_queries_v1` Planner；`binary_query` Reader、individual分组、queries任务来源。
- single Writer、非task-matrix；无自动格式重试。

`native_json_constraints`默认`disabled`，不迁移现有设置。Schema缺失/不支持的组合在配置校验时拒绝；端点拒绝约束不换provider、不重发无约束请求。显式profile中的Planner HTTP/长度/解析失败会返回`planner_failed`，不继续使用原问句作为失败兜底计划。Reader NO仍不保留证据，程序不强制YES。

`schema_v1`只负责上述三类JSON调用，不声明外部auto路由模型或自由正文Writer也受此Schema约束。State配置、模型、prompt、检索预算、语义解析规则和Writer输出保持原配置。

## 服务端兼容性门槛

启用前必须验证**实际模型Tokenizer＋实际约束后端＋这三类完整Schema**。OpenAPI声明支持、单独构建Grammar成功，都不足以证明它们能一起运行。

已发现的RwkvTrie组合问题：`pattern`与字符串长度边界同时存在时，服务的`auto`可能选择Guidance，而其内置HF tokenizer加载器不认识RwkvTrie，返回HTTP500。单独设置`--structured-outputs-config.backend guidance`不能修复这个Tokenizer问题；强制XGrammar也不能绕过其不支持的组合约束检查，不能通过删长度/非空白约束冒充原合同通过。

该应用开关不会安装服务端兼容适配。需要服务侧经过验证的逐字节Tokenizer适配，保持原输入token IDs、EOS、非法/未使用ID屏蔽，并测试实际mask及合法/非法边界。当前兼容候选仍未完成全量及语义验收，默认保持关闭。若端点拒绝请求，保留错误，不自动改成无约束调用。

## 能保证什么，不能保证什么

支持相应Schema的后端在正常完成生成时，可以约束字段、类型、枚举、数组数量等。程序仍保留严格解析：backend错误、长度耗尽、超时、外部停止串等不构成合法完成；不补右括号、不删除超额查询、不修原始JSON。

合法JSON仍可能指代错误、丢条件、选错证据或引用不支持结论。`{"answer":"YES"}`合法不代表证据有用；queries不超过6条也不代表覆盖全部要求。格式与语义分别评估，不能将这项接入称为总体质量修复。

## 验证与下一步

离线测试覆盖普通历史→Planner→Reader的实际wire、不同Schema、参数/prompt保持、Writer原始输出、不支持配置、错误响应不降级、NO/非法选择不补证据；已有并发Schema及State隔离测试继续保留。这些不是新引擎推理或语义成绩。

真实A/B须另建冻结版本，先统一两臂的自动格式修复设置，再只改变Schema开关。不要与自动重试开启的旧端到端结果混称单变量对照。报告结构有效、正常结束、原任务覆盖、过选/漏选、逐句来源支持及耗时；失败/未运行保留。

语义架构的后续候选应与本次解码改动分开：可追溯任务及条件表示、任务与检索提示分离、正文证据/标题上下文分离、模型逐项输出断言与受限证据ID。新Writer协议需要单独的API/UI设计与回归，不以程序拼接或修引用冒充模型原答。已有[任务锚定候选](G1K_ARCHITECTURE.md)不因这项工程接入而自动通过语义验收。
